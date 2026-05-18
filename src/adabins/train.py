"""Train AdaBins on custom RGB-depth pairs with AMP/DDP/EMA/reporting."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import pprint
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from adabins.losses import SILogLoss
from adabins.model import UnetAdaptiveBins
from adabins.yaml_config import AdaBinsTrainConfig, build_train_arg_parser, train_config_from_args
from adabins.checkpoints import unwrap_state_dict
from adabins.dataset import CustomDepthPairDataset
from adabins.dist_helper import setup_distributed
from adabins.ema import ModelEMA
from adabins.log_utils import init_log
from adabins.metrics import eval_depth
from adabins.reporting import append_csv, plot_curves, save_json

METRIC_KEYS = ["d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog"]

# Autocast dtype for CUDA AMP. GradScaler is only intended for float16; do not combine with bfloat16.
# float16 + GradScaler is the default/stable path for this EfficientNet-heavy pipeline (pure bfloat16 autocast had NaNs on some runs).
AMP_AUTOCAST_DTYPE = torch.float16


def _distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _ddp_any(flag: bool, device: torch.device) -> bool:
    """Return True if ANY rank has flag=True (all-reduce MAX). No-op when not distributed."""
    if not _distributed():
        return flag
    t = torch.tensor(1.0 if flag else 0.0, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item() > 0


def _device_for_process(cfg: AdaBinsTrainConfig, local_rank: int, world_size: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if cfg.cuda_indices is not None:
        if _distributed():
            if len(cfg.cuda_indices) != world_size:
                raise ValueError("device.cuda_indices length must match WORLD_SIZE for DDP runs.")
            return torch.device(f"cuda:{cfg.cuda_indices[local_rank]}")
        return torch.device(f"cuda:{cfg.cuda_indices[0]}")
    return torch.device(f"cuda:{local_rank}" if _distributed() else "cuda:0")


def _is_better(current: float, best: float, mode: str) -> bool:
    return current < best if mode == "min" else current > best


def _median_scale_align(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Align prediction scale to target via median ratio for relative-depth eval."""
    pred_median = torch.median(pred)
    target_median = torch.median(target)
    scale = target_median / pred_median.clamp_min(1e-6)
    return pred * scale


def _load_model_state(path: str) -> dict:
    if path.startswith("url::"):
        import torch.hub
        blob = torch.hub.load_state_dict_from_url(path[5:], map_location="cpu", progress=True)
    else:
        blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "model" in blob:
        blob = blob["model"]
    if isinstance(blob, dict) and any(k.startswith("module.") for k in blob):
        blob = {k.replace("module.", "", 1): v for k, v in blob.items()}
    if not isinstance(blob, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return blob


def _probe_batch_size(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
    dataset: CustomDepthPairDataset,
    cfg: AdaBinsTrainConfig,
    logger: logging.Logger,
) -> tuple[int, int]:
    if device.type != "cuda":
        return cfg.bs, cfg.grad_accum_steps

    target_effective = cfg.bs * cfg.grad_accum_steps
    amp_enabled = cfg.amp and device.type == "cuda"
    max_by_data = len(dataset)

    def _fits(batch_size: int) -> bool:
        try:
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
            sample = next(iter(loader))
            img = sample["image"].to(device)
            depth = sample["depth"].to(device)
            valid_mask = sample["valid_mask"].to(device)
            model.train()
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=AMP_AUTOCAST_DTYPE):
                _, pred = model(img)
            with torch.amp.autocast("cuda", enabled=False):
                mask = valid_mask & (depth >= cfg.min_depth) & (depth <= cfg.max_depth)
                loss = criterion(pred.float(), depth.float(), mask=mask)
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return True
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)
            return False

    if not _fits(1):
        raise RuntimeError(f"Cannot fit batch_size=1 on {device}")
    low, high = 1, 2
    while high <= max_by_data and _fits(high):
        low = high
        high *= 2
    high = min(high, max_by_data + 1)
    left, right = low, high
    while left + 1 < right:
        mid = (left + right) // 2
        if _fits(mid):
            left = mid
        else:
            right = mid
    actual_bs = max(1, left)
    if _distributed():
        actual_bs = max(1, int(actual_bs * 0.7))
    grad_accum = max(1, target_effective // actual_bs)
    logger.info("Auto batch size selected=%d, grad_accum=%d", actual_bs, grad_accum)
    return actual_bs, grad_accum


@torch.no_grad()
def _run_validation(
    eval_model: torch.nn.Module,
    valloader: DataLoader,
    device: torch.device,
    cfg: AdaBinsTrainConfig,
) -> tuple[dict[str, float], float]:
    """Returns (metrics, n_valid_val_samples). If n==0, metrics are NaN and must not be treated as a real score."""
    eval_model.eval()
    sums = {k: torch.tensor([0.0], device=device) for k in METRIC_KEYS}
    nsamples = torch.tensor([0.0], device=device)
    for sample in valloader:
        img = sample["image"].to(device, non_blocking=True).float()
        depth = sample["depth"].to(device, non_blocking=True)[0]
        vm = sample["valid_mask"].to(device, non_blocking=True)[0]
        _, pred = eval_model(img)
        pred = F.interpolate(pred, depth.shape[-2:], mode="bilinear", align_corners=True)[0, 0]
        mask = vm & (depth >= cfg.min_depth) & (depth <= cfg.max_depth)
        if mask.sum() < 10:
            continue
        pred_eval = pred[mask].clamp(min=1e-6)
        depth_eval = depth[mask]
        if cfg.scale_align_eval:
            pred_eval = _median_scale_align(pred_eval, depth_eval).clamp(min=1e-6)
        if not torch.isfinite(pred_eval).all():
            continue
        cur = eval_depth(pred_eval, depth_eval)
        for k in METRIC_KEYS:
            sums[k] += float(cur[k])
        nsamples += 1

    if _distributed():
        torch.distributed.barrier()
        for v in sums.values():
            dist.reduce(v, dst=0)
        dist.reduce(nsamples, dst=0)

    n = nsamples.item()
    if n <= 0:
        return {k: float("nan") for k in METRIC_KEYS}, 0.0
    return {k: (sums[k] / n).item() for k in METRIC_KEYS}, n


def _save_checkpoint(path: Path, blob: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)


def _build_checkpoint(
    model: torch.nn.Module,
    ema: ModelEMA | None,
    optimizer: AdamW,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    previous_best: dict,
    best_metric_val: float,
    best_epoch: int,
    no_improve_count: int,
) -> dict:
    raw = model.module if isinstance(model, DDP) else model
    blob = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "previous_best": previous_best,
        "best_metric_val": best_metric_val,
        "best_epoch": best_epoch,
        "no_improve_count": no_improve_count,
    }
    if ema is not None:
        blob["ema"] = ema.state_dict()
    return blob


def train_main(cfg: AdaBinsTrainConfig) -> None:
    logger = init_log("global", logging.INFO)
    logger.propagate = 0
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

    if _distributed():
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA.")
        rank, world_size = setup_distributed(port=cfg.port)
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size, local_rank = 0, 1, 0

    device = _device_for_process(cfg, local_rank, world_size)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    save_root = Path(cfg.save_path)
    ckpt_dir = save_root / "checkpoints"
    log_dir = save_root / "logs"
    report_dir = save_root / "reports"
    fig_dir = save_root / "figures"
    if rank == 0:
        for d in (ckpt_dir, log_dir, report_dir, fig_dir):
            d.mkdir(parents=True, exist_ok=True)
        logger.info("%s\n", pprint.pformat({**asdict(cfg), "ngpus": world_size, "distributed": _distributed()}))
        logger.info("Device: %s", device)

    writer = SummaryWriter(log_dir) if rank == 0 else None
    cudnn.enabled = device.type == "cuda"
    cudnn.benchmark = device.type == "cuda"
    amp_enabled = cfg.amp and device.type == "cuda"

    trainset = CustomDepthPairDataset(
        cfg.train_list,
        "train",
        size=(cfg.img_size, cfg.img_size),
        depth_scale=cfg.depth_scale,
        list_root=cfg.list_root,
    )
    valset = CustomDepthPairDataset(
        cfg.val_list,
        "val",
        size=(cfg.img_size, cfg.img_size),
        depth_scale=cfg.depth_scale,
        list_root=cfg.list_root,
    )

    model = UnetAdaptiveBins.build(
        n_bins=cfg.n_bins,
        min_val=cfg.min_depth,
        max_val=cfg.max_depth,
        norm=cfg.norm,
        backend_name=cfg.backend_name,
        backend_pretrained=cfg.backend_pretrained and (cfg.pretrained_from is None and cfg.resume_from is None),
    )
    if cfg.resume_from:
        resume_blob = torch.load(cfg.resume_from, map_location="cpu")
        model.load_state_dict(_load_model_state(cfg.resume_from), strict=True)
    elif cfg.pretrained_from:
        model.load_state_dict(_load_model_state(cfg.pretrained_from), strict=False)

    model = model.to(device)
    criterion = SILogLoss().to(device)

    actual_bs, grad_accum = cfg.bs, cfg.grad_accum_steps
    if cfg.auto_batch_size:
        if rank == 0:
            actual_bs, grad_accum = _probe_batch_size(model, criterion, device, trainset, cfg, logger)
        if _distributed():
            info = torch.tensor([actual_bs, grad_accum], dtype=torch.long, device=device)
            dist.broadcast(info, src=0)
            actual_bs, grad_accum = int(info[0].item()), int(info[1].item())

    if _distributed():
        train_sampler = DistributedSampler(trainset, shuffle=True)
        val_sampler = DistributedSampler(valset, shuffle=False)
        model = DDP(
            model,
            device_ids=[device.index],
            broadcast_buffers=False,
            find_unused_parameters=cfg.find_unused_parameters,
        )
    else:
        train_sampler = val_sampler = None

    trainloader = DataLoader(
        trainset,
        batch_size=actual_bs,
        pin_memory=device.type == "cuda",
        num_workers=cfg.num_workers,
        drop_last=True,
        sampler=train_sampler,
        shuffle=not _distributed(),
    )
    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=device.type == "cuda",
        num_workers=cfg.num_workers,
        drop_last=False,
        sampler=val_sampler,
        shuffle=False,
    )

    raw_model = model.module if isinstance(model, DDP) else model
    optimizer = AdamW(
        [
            {"params": raw_model.get_1x_lr_params(), "lr": cfg.lr / 10.0},
            {"params": raw_model.get_10x_lr_params(), "lr": cfg.lr},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    use_grad_scaler = amp_enabled and AMP_AUTOCAST_DTYPE == torch.float16
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=use_grad_scaler)
    ema: ModelEMA | None = ModelEMA(raw_model, decay=cfg.ema_decay, device=device) if cfg.ema_decay > 0.0 else None

    start_epoch = 0
    global_step = 0
    best_metric_val = float("inf") if cfg.monitor_mode == "min" else float("-inf")
    best_epoch = 0
    no_improve_count = 0
    previous_best = {k: (0.0 if k in {"d1", "d2", "d3"} else 1e9) for k in METRIC_KEYS}
    history: list[dict] = []

    if cfg.resume_from:
        blob = resume_blob
        if "optimizer" in blob:
            optimizer.load_state_dict(blob["optimizer"])
        if "scaler" in blob and scaler.is_enabled():
            scaler.load_state_dict(blob["scaler"])
        elif "scaler" in blob and rank == 0:
            logger.info("Resume: omitting scaler state (GradScaler off for current AMP dtype).")
        if ema is not None and "ema" in blob:
            ema.load_state_dict(blob["ema"])
        start_epoch = int(blob.get("epoch", -1)) + 1
        global_step = int(blob.get("global_step", 0))
        best_metric_val = float(blob.get("best_metric_val", best_metric_val))
        best_epoch = int(blob.get("best_epoch", 0))
        no_improve_count = int(blob.get("no_improve_count", 0))
        if "previous_best" in blob:
            previous_best.update(blob["previous_best"])

    steps_per_epoch = max(1, (len(trainloader) + grad_accum - 1) // grad_accum)
    total_iters = cfg.epochs * steps_per_epoch

    for epoch in range(start_epoch, cfg.epochs):
        if rank == 0:
            logger.info(
                "===========> Epoch %d/%d | best %s=%.4f (ep%d) | patience %d/%s",
                epoch,
                cfg.epochs,
                cfg.monitor_metric,
                best_metric_val,
                best_epoch,
                no_improve_count,
                str(cfg.patience),
            )
        if train_sampler is not None:
            train_sampler.set_epoch(epoch + 1)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch}", disable=(rank != 0), dynamic_ncols=True)
        skipped_empty = 0
        skipped_nan = 0
        accum_progress = 0
        bn_modules = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        for i, sample in enumerate(pbar):
            img = sample["image"].to(device, non_blocking=True)
            depth = sample["depth"].to(device, non_blocking=True)
            valid_mask = sample["valid_mask"].to(device, non_blocking=True)
            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            mask = valid_mask & (depth >= cfg.min_depth) & (depth <= cfg.max_depth)
            if _ddp_any(mask.sum().item() == 0, device):
                skipped_empty += 1
                if accum_progress > 0:
                    optimizer.zero_grad(set_to_none=True)
                    accum_progress = 0
                continue

            saved_bn_mean = [m.running_mean.clone() for m in bn_modules]
            saved_bn_var = [m.running_var.clone() for m in bn_modules]

            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=AMP_AUTOCAST_DTYPE):
                _, pred = model(img)

            if _ddp_any(not torch.isfinite(pred).all().item(), device):
                for m, mean, var in zip(bn_modules, saved_bn_mean, saved_bn_var):
                    m.running_mean.copy_(mean)
                    m.running_var.copy_(var)
                skipped_nan += 1
                if accum_progress > 0:
                    optimizer.zero_grad(set_to_none=True)
                    accum_progress = 0
                continue

            with torch.amp.autocast("cuda", enabled=False):
                silog = criterion(pred.float(), depth.float(), mask=mask)
                # Match SILogLoss: pred is lower-res than GT; interpolate before L1 + mask indexing.
                pred_l1 = pred.float()
                if pred_l1.ndim == 4 and pred_l1.shape[1] == 1:
                    pred_l1 = pred_l1[:, 0]
                depth_l1 = depth.float()
                if depth_l1.ndim == 4 and depth_l1.shape[1] == 1:
                    depth_l1 = depth_l1[:, 0]
                if pred_l1.shape[-2:] != depth_l1.shape[-2:]:
                    pred_l1 = F.interpolate(
                        pred_l1.unsqueeze(1),
                        depth_l1.shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    ).squeeze(1)
                m = mask[:, 0] if mask.ndim == 4 and mask.shape[1] == 1 else mask
                n_valid = m.sum()
                l1 = F.l1_loss(pred_l1[m], depth_l1[m]) if n_valid > 0 else pred.new_zeros(())
                loss = (silog + 0.1 * l1) / grad_accum

            if _ddp_any(not torch.isfinite(loss), device):
                for m, mean, var in zip(bn_modules, saved_bn_mean, saved_bn_var):
                    m.running_mean.copy_(mean)
                    m.running_var.copy_(var)
                skipped_nan += 1
                if accum_progress > 0:
                    optimizer.zero_grad(set_to_none=True)
                    accum_progress = 0
                continue

            will_finish_window = (accum_progress + 1) == grad_accum
            sync_ctx = contextlib.nullcontext() if (not _distributed() or will_finish_window) else model.no_sync()
            with sync_ctx:
                scaler.scale(loss).backward()
            accum_progress += 1
            epoch_loss_sum += loss.item() * grad_accum

            if accum_progress == grad_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum_progress = 0

                lr = cfg.lr * (1 - global_step / max(total_iters, 1)) ** 0.9
                optimizer.param_groups[0]["lr"] = lr / 10.0
                optimizer.param_groups[1]["lr"] = lr
                if ema is not None:
                    ema.update(raw_model)
                if rank == 0 and writer is not None:
                    writer.add_scalar("train/loss", loss.item() * grad_accum, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    if global_step % 100 == 0:
                        pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}", lr=f"{lr:.2e}")
                global_step += 1

        if accum_progress > 0:
            optimizer.zero_grad(set_to_none=True)

        skipped_total = skipped_empty + skipped_nan
        denom = max(len(trainloader) - skipped_total, 1)
        avg_epoch_loss = epoch_loss_sum / denom
        if rank == 0 and skipped_total > 0:
            logger.warning(
                "Skipped %d training batches (empty_mask=%d, nonfinite_loss=%d).",
                skipped_total,
                skipped_empty,
                skipped_nan,
            )
        if rank == 0 and skipped_total >= len(trainloader):
            logger.warning("All training batches in this epoch were skipped; check AMP/loss and data masks.")

        if (epoch + 1) % cfg.eval_freq == 0:
            eval_model = ema.ema if ema is not None else raw_model
            metrics, n_val = _run_validation(eval_model, valloader, device, cfg)
            model.train()

            if rank == 0:
                logger.info("=" * 90)
                logger.info("  ".join(f"{k:>9}" for k in metrics))
                logger.info(
                    "  ".join(
                        (f"{v:9.4f}" if isinstance(v, (int, float)) and math.isfinite(float(v)) else f"{'nan':>9}")
                        for v in metrics.values()
                    )
                )
                logger.info("=" * 90)
                if n_val <= 0:
                    logger.warning(
                        "Validation: no valid samples (predictions non-finite or masks empty). "
                        "Skipping best-checkpoint comparison for this round."
                    )

                if writer is not None:
                    if n_val > 0:
                        for k, v in metrics.items():
                            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                                writer.add_scalar(f"val/{k}", float(v), epoch)
                    writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)

                if n_val > 0:
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)) and math.isfinite(float(v)):
                            fv = float(v)
                            previous_best[k] = max(previous_best[k], fv) if k in {"d1", "d2", "d3"} else min(previous_best[k], fv)

                current = metrics[cfg.monitor_metric]
                cur_f = float(current) if isinstance(current, (int, float)) else float("nan")
                improved = (
                    n_val > 0
                    and math.isfinite(cur_f)
                    and _is_better(cur_f, best_metric_val, cfg.monitor_mode)
                )
                if improved:
                    best_metric_val = cur_f
                    best_epoch = epoch
                    no_improve_count = 0
                    _save_checkpoint(
                        ckpt_dir / "best.pth",
                        _build_checkpoint(
                            model,
                            ema,
                            optimizer,
                            scaler,
                            epoch,
                            global_step,
                            previous_best,
                            best_metric_val,
                            best_epoch,
                            no_improve_count,
                        ),
                    )
                    logger.info("✓ New best %s=%.4f — saved best.pth", cfg.monitor_metric, best_metric_val)
                else:
                    no_improve_count += 1

                _save_checkpoint(
                    ckpt_dir / "last.pth",
                    _build_checkpoint(
                        model,
                        ema,
                        optimizer,
                        scaler,
                        epoch,
                        global_step,
                        previous_best,
                        best_metric_val,
                        best_epoch,
                        no_improve_count,
                    ),
                )
                row = {"epoch": epoch, "global_step": global_step, "train_loss": avg_epoch_loss, **metrics}
                history.append(row)
                append_csv(report_dir / "results.csv", row)
                save_json(
                    report_dir / "results.json",
                    {"best": {cfg.monitor_metric: best_metric_val, "epoch": best_epoch}, "previous_best": previous_best, "history": history},
                )
                plot_curves(fig_dir, history)

            if cfg.patience is not None:
                if _distributed():
                    nic = torch.tensor([no_improve_count], device=device)
                    dist.broadcast(nic, src=0)
                    no_improve_count = int(nic.item())
                if no_improve_count >= cfg.patience:
                    if rank == 0:
                        logger.info(
                            "Early stopping: no improvement for %d epochs (patience=%d).",
                            no_improve_count,
                            cfg.patience,
                        )
                    break

    if _distributed():
        dist.destroy_process_group()
    if writer is not None:
        writer.close()
    if rank == 0:
        logger.info("Training complete. Best %s = %.4f (epoch %d)", cfg.monitor_metric, best_metric_val, best_epoch)


def main() -> None:
    args = build_train_arg_parser().parse_args()
    cfg = train_config_from_args(args)
    Path(cfg.save_path).mkdir(parents=True, exist_ok=True)
    train_main(cfg)


if __name__ == "__main__":
    main()
