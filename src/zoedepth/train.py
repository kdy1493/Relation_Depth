"""Fine-tune ZoeDepth on custom RGB–depth pairs (AMP/DDP/EMA/reporting, AdaBins-style loop)."""

from __future__ import annotations

import contextlib
import logging
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

from zoedepth.checkpoints import unwrap_state_dict
from zoedepth.dataset import CustomDepthPairDataset
from zoedepth.dist_helper import setup_distributed
from zoedepth.ema import ModelEMA
from zoedepth.log_utils import init_log
from zoedepth.metrics import eval_depth
from zoedepth.reporting import append_csv, plot_curves, save_json

from .yaml_config import ZoeDepthTrainConfig, build_train_arg_parser, train_config_from_args
from .zoe_model import build_zoe_model

METRIC_KEYS = ["d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog"]

# Autocast dtype for CUDA AMP. GradScaler is only intended for float16; do not combine with bfloat16.
AMP_AUTOCAST_DTYPE = torch.bfloat16


def _distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _device_for_process(cfg: ZoeDepthTrainConfig, local_rank: int, world_size: int) -> torch.device:
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
    pred_median = torch.median(pred)
    target_median = torch.median(target)
    scale = target_median / pred_median.clamp_min(1e-6)
    return pred * scale


def _load_model_state(path: str) -> dict[str, torch.Tensor]:
    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "model" in blob:
        blob = blob["model"]
    if isinstance(blob, dict) and any(k.startswith("module.") for k in blob):
        blob = {k.replace("module.", "", 1): v for k, v in blob.items()}
    if not isinstance(blob, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return unwrap_state_dict(blob)


def _ensure_zoe_imports() -> None:
    from ._vendor import ensure_zoedepth_on_path

    ensure_zoedepth_on_path()


def _probe_batch_size(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
    dataset: CustomDepthPairDataset,
    cfg: ZoeDepthTrainConfig,
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
                out = model(img, denorm=True)
                mask = valid_mask & (depth >= cfg.min_depth) & (depth <= cfg.max_depth)
                loss = criterion(out, depth, mask=mask)
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
    cfg: ZoeDepthTrainConfig,
) -> dict[str, float]:
    eval_model.eval()
    sums = {k: torch.tensor([0.0], device=device) for k in METRIC_KEYS}
    nsamples = torch.tensor([0.0], device=device)
    for sample in valloader:
        img = sample["image"].to(device, non_blocking=True).float()
        depth = sample["depth"].to(device, non_blocking=True)[0]
        vm = sample["valid_mask"].to(device, non_blocking=True)[0]
        out = eval_model(img, denorm=True)
        pred = out["metric_depth"]
        pred = F.interpolate(pred, depth.shape[-2:], mode="bilinear", align_corners=True)[0, 0]
        mask = vm & (depth >= cfg.min_depth) & (depth <= cfg.max_depth)
        if mask.sum() < 10:
            continue
        pred_eval = pred[mask]
        depth_eval = depth[mask]
        if cfg.scale_align_eval:
            pred_eval = _median_scale_align(pred_eval, depth_eval)
        cur = eval_depth(pred_eval, depth_eval)
        for k in METRIC_KEYS:
            sums[k] += float(cur[k])
        nsamples += 1

    if _distributed():
        dist.barrier()
        for v in sums.values():
            dist.reduce(v, dst=0)
        dist.reduce(nsamples, dst=0)

    n = max(nsamples.item(), 1.0)
    return {k: (sums[k] / n).item() for k in METRIC_KEYS}


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
    blob: dict = {
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


def train_main(cfg: ZoeDepthTrainConfig) -> None:
    _ensure_zoe_imports()
    from zoedepth.trainers.loss import SILogLoss

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
    if rank == 0 and cfg.amp and device.type == "cuda":
        logger.warning(
            "ZoeDepth + AMP: mixed-precision forward can yield NaN metric_depth and useless SILog; "
            "if training logs 'Nan SILog loss', set training.amp to false in the YAML."
        )

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

    model = build_zoe_model(cfg)
    if cfg.resume_from:
        model.load_state_dict(_load_model_state(cfg.resume_from), strict=True)

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
    param_groups = raw_model.get_lr_params(cfg.lr)
    optimizer = AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
    use_grad_scaler = amp_enabled and AMP_AUTOCAST_DTYPE == torch.float16
    scaler = GradScaler("cuda", enabled=use_grad_scaler)
    ema: ModelEMA | None = ModelEMA(raw_model, decay=cfg.ema_decay, device=device) if cfg.ema_decay > 0.0 else None

    start_epoch = 0
    global_step = 0
    best_metric_val = float("inf") if cfg.monitor_mode == "min" else float("-inf")
    best_epoch = 0
    no_improve_count = 0
    previous_best = {k: (0.0 if k in {"d1", "d2", "d3"} else 1e9) for k in METRIC_KEYS}
    history: list[dict] = []

    resume_blob: dict | None = None
    if cfg.resume_from:
        resume_blob = torch.load(cfg.resume_from, map_location="cpu")
        if isinstance(resume_blob, dict):
            if "optimizer" in resume_blob:
                optimizer.load_state_dict(resume_blob["optimizer"])
            if "scaler" in resume_blob and scaler.is_enabled():
                scaler.load_state_dict(resume_blob["scaler"])
            elif "scaler" in resume_blob and rank == 0:
                logger.info("Resume: omitting scaler state (GradScaler off for current AMP dtype).")
            if ema is not None and "ema" in resume_blob:
                ema.load_state_dict(resume_blob["ema"])
            start_epoch = int(resume_blob.get("epoch", -1)) + 1
            global_step = int(resume_blob.get("global_step", 0))
            best_metric_val = float(resume_blob.get("best_metric_val", best_metric_val))
            best_epoch = int(resume_blob.get("best_epoch", 0))
            no_improve_count = int(resume_blob.get("no_improve_count", 0))
            if "previous_best" in resume_blob:
                previous_best.update(resume_blob["previous_best"])

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
        skipped_batches = 0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch}", disable=(rank != 0), dynamic_ncols=True)
        for i, sample in enumerate(pbar):
            img = sample["image"].to(device, non_blocking=True)
            depth = sample["depth"].to(device, non_blocking=True)
            valid_mask = sample["valid_mask"].to(device, non_blocking=True)
            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            mask = valid_mask & (depth >= cfg.min_depth) & (depth <= cfg.max_depth)
            if mask.sum().item() == 0:
                skipped_batches += 1
                continue

            is_last_accum = ((i + 1) % grad_accum == 0) or ((i + 1) == len(trainloader))
            sync_ctx = contextlib.nullcontext() if (not _distributed() or is_last_accum) else model.no_sync()
            with sync_ctx:
                with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=AMP_AUTOCAST_DTYPE):
                    out = model(img, denorm=True)
                    loss = criterion(out, depth, mask=mask) / grad_accum
                if not torch.isfinite(loss):
                    skipped_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss).backward()
            epoch_loss_sum += float(loss.item()) * grad_accum

            if is_last_accum:
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                lr_scale = (1.0 - global_step / max(total_iters, 1)) ** 0.9
                for g, blr in zip(optimizer.param_groups, base_lrs):
                    g["lr"] = blr * lr_scale
                if ema is not None:
                    ema.update(raw_model)
                if rank == 0 and writer is not None:
                    writer.add_scalar("train/loss", float(loss.item()) * grad_accum, global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    if global_step % 100 == 0:
                        pbar.set_postfix(loss=f"{float(loss.item()) * grad_accum:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                global_step += 1

        avg_epoch_loss = epoch_loss_sum / max(len(trainloader), 1)
        if rank == 0 and skipped_batches > 0:
            skip_pct = 100.0 * skipped_batches / max(len(trainloader), 1)
            logger.warning(
                "Skipped %d/%d training batches (%.1f%%) — NaN loss or empty mask.",
                skipped_batches, len(trainloader), skip_pct,
            )
            if skip_pct > 5.0 and amp_enabled:
                logger.warning("High NaN skip rate with AMP; consider setting training.amp: false in the YAML.")

        if (epoch + 1) % cfg.eval_freq == 0:
            eval_model = ema.ema if ema is not None else raw_model
            metrics = _run_validation(eval_model, valloader, device, cfg)
            model.train()

            if rank == 0:
                logger.info("=" * 90)
                logger.info("  ".join(f"{k:>9}" for k in metrics))
                logger.info("  ".join(f"{v:9.4f}" for v in metrics.values()))
                logger.info("=" * 90)

                if writer is not None:
                    for k, v in metrics.items():
                        writer.add_scalar(f"val/{k}", v, epoch)
                    writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)

                for k, v in metrics.items():
                    previous_best[k] = max(previous_best[k], v) if k in {"d1", "d2", "d3"} else min(previous_best[k], v)

                current = metrics[cfg.monitor_metric]
                improved = _is_better(current, best_metric_val, cfg.monitor_mode)
                if improved:
                    best_metric_val = current
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
                    {
                        "best": {cfg.monitor_metric: best_metric_val, "epoch": best_epoch},
                        "previous_best": previous_best,
                        "history": history,
                    },
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
