"""Fine-tune Depth Anything V2 (metric head) on custom RGB–depth pairs.

Features
--------
- Auto Batch Size   : OOM probe + automatic grad_accum compensation
- Periodic Eval     : eval_freq-epoch interval, best checkpoint on improvement
- Early Stopping    : patience epochs without improvement on monitor_metric
- EMA               : optional shadow weights used for evaluation
- AMP               : torch.cuda.amp mixed precision
- Gradient Accum    : effective batch = batch_size × grad_accum_steps
- DDP               : torchrun multi-GPU with no_sync optimization
- Resumable         : model / optimizer / scaler / EMA / global_step / early-stop state
- Reporting         : results.csv, results.json, loss + metric curve plots

Examples
--------
  # Single GPU
  python scripts/depth_anything_v2/train_depth_anything_v2.py -c configs/syntable_DA_0.yaml 2>&1 | tee logs/syntable_DA_0.log

  # Multi-GPU (DDP)
  torchrun --nproc_per_node=4 scripts/depth_anything_v2/train_depth_anything_v2.py -c configs/syntable_DA_0.yaml 2>&1 | tee logs/syntable_DA_0.log
"""

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
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from depthanything.checkpoints import load_pretrained_encoder, maybe_sync_batchnorm, unwrap_state_dict
from depthanything.custom_dataset import CustomDepthPairDataset
from depthanything.da2_vendor.depth_anything_v2.dpt import DepthAnythingV2
from depthanything.da2_vendor.metric_train.dist_helper import setup_distributed
from depthanything.da2_vendor.metric_train.loss import SiLogLoss
from depthanything.da2_vendor.metric_train.metric import eval_depth
from depthanything.da2_vendor.metric_train.utils import init_log
from depthanything.ema import ModelEMA
from depthanything.model_configs import MODEL_CONFIGS
from depthanything.reporting import append_csv, plot_curves, save_json
from depthanything.yaml_config import TrainConfig, build_train_arg_parser, train_config_from_args

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _device_for_process(cfg: TrainConfig, local_rank: int, world_size: int) -> torch.device:
    if torch.cuda.is_available():
        if cfg.cuda_indices is not None:
            if _distributed():
                if len(cfg.cuda_indices) != world_size:
                    raise ValueError(
                        f"device.cuda_indices length ({len(cfg.cuda_indices)}) must match "
                        f"WORLD_SIZE ({world_size}) for distributed training."
                    )
                return torch.device(f"cuda:{cfg.cuda_indices[local_rank]}")
            return torch.device(f"cuda:{cfg.cuda_indices[0]}")
        return torch.device(f"cuda:{local_rank}" if _distributed() else "cuda:0")
    return torch.device("cpu")


def _reduce_metrics(results: dict[str, torch.Tensor], nsamples: torch.Tensor) -> None:
    for v in results.values():
        dist.reduce(v, dst=0)
    dist.reduce(nsamples, dst=0)


def _is_better(current: float, best: float, mode: str) -> bool:
    return current < best if mode == "min" else current > best


def _median_scale_align(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Align prediction scale to target via median ratio for relative-depth eval."""
    pred_median = torch.median(pred)
    target_median = torch.median(target)
    scale = target_median / pred_median.clamp_min(1e-6)
    return pred * scale


# ---------------------------------------------------------------------------
# Auto batch size
# ---------------------------------------------------------------------------

def _probe_batch_size(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
    dataset: CustomDepthPairDataset,
    cfg: TrainConfig,
    logger: logging.Logger,
) -> tuple[int, int]:
    """YOLO-style OOM probe: find the largest batch size that fits.

    Returns (actual_bs, grad_accum_steps).
    """
    if device.type != "cuda":
        return cfg.bs, cfg.grad_accum_steps

    target_effective = cfg.bs * cfg.grad_accum_steps
    amp_enabled = cfg.amp and device.type == "cuda"
    max_by_data = len(dataset)
    start_bs = max(1, min(cfg.bs, max_by_data))

    def _fits(batch_size: int) -> bool:
        try:
            probe_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
            sample = next(iter(probe_loader))
            img = sample["image"].to(device)
            depth = sample["depth"].to(device)
            vm = sample["valid_mask"].to(device)

            model.train()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                pred = model(img)
                mask = vm & (depth >= cfg.min_depth) & (depth <= cfg.max_depth)
                loss = criterion(pred, depth, mask)
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
        raise RuntimeError(f"Cannot fit even batch_size=1 on {device}.")

    # Exponential growth to find upper failure bound.
    low = 1
    high = 2
    while high <= max_by_data and _fits(high):
        low = high
        high *= 2

    # Ensure search upper bound is valid for binary search.
    high = min(high, max_by_data + 1)

    # Binary search max feasible batch size in [low, high).
    left, right = low, high
    while left + 1 < right:
        mid = (left + right) // 2
        if _fits(mid):
            left = mid
        else:
            right = mid

    actual_bs = max(start_bs, left) if left >= start_bs else left
    if actual_bs > max_by_data:
        actual_bs = max_by_data
    if not _fits(actual_bs):
        actual_bs = left

    # DDP adds communication buckets/grad views and can require notably more memory
    # than single-process probing. Apply a conservative safety margin.
    if _distributed():
        ddp_safe_bs = max(1, int(actual_bs * 0.7))
        if ddp_safe_bs < actual_bs:
            logger.info(
                "Auto batch DDP safety margin: %d -> %d (70%% of probed max)",
                actual_bs,
                ddp_safe_bs,
            )
        actual_bs = ddp_safe_bs

    accum = max(1, target_effective // max(1, actual_bs))
    logger.info(
        "Auto batch size: start=%d -> selected=%d, grad_accum: %d -> %d (target_effective=%d, actual_effective=%d)",
        cfg.bs,
        actual_bs,
        cfg.grad_accum_steps,
        accum,
        target_effective,
        actual_bs * accum,
    )
    return actual_bs, accum


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_validation(
    eval_model: torch.nn.Module,
    valloader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Full validation pass. Returns averaged metric dict."""
    eval_model.eval()
    metric_keys = ["d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog"]
    results = {k: torch.tensor([0.0], device=device) for k in metric_keys}
    nsamples = torch.tensor([0.0], device=device)

    for sample in valloader:
        img = sample["image"].to(device, non_blocking=True).float()
        depth = sample["depth"].to(device, non_blocking=True)[0]
        vm = sample["valid_mask"].to(device, non_blocking=True)[0]

        pred = eval_model(img)
        pred = F.interpolate(pred[:, None], depth.shape[-2:], mode="bilinear", align_corners=True)[0, 0]

        mask = vm & (depth >= cfg.min_depth) & (depth <= cfg.max_depth)
        if mask.sum() < 10:
            continue

        pred_eval = pred[mask]
        depth_eval = depth[mask]
        if cfg.scale_align_eval:
            pred_eval = _median_scale_align(pred_eval, depth_eval)
        cur = eval_depth(pred_eval, depth_eval)
        for k in metric_keys:
            results[k] += cur[k]
        nsamples += 1

    if _distributed():
        torch.distributed.barrier()
        _reduce_metrics(results, nsamples)

    n = max(nsamples.item(), 1.0)
    return {k: (results[k] / n).item() for k in metric_keys}


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_main(cfg: TrainConfig) -> None:  # noqa: C901
    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    # --- Reproducibility ---
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

    # --- Distributed setup ---
    if _distributed():
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA.")
        rank, world_size = setup_distributed(port=cfg.port)
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size = 0, 1
        local_rank = 0

    device = _device_for_process(cfg, local_rank, world_size)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # --- Output dirs ---
    save_root = Path(cfg.save_path)
    ckpt_dir  = save_root / "checkpoints"
    log_dir   = save_root / "logs"
    report_dir = save_root / "reports"
    fig_dir   = save_root / "figures"
    if rank == 0:
        for d in (ckpt_dir, log_dir, report_dir, fig_dir):
            d.mkdir(parents=True, exist_ok=True)
        logger.info("%s\n", pprint.pformat({**asdict(cfg), "ngpus": world_size, "distributed": _distributed()}))
        logger.info("Device: %s", device)

    writer = SummaryWriter(log_dir) if rank == 0 else None

    cudnn.enabled = device.type == "cuda"
    cudnn.benchmark = device.type == "cuda"
    amp_enabled = cfg.amp and device.type == "cuda"

    # --- Datasets ---
    size = (cfg.img_size, cfg.img_size)
    trainset = CustomDepthPairDataset(
        cfg.train_list, "train", size=size,
        depth_scale=cfg.depth_scale, list_root=cfg.list_root,
    )
    valset = CustomDepthPairDataset(
        cfg.val_list, "val", size=size,
        depth_scale=cfg.depth_scale, list_root=cfg.list_root,
    )

    # --- Model ---
    model = DepthAnythingV2(**{**MODEL_CONFIGS[cfg.encoder], "max_depth": cfg.max_depth})
    model = maybe_sync_batchnorm(model, _distributed())

    # Load weights before probe (model size determines OOM threshold)
    if cfg.resume_from:
        resume_blob = torch.load(cfg.resume_from, map_location="cpu")
        if not isinstance(resume_blob, dict) or "model" not in resume_blob:
            raise ValueError("--resume-from must be a checkpoint saved by this trainer.")
        model.load_state_dict(unwrap_state_dict(resume_blob["model"]), strict=True)
    elif cfg.pretrained_from:
        load_pretrained_encoder(model, cfg.pretrained_from)

    model = model.to(device)
    criterion = SiLogLoss().to(device)

    # --- Auto batch size ---
    actual_bs = cfg.bs
    grad_accum = cfg.grad_accum_steps
    if cfg.auto_batch_size:
        if rank == 0:
            actual_bs, grad_accum = _probe_batch_size(model, criterion, device, trainset, cfg, logger)
        if _distributed():
            info = torch.tensor([actual_bs, grad_accum], dtype=torch.long, device=device)
            dist.broadcast(info, src=0)
            actual_bs, grad_accum = int(info[0].item()), int(info[1].item())
        if rank == 0:
            logger.info("Effective batch size: %d × %d = %d", actual_bs, grad_accum, actual_bs * grad_accum)

    # --- DDP ---
    if _distributed():
        train_sampler = DistributedSampler(trainset, shuffle=True)
        val_sampler   = DistributedSampler(valset,   shuffle=False)
        model = DDP(
            model,
            device_ids=[device.index],
            broadcast_buffers=False,
            find_unused_parameters=cfg.find_unused_parameters,
        )
    else:
        train_sampler = val_sampler = None

    # --- DataLoaders ---
    trainloader = DataLoader(
        trainset, batch_size=actual_bs,
        pin_memory=device.type == "cuda", num_workers=cfg.num_workers,
        drop_last=True, sampler=train_sampler, shuffle=not _distributed(),
    )
    valloader = DataLoader(
        valset, batch_size=1,
        pin_memory=device.type == "cuda", num_workers=cfg.num_workers,
        drop_last=False, sampler=val_sampler, shuffle=False,
    )

    if len(trainloader) == 0:
        raise RuntimeError(
            f"trainloader is empty (dataset={len(trainset)}, batch_size={actual_bs}). "
            "Reduce batch_size or add more training samples."
        )

    # --- Optimizer / Scaler / EMA ---
    raw_model = model.module if isinstance(model, DDP) else model
    optimizer = AdamW(
        [
            {"params": [p for n, p in raw_model.named_parameters() if "pretrained" in n],     "lr": cfg.lr},
            {"params": [p for n, p in raw_model.named_parameters() if "pretrained" not in n], "lr": cfg.lr * 10.0},
        ],
        lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.01,
    )
    scaler = GradScaler(enabled=amp_enabled)
    ema: ModelEMA | None = ModelEMA(raw_model, decay=cfg.ema_decay, device=device) if cfg.ema_decay > 0.0 else None

    # --- Training state ---
    start_epoch     = 0
    global_step     = 0
    best_metric_val = float("inf") if cfg.monitor_mode == "min" else float("-inf")
    best_epoch      = 0
    no_improve_count = 0
    previous_best   = {k: (0.0 if k in {"d1", "d2", "d3"} else 1e9)
                       for k in ["d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog"]}
    history: list[dict] = []

    # Restore optimizer / scaler / EMA / step counters from resume
    if cfg.resume_from:
        blob = resume_blob  # already loaded above
        if "optimizer" in blob:
            optimizer.load_state_dict(blob["optimizer"])
        if "scaler" in blob:
            scaler.load_state_dict(blob["scaler"])
        if ema is not None and "ema" in blob:
            ema.load_state_dict(blob["ema"])
        start_epoch      = int(blob.get("epoch", -1)) + 1
        global_step      = int(blob.get("global_step", 0))
        best_metric_val  = float(blob.get("best_metric_val", best_metric_val))
        best_epoch       = int(blob.get("best_epoch", 0))
        no_improve_count = int(blob.get("no_improve_count", 0))
        if "previous_best" in blob:
            previous_best.update(blob["previous_best"])

    # Number of optimizer steps per epoch under gradient accumulation.
    # Use ceiling division so a partial accumulation at epoch end counts as one step.
    steps_per_epoch = max(1, (len(trainloader) + grad_accum - 1) // grad_accum)
    total_iters     = cfg.epochs * steps_per_epoch

    # =========================================================================
    # Training loop
    # =========================================================================
    for epoch in range(start_epoch, cfg.epochs):
        if rank == 0:
            logger.info(
                "===========> Epoch %d/%d | best %s=%.4f (ep%d) | patience %d/%s",
                epoch, cfg.epochs,
                cfg.monitor_metric, best_metric_val, best_epoch,
                no_improve_count, str(cfg.patience),
            )

        if train_sampler is not None:
            train_sampler.set_epoch(epoch + 1)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch}", disable=(rank != 0), dynamic_ncols=True)
        for i, sample in enumerate(pbar):
            img        = sample["image"].to(device, non_blocking=True)
            depth      = sample["depth"].to(device, non_blocking=True)
            valid_mask = sample["valid_mask"].to(device, non_blocking=True)

            if random.random() < 0.5:
                img        = img.flip(-1)
                depth      = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            mask = valid_mask & (depth >= cfg.min_depth) & (depth <= cfg.max_depth)

            is_last_accum = ((i + 1) % grad_accum == 0) or ((i + 1) == len(trainloader))
            sync_ctx = (
                contextlib.nullcontext()
                if (not _distributed() or is_last_accum)
                else model.no_sync()
            )

            with sync_ctx:
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    pred = model(img)
                    loss = criterion(pred, depth, mask) / grad_accum
                scaler.scale(loss).backward()

            epoch_loss_sum += loss.item() * grad_accum  # restore true loss for logging

            if is_last_accum:
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                lr = cfg.lr * (1 - global_step / max(total_iters, 1)) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * 10.0

                if ema is not None:
                    ema.update(raw_model)

                if rank == 0:
                    if writer is not None:
                        writer.add_scalar("train/loss", loss.item() * grad_accum, global_step)
                        writer.add_scalar("train/lr",   lr,                       global_step)
                    if global_step % 100 == 0:
                        pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}", lr=f"{lr:.2e}")

                global_step += 1

        avg_epoch_loss = epoch_loss_sum / max(len(trainloader), 1)

        # =====================================================================
        # Periodic evaluation
        # =====================================================================
        if (epoch + 1) % cfg.eval_freq == 0:
            eval_model = ema.ema if ema is not None else raw_model
            metrics = _run_validation(eval_model, valloader, device, cfg)
            model.train()  # restore train mode after eval

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

                # --- best checkpoint & early stopping counter ---
                current = metrics[cfg.monitor_metric]
                improved = _is_better(current, best_metric_val, cfg.monitor_mode)
                if improved:
                    best_metric_val  = current
                    best_epoch       = epoch
                    no_improve_count = 0
                    ckpt = _build_checkpoint(model, ema, optimizer, scaler, epoch, global_step, previous_best, best_metric_val, best_epoch, no_improve_count)
                    _save_checkpoint(ckpt_dir / "best.pth", ckpt)
                    logger.info("✓ New best %s=%.4f — saved best.pth", cfg.monitor_metric, best_metric_val)
                else:
                    no_improve_count += 1

                # --- last checkpoint ---
                ckpt = _build_checkpoint(model, ema, optimizer, scaler, epoch, global_step, previous_best, best_metric_val, best_epoch, no_improve_count)
                _save_checkpoint(ckpt_dir / "last.pth", ckpt)

                # --- reporting ---
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

            # --- early stopping (broadcast counter to all ranks) ---
            if cfg.patience is not None:
                if _distributed():
                    nic = torch.tensor([no_improve_count], device=device)
                    dist.broadcast(nic, src=0)
                    no_improve_count = int(nic.item())
                if no_improve_count >= cfg.patience:
                    if rank == 0:
                        logger.info(
                            "Early stopping: no improvement for %d epochs (patience=%d).",
                            no_improve_count, cfg.patience,
                        )
                    break

    # =========================================================================
    # Cleanup
    # =========================================================================
    if _distributed():
        dist.destroy_process_group()
    if writer is not None:
        writer.close()
    if rank == 0:
        logger.info("Training complete. Best %s = %.4f (epoch %d)", cfg.monitor_metric, best_metric_val, best_epoch)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    warnings = __import__("warnings")
    try:
        from numpy.exceptions import RankWarning
    except ImportError:
        from numpy import RankWarning  # type: ignore[attr-defined,no-redef]
    warnings.simplefilter("ignore", RankWarning)
    args = build_train_arg_parser().parse_args()
    cfg  = train_config_from_args(args)
    Path(cfg.save_path).mkdir(parents=True, exist_ok=True)
    train_main(cfg)


if __name__ == "__main__":
    main()
