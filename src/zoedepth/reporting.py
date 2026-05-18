"""Training reporting utilities: CSV logging, JSON summary, and metric curve plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def append_csv(csv_path: Path, row: dict[str, Any]) -> None:
    """Append one row to a CSV file, writing the header on first write."""
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})


def save_json(json_path: Path, data: dict[str, Any]) -> None:
    json_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def plot_curves(figures_dir: Path, history: list[dict[str, Any]]) -> None:
    """Save loss and metric curve plots. Silently skips if matplotlib is unavailable."""
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in history]

    # --- loss ---
    if "train_loss" in history[0]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, [r["train_loss"] for r in history], marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("SiLog Loss")
        ax.set_title("Training Loss")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "train_loss.png", dpi=120)
        plt.close(fig)

    # --- metrics grid ---
    metric_keys = [k for k in history[0] if k not in {"epoch", "global_step", "train_loss"}]
    if not metric_keys:
        return

    ncols = 3
    nrows = -(-len(metric_keys) // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
    axes = [axes] if nrows * ncols == 1 else list(axes.flatten())

    for i, key in enumerate(metric_keys):
        ax = axes[i]
        ax.plot(epochs, [r.get(key, float("nan")) for r in history], marker="o", markersize=3)
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)

    for j in range(len(metric_keys), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(figures_dir / "metrics.png", dpi=120)
    plt.close(fig)
