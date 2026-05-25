"""Plot BCNet training losses from Detectron2's metrics.json.

Detectron2's `JSONWriter` (active in DefaultTrainer) appends one
JSON object per logging step to `<OUTPUT_DIR>/metrics.json`. We
pull the per-step values for every `loss_*` key and render them
as overlaid line plots.

Usage:
    PYTHONUTF8=1 python tools/plot_losses.py \
        --metrics output/overfit200/metrics.json \
        --out     output/overfit200/loss_curves.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    iters = []
    series: dict[str, list[float]] = defaultdict(list)
    series_iters: dict[str, list[int]] = defaultdict(list)

    # metrics.json is JSON-lines, one object per logging step.
    with args.metrics.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            it = rec.get("iteration")
            if it is None:
                continue
            iters.append(it)
            for k, v in rec.items():
                if k.startswith("loss_") or k == "total_loss":
                    series[k].append(float(v))
                    series_iters[k].append(int(it))

    if not series:
        raise SystemExit(f"No loss_* keys found in {args.metrics}")

    # Order: total_loss first if present, then BCNet-specific, then others.
    keys = sorted(
        series.keys(),
        key=lambda k: (
            0 if k == "total_loss" else (1 if "mask" in k else 2),
            k,
        ),
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    for k in keys:
        ax.plot(series_iters[k], series[k], label=k, linewidth=1.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title(f"BCNet training losses ({args.metrics.parent.name})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_yscale("log")  # mask losses sit at ~0.7 while loss_cls drops 2+ orders of magnitude
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"Saved {args.out}  ({len(iters)} log points, {len(series)} series)")


if __name__ == "__main__":
    main()
