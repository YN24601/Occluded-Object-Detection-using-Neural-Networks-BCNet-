"""Plot evaluation AP curves from Detectron2's metrics.json.

Companion to `plot_losses.py`. When `TEST.EVAL_PERIOD > 0`, the periodic
`COCOEvaluator` writes its results (`bbox/AP`, `segm/AP`, `segm/AP50`, ...)
back into `<OUTPUT_DIR>/metrics.json` as extra keys on the eval-step record.
This script pulls those keys and renders AP-vs-iteration line plots so you can
see the model converging on the metric, not just the loss.

If the run had `EVAL_PERIOD: 0` (eval only at the very end) there is just one
data point per series -- the plot still renders (a single marker).

Usage:
    python tools/plot_metrics.py \
        --metrics output/bcnet_full/metrics.json \
        --out     output/bcnet_full/ap_curves.png
"""

from __future__ import annotations

import os

# Windows: matplotlib (and numpy/MKL) ship Intel's OpenMP runtime; allow the
# duplicate before importing them. No-op / harmless on Linux.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


# AP keys COCOEvaluator emits, in the order we want them drawn. We plot the
# headline AP/AP50/AP75 for both tasks; per-size (APs/APm/APl) are skipped to
# keep the figure readable.
_AP_KEYS = [
    "segm/AP",
    "segm/AP50",
    "segm/AP75",
    "bbox/AP",
    "bbox/AP50",
    "bbox/AP75",
]


def load_ap_series(metrics_path: Path) -> tuple[dict[str, list[float]], dict[str, list[int]]]:
    """Return {key: [values]} and {key: [iterations]} for every AP key present."""
    series: dict[str, list[float]] = defaultdict(list)
    series_iters: dict[str, list[int]] = defaultdict(list)

    with metrics_path.open("r", encoding="utf-8") as f:
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
            for k in _AP_KEYS:
                if k in rec and rec[k] is not None:
                    series[k].append(float(rec[k]))
                    series_iters[k].append(int(it))
    return series, series_iters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    series, series_iters = load_ap_series(args.metrics)
    if not series:
        raise SystemExit(
            f"No AP keys found in {args.metrics}. Was the run trained with "
            f"TEST.EVAL_PERIOD > 0 (or evaluated at the end)?"
        )

    fig, ax = plt.subplots(figsize=(11, 6))
    for k in _AP_KEYS:
        if k not in series:
            continue
        ax.plot(series_iters[k], series[k], marker="o", linewidth=1.5, label=k)
    ax.set_xlabel("iteration")
    ax.set_ylabel("AP")
    ax.set_title(f"BCNet eval AP ({args.metrics.parent.name})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"Saved {args.out}  ({len(series)} AP series)")


if __name__ == "__main__":
    main()
