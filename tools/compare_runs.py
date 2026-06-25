"""Compare two (or more) finished runs: BCNet vs baseline.

Reads each run's metrics.json, emits a side-by-side final-AP table (markdown)
and an overlaid `segm/AP`-vs-iteration plot so the convergence of BCNet and the
baseline can be read off one figure. W&B already overlays the runs live via the
shared `WANDB.GROUP`; this is the offline / for-the-writeup equivalent.

Usage:
    python tools/compare_runs.py \
        --runs output/bcnet_full output/baseline_full \
        --out-dir output/comparison
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

_AP_KEYS = ["bbox/AP", "bbox/AP50", "bbox/AP75", "segm/AP", "segm/AP50", "segm/AP75"]


def _load_records(metrics_path: Path) -> list[dict]:
    recs = []
    if not metrics_path.exists():
        return recs
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def _final_ap(recs: list[dict]) -> dict[str, float]:
    latest: dict[str, float] = {}
    for rec in recs:
        for k in _AP_KEYS:
            if k in rec and rec[k] is not None:
                latest[k] = float(rec[k])
    return latest


def _ap_curve(recs: list[dict], key: str) -> tuple[list[int], list[float]]:
    its, vals = [], []
    for rec in recs:
        it = rec.get("iteration")
        if it is not None and key in rec and rec[key] is not None:
            its.append(int(it))
            vals.append(float(rec[key]))
    return its, vals


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, type=Path,
                   help="Run output dirs (each must contain metrics.json).")
    p.add_argument("--out-dir", type=Path, default=Path("output/comparison"))
    p.add_argument("--curve-key", default="segm/AP", help="AP key to overlay.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = {run.name: _load_records(run / "metrics.json") for run in args.runs}

    # --- Markdown table ---
    md = ["# Run comparison", "", "## Final AP", "",
          "| Run | " + " | ".join(_AP_KEYS) + " |",
          "|-----|" + "|".join(["----"] * len(_AP_KEYS)) + "|"]
    for name, recs in runs.items():
        ap = _final_ap(recs)
        row = " | ".join(
            f"{ap[k]:.2f}" if k in ap else "-" for k in _AP_KEYS
        )
        md.append(f"| {name} | {row} |")
    md_path = args.out_dir / "compare.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")

    # --- Overlaid AP curve ---
    fig, ax = plt.subplots(figsize=(11, 6))
    drew = False
    for name, recs in runs.items():
        its, vals = _ap_curve(recs, args.curve_key)
        if its:
            ax.plot(its, vals, marker="o", linewidth=1.8, label=name)
            drew = True
    if drew:
        ax.set_xlabel("iteration")
        ax.set_ylabel(args.curve_key)
        ax.set_title(f"{args.curve_key} vs iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=10)
        fig.tight_layout()
        plot_path = args.out_dir / "compare_ap.png"
        fig.savefig(plot_path, dpi=110, bbox_inches="tight")
        print(f"Wrote {plot_path}")
    else:
        print(f"NOTE: no '{args.curve_key}' records found; skipped the curve "
              f"(train with TEST.EVAL_PERIOD > 0 to get it).")


if __name__ == "__main__":
    main()
