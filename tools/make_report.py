"""One-shot post-training report for a finished run.

Bundles the existing post-hoc tools into a single command so every run yields
the same artifacts: an end-of-training eval, loss + AP convergence curves, a
handful of random test-case prediction overlays, and a short markdown summary
with the final AP table. Works for BOTH the BCNet and the baseline runs (the
viz step auto-detects the missing occluder field on the baseline head).

It shells out to the existing scripts (train.py --eval-only, plot_losses.py,
plot_metrics.py, viz_predictions.py) with the SAME Python interpreter, so there
is no duplicated logic and the env (CUDA_VISIBLE_DEVICES etc.) is inherited.

Usage:
    CUDA_VISIBLE_DEVICES=2 python tools/make_report.py \
        --config-file configs/server_bcnet.yaml \
        --run-dir     output/bcnet_full

    # skip the (GPU-using) re-eval and viz, just regenerate the curves:
    python tools/make_report.py --config-file ... --run-dir ... --no-eval --no-viz
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> int:
    """Run a subcommand from the repo root, streaming its output. Returns rc."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def _final_ap_table(metrics_path: Path) -> dict[str, float]:
    """Pull the last-seen value of each AP key from metrics.json."""
    keys = ["bbox/AP", "bbox/AP50", "bbox/AP75", "segm/AP", "segm/AP50", "segm/AP75"]
    latest: dict[str, float] = {}
    if not metrics_path.exists():
        return latest
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k in keys:
                if k in rec and rec[k] is not None:
                    latest[k] = float(rec[k])
    return latest


def _write_summary(run_dir: Path, config_file: str, ap: dict[str, float]) -> Path:
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "summary.md"
    lines = [
        f"# Report: {run_dir.name}",
        "",
        f"- config: `{config_file}`",
        f"- weights: `{run_dir / 'model_final.pth'}`",
        "",
        "## Final evaluation (visible-mask COCO AP)",
        "",
    ]
    if ap:
        lines += [
            "| Task | AP | AP50 | AP75 |",
            "|------|----|------|------|",
            f"| bbox | {ap.get('bbox/AP', float('nan')):.2f} | "
            f"{ap.get('bbox/AP50', float('nan')):.2f} | {ap.get('bbox/AP75', float('nan')):.2f} |",
            f"| segm | {ap.get('segm/AP', float('nan')):.2f} | "
            f"{ap.get('segm/AP50', float('nan')):.2f} | {ap.get('segm/AP75', float('nan')):.2f} |",
        ]
    else:
        lines.append("_No AP found in metrics.json (was the model evaluated?)._")
    lines += [
        "",
        "## Artifacts",
        "",
        "- `loss_curves.png` -- per-step training losses",
        "- `ap_curves.png` -- eval AP vs iteration",
        "- `viz/` -- random test-case prediction overlays",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", required=True)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--weights", default=None, help="Default: <run-dir>/model_final.pth")
    p.add_argument("--n-viz", type=int, default=6, help="Number of random test images to render.")
    p.add_argument("--no-eval", action="store_true", help="Skip the end-of-training re-eval.")
    p.add_argument("--no-viz", action="store_true", help="Skip prediction visualization.")
    args = p.parse_args()

    run_dir = args.run_dir
    weights = args.weights or str(run_dir / "model_final.pth")
    py = sys.executable

    # 1. End-of-training eval (writes eval/<dataset>/ and an AP record into
    #    metrics.json so the AP table / curve below have data).
    if not args.no_eval:
        # Disable W&B for this re-eval: the run is already finished and logged,
        # and init_wandb would otherwise spawn a duplicate-named run in the
        # group every time a report is generated.
        rc = _run([py, "train.py", "--config-file", args.config_file,
                   "--eval-only", "MODEL.WEIGHTS", weights,
                   "WANDB.ENABLED", "False"])
        if rc != 0:
            print("WARNING: eval step returned non-zero; continuing with existing metrics.json.")

    metrics = run_dir / "metrics.json"

    # 2. Loss curves.
    _run([py, "tools/plot_losses.py", "--metrics", str(metrics),
          "--out", str(run_dir / "loss_curves.png")])

    # 3. AP curves (best-effort: needs at least one eval record).
    rc = _run([py, "tools/plot_metrics.py", "--metrics", str(metrics),
               "--out", str(run_dir / "ap_curves.png")])
    if rc != 0:
        print("NOTE: no AP curve (no eval records in metrics.json).")

    # 4. Random test-case prediction overlays (baseline-safe).
    if not args.no_viz:
        _run([py, "tools/viz_predictions.py", "--config-file", args.config_file,
              "--weights", weights, "--out-dir", str(run_dir / "viz"),
              "--n", str(args.n_viz)])

    # 5. Markdown summary with the final AP table.
    summary = _write_summary(run_dir, args.config_file, _final_ap_table(metrics))
    print(f"\nWrote {summary}")


if __name__ == "__main__":
    main()
