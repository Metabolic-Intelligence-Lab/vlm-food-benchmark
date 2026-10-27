"""Linear calibration at the primary configurations of Table 2 (eps = 0.5).

Calibration analysis of Section 3.2 (Table 4, Fig. 9) at the primary
threshold eps = 0.5 and at the temperature of Table 2 for eight representative
models. Same procedure as 05_calibration.py: OLS fit
GT = b + k * pred on the matched items with a numerical estimate, errors of
the calibrated estimates evaluated with leave-one-image-out cross-validation.

Sources: per_item_greedy.parquet (first campaign) and per_item_v2.parquet
(newer models; GPT-5.5 and Claude pool the three replicates at the provider
default temperature).

Outputs:
  analysis/outputs/calibration_primary.csv
  analysis/outputs/figures/GT_PRED_qwen25vl_7b.png
  analysis/outputs/figures/GT_PRED_gpt4o.png

Usage: python analysis/24_calibration_primary.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import OUTPUTS_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(OUTPUTS_DIR, "figures")
EPS = 0.5

# model -> (source, temperature, runs)
CONFIGS = [
    ("qwen2.5vl:7b", "greedy", 0.0, None),
    ("qwen3.5:9b", "v2", 0.0, ["r1"]),
    ("minicpm-v4.5:8b", "v2", 0.0, ["r1"]),
    ("gpt-4o", "greedy", 0.2, None),
    ("gpt-4.1", "greedy", 0.4, None),
    ("gpt-5.2", "greedy", 0.0, None),
    ("gpt-5.5", "v2", 1.0, ["r1", "r2", "r3"]),
    ("claude-sonnet-5", "v2", 1.0, ["r1", "r2", "r3"]),
]
FIGS = {"qwen2.5vl:7b": "GT_PRED_qwen25vl_7b.png", "gpt-4o": "GT_PRED_gpt4o.png"}


def fit_line(pred, gt):
    if len(pred) < 3 or np.all(pred == pred[0]):
        return 0.0, 1.0
    k, b = np.polyfit(pred, gt, 1)
    return float(b), float(k)


def loio_calibrated(items):
    out = np.empty(len(items))
    images = items["image"].to_numpy()
    pred = items["pred_weight"].to_numpy(dtype=float)
    gt = items["gt_weight"].to_numpy(dtype=float)
    for img in np.unique(images):
        test = images == img
        b, k = fit_line(pred[~test], gt[~test])
        out[test] = b + k * pred[test]
    return out


def scatter(items, b, k, path):
    pred = items["pred_weight"].to_numpy(dtype=float)
    gt = items["gt_weight"].to_numpy(dtype=float)
    x = np.array([0, 350])
    plt.figure(figsize=(6, 6))
    plt.plot(pred, gt, "o", alpha=0.6)
    plt.plot(x, x, "k--")
    plt.plot(x, b + k * x, "r-")
    plt.xlabel("Predicted portion size [g]")
    plt.ylabel("GT portion size [g]")
    plt.xlim(0, 350)
    plt.ylim(0, 350)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    src = {m: pd.read_parquet(os.path.join(OUTPUTS_DIR, f"per_item_{m}.parquet"))
           for m in ("greedy", "v2")}
    rows = []
    for model, source, temp, runs in CONFIGS:
        d = src[source]
        s = d[(d.model == model) & (d.temperature == temp) & (d.epsilon == EPS)
              & d.pred_weight.notna()]
        if runs is not None:
            s = s[s.run.isin(runs)]
        pred = s.pred_weight.to_numpy(dtype=float)
        gt = s.gt_weight.to_numpy(dtype=float)
        b, k = fit_line(pred, gt)
        ins = b + k * pred
        cv = loio_calibrated(s)
        rows.append({
            "model": model, "temperature": temp if runs is None or len(runs) == 1
            else "default (r1-r3)", "epsilon": EPS, "n": len(s),
            "b": round(b, 2), "k": round(k, 3),
            "mae_raw": np.mean(np.abs(gt - pred)),
            "medae_raw": np.median(np.abs(gt - pred)),
            "mae_cal_insample": np.mean(np.abs(gt - ins)),
            "medae_cal_insample": np.median(np.abs(gt - ins)),
            "mae_cal_cv": np.mean(np.abs(gt - cv)),
            "medae_cal_cv": np.median(np.abs(gt - cv)),
        })
        if model in FIGS:
            scatter(s, b, k, os.path.join(FIG_DIR, FIGS[model]))
    df = pd.DataFrame(rows)
    out = os.path.join(OUTPUTS_DIR, "calibration_primary.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
