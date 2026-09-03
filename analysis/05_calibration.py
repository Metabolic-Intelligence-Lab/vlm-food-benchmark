"""WP4 - Linear calibration evaluated OUT-OF-SAMPLE.

The submitted paper fitted GT = w0 + k*pred on all matched items and reported
the error reduction on the SAME items (in-sample). Here the calibration is
evaluated with leave-one-image-out cross-validation (clusters = images, same
clustering as the bootstrap): for every image, the line is fitted on the items
of the other images and applied to this image's items.

Outputs: analysis/outputs/calibration_<mode>.csv with, per (model, T, eps):
  mae/medae raw, in-sample calibrated (paper's optimistic number),
  LOIO-CV calibrated (honest number), fitted w0/k on the full data.

Usage: analysis_env/bin/python analysis/05_calibration.py [--mode greedy]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import OUTPUTS_DIR


def fit_line(pred, gt):
    """OLS fit gt = w0 + k*pred; identity fallback for degenerate cases."""
    if len(pred) < 3 or np.all(pred == pred[0]):
        return 0.0, 1.0
    k, w0 = np.polyfit(pred, gt, 1)
    return float(w0), float(k)


def loio_calibrated(items):
    """Leave-one-image-out calibrated predictions."""
    out = np.empty(len(items))
    images = items["image"].to_numpy()
    pred = items["pred_weight"].to_numpy(dtype=float)
    gt = items["gt_weight"].to_numpy(dtype=float)
    for img in np.unique(images):
        test = images == img
        w0, k = fit_line(pred[~test], gt[~test])
        out[test] = w0 + k * pred[test]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="greedy")
    args = parser.parse_args()

    per_item = pd.read_parquet(
        os.path.join(OUTPUTS_DIR, f"per_item_{args.mode}.parquet"))
    per_item = per_item[per_item["pred_weight"].notna()]

    rows = []
    for (model, temp, eps), items in per_item.groupby(
            ["model", "temperature", "epsilon"]):
        pred = items["pred_weight"].to_numpy(dtype=float)
        gt = items["gt_weight"].to_numpy(dtype=float)
        if len(items) < 10:
            continue
        w0, k = fit_line(pred, gt)
        insample = w0 + k * pred
        cv = loio_calibrated(items)
        rows.append({
            "model": model, "temperature": temp, "epsilon": eps, "n": len(items),
            "w0": round(w0, 2), "k": round(k, 3),
            "mae_raw": float(np.mean(np.abs(gt - pred))),
            "medae_raw": float(np.median(np.abs(gt - pred))),
            "mae_cal_insample": float(np.mean(np.abs(gt - insample))),
            "medae_cal_insample": float(np.median(np.abs(gt - insample))),
            "mae_cal_cv": float(np.mean(np.abs(gt - cv))),
            "medae_cal_cv": float(np.median(np.abs(gt - cv))),
        })
    df = pd.DataFrame(rows)
    out = os.path.join(OUTPUTS_DIR, f"calibration_{args.mode}.csv")
    df.to_csv(out, index=False)
    print(f"saved {out} ({len(df)} rows)")

    # Paper's Table 2 configuration: T=0.4, eps=0.6
    print("\n=== T=0.4, eps=0.6 (paper Table 2 config) ===")
    sel = df[(df.temperature == 0.4) & (df.epsilon == 0.6) &
             df.model.isin(["qwen2.5vl:7b", "gpt-4o", "gpt-4.1", "gpt-5.2"])]
    cols = ["model", "n", "w0", "k", "medae_raw", "medae_cal_insample",
            "medae_cal_cv", "mae_raw", "mae_cal_insample", "mae_cal_cv"]
    print(sel[cols].to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    gain_in = (df.medae_raw - df.medae_cal_insample)
    gain_cv = (df.medae_raw - df.medae_cal_cv)
    print(f"\nMedAE gain in-sample (mean over combos): {gain_in.mean():.1f} g")
    print(f"MedAE gain LOIO-CV   (mean over combos): {gain_cv.mean():.1f} g")


if __name__ == "__main__":
    main()
