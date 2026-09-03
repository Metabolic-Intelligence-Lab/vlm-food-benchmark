"""WP2 - Revised portion metrics and reference baselines.

Metrics per (model, T, eps), computed from per_image_greedy.parquet:
  mae, medae            over matched items with a parsed weight (as before)
  nmae_legacy           (sum|err| + 129 g * n_missed) / N_tot   [paper metric]
  cpmae                 (sum|err| + sum of TRUE GT weights of missed items) / N_tot
  cpmae_fp              cpmae + spurious predicted mass of false positives / N_tot

"missed" = GT item without a matched, parseable portion estimate (same
definition as the paper). cpmae fixes the two criticisms: the penalty is the
item's actual weight (not an arbitrary constant), and cpmae_fp additionally
penalizes hallucinated items.

Reference baselines (model-independent, answer "do VLMs beat priors?"):
  null            predict nothing        -> cpmae = mean GT weight
  global_median   every GT item matched, weight = median(129 g)
  typical_serving every GT item matched, weight = per-food median computed
                  leave-one-image-out (global median for foods seen only once)

Usage: analysis_env/bin/python analysis/03_portion_metrics.py [--mode greedy]
Outputs: analysis/outputs/portion_metrics_<mode>.csv, portion_baselines.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import OUTPUTS_DIR, load_ground_truth


def compute_baselines():
    gt = load_ground_truth()
    items = [(img, food, w) for img, lst in gt.items() for food, w in lst]
    weights = np.array([w for _, _, w in items])
    n_tot = len(items)
    global_median = float(np.median(weights))

    by_food = defaultdict(list)
    for img, food, w in items:
        by_food[food].append((img, w))

    typical_preds = []
    n_fallback = 0
    for img, food, w in items:
        others = [ow for oimg, ow in by_food[food] if oimg != img]
        if others:
            typical_preds.append(float(np.median(others)))
        else:
            typical_preds.append(global_median)
            n_fallback += 1
    typical_preds = np.array(typical_preds)

    baselines = {
        "n_total_gt_items": n_tot,
        "gt_mean_weight_g": float(np.mean(weights)),
        "gt_median_weight_g": global_median,
        "null_predictor": {
            "cpmae": float(np.mean(weights)),  # every item missed at its true weight
            "nmae_legacy": 129.0,
            "mae": None,
        },
        "global_median_predictor": {
            "mae": float(np.mean(np.abs(weights - global_median))),
            "medae": float(np.median(np.abs(weights - global_median))),
            "cpmae": float(np.mean(np.abs(weights - global_median))),
        },
        "typical_serving_predictor": {
            "mae": float(np.mean(np.abs(weights - typical_preds))),
            "medae": float(np.median(np.abs(weights - typical_preds))),
            "cpmae": float(np.mean(np.abs(weights - typical_preds))),
            "n_fallback_to_global_median": n_fallback,
            "note": "per-food median, leave-one-image-out; assumes perfect recognition",
        },
    }
    return baselines


def compute_model_metrics(mode):
    per_image = pd.read_parquet(os.path.join(OUTPUTS_DIR, f"per_image_{mode}.parquet"))
    per_item = pd.read_parquet(os.path.join(OUTPUTS_DIR, f"per_item_{mode}.parquet"))
    n_tot = int(per_image.groupby(["model", "temperature", "epsilon"])["n_gt"]
                .sum().iloc[0])

    rows = []
    grouped = per_image.groupby(["model", "temperature", "epsilon"])
    item_grouped = per_item[per_item["pred_weight"].notna()].groupby(
        ["model", "temperature", "epsilon"])
    for (model, temp, eps), g in grouped:
        sum_abs_err = g["sum_abs_err"].sum()
        n_matched = int(g["n_matched_with_weight"].sum())
        missed_weight = (g["sum_gt_weight_total"] - g["sum_gt_weight_matched"]).sum()
        fp_weight = g["sum_fp_pred_weight"].sum()
        n_missed = n_tot - n_matched
        try:
            items = item_grouped.get_group((model, temp, eps))
            abs_err = (items["gt_weight"] - items["pred_weight"]).abs()
            mae = float(abs_err.mean())
            medae = float(abs_err.median())
        except KeyError:
            mae = medae = None
        rows.append({
            "model": model, "temperature": temp, "epsilon": eps,
            "n_matched_with_weight": n_matched, "n_missed": n_missed,
            "mae": mae, "medae": medae,
            "nmae_legacy": float((sum_abs_err + 129.0 * n_missed) / n_tot),
            "cpmae": float((sum_abs_err + missed_weight) / n_tot),
            "cpmae_fp": float((sum_abs_err + missed_weight + fp_weight) / n_tot),
        })
    return pd.DataFrame(rows), n_tot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="greedy")
    args = parser.parse_args()

    baselines = compute_baselines()
    with open(os.path.join(OUTPUTS_DIR, "portion_baselines.json"), "w") as f:
        json.dump(baselines, f, indent=2)

    print("=== BASELINES ===")
    print(json.dumps(baselines, indent=2))

    df, n_tot = compute_model_metrics(args.mode)
    out_csv = os.path.join(OUTPUTS_DIR, f"portion_metrics_{args.mode}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv} ({len(df)} rows, N_tot={n_tot})")

    print(f"\n=== TOP 15 BY cpmae (mode={args.mode}) ===")
    cols = ["model", "temperature", "epsilon", "n_matched_with_weight",
            "mae", "medae", "nmae_legacy", "cpmae", "cpmae_fp"]
    top = df.sort_values("cpmae").head(15)[cols]
    print(top.to_string(index=False,
                        float_format=lambda x: f"{x:.1f}"))
    null_cp = baselines["null_predictor"]["cpmae"]
    ts_cp = baselines["typical_serving_predictor"]["cpmae"]
    print(f"\nnull predictor cpmae:            {null_cp:.1f} g")
    print(f"typical-serving predictor cpmae: {ts_cp:.1f} g "
          f"(assumes perfect recognition)")


if __name__ == "__main__":
    main()
