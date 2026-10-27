"""Recompute all benchmark metrics from the pair-score cache.

Evaluation modes:
  greedy   : one-to-one greedy matching by descending similarity (main results).
  hungarian: optimal assignment (robustness check on the matching algorithm).
  clean    : as greedy, but an empty raw response is treated as an empty
             prediction instead of being repaired (repair-step sensitivity).

Outputs (analysis/outputs/):
  metrics_<mode>.json          aggregate metrics per (model, T, eps)
  per_image_<mode>.parquet     per-image TP/FP/FN and portion errors
  per_item_<mode>.parquet      matched-item detail (gt vs pred weights)

Usage: python analysis/02_compute_metrics.py [--modes greedy,hungarian,clean]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import (MATCHERS, OUTPUTS_DIR, THRESHOLDS, PairScores,
                        complete_runs, extract_grams_robust, load_annotator_map,
                        load_ground_truth, load_run_predictions)

PAIR_SCORES = os.path.join(OUTPUTS_DIR, "pair_scores.parquet")

MODES = {
    # mode: (matcher_name, gram_parser, empty_raw_as_empty)
    "greedy": ("greedy", extract_grams_robust, False),
    "hungarian": ("hungarian", extract_grams_robust, False),
    # "clean" discards the output of the JSON-repair step for images where the
    # VLM produced an empty response: empty raw -> empty prediction list.
    "clean": ("greedy", extract_grams_robust, True),
}


def evaluate_mode(mode, runs, pair_scores, gt, annotator):
    matcher_name, gram_parser, empty_raw_as_empty = MODES[mode]
    matcher = MATCHERS[matcher_name]

    metrics = {}
    per_image_rows = []
    per_item_rows = []

    for (model, temp) in runs:
        preds_by_img = load_run_predictions(model, temp, empty_raw_as_empty)
        images = sorted(preds_by_img)

        # Pre-compute score matrices once per image (thresholds share them)
        mats = {}
        for img in images:
            pred_names = [n for n, _ in preds_by_img[img][0]]
            gt_names = [n for n, _ in gt[img]]
            mats[img] = pair_scores.matrix(pred_names, gt_names)

        for eps in THRESHOLDS:
            tp = fp = fn = 0
            gt_p, pred_p = [], []
            for img in images:
                pred_list, n_empty = preds_by_img[img]
                gt_list = gt[img]
                res = matcher(mats[img], eps)
                tp += res.tp
                fp += res.fp + n_empty
                fn += res.fn
                img_gt_w, img_pred_w = [], []
                for (pi, gi, score) in res.pairs:
                    grams = gram_parser(pred_list[pi][1])
                    if grams is not None:
                        gt_p.append(gt_list[gi][1])
                        pred_p.append(grams)
                        img_gt_w.append(gt_list[gi][1])
                        img_pred_w.append(grams)
                    per_item_rows.append({
                        "mode": mode, "model": model, "temperature": float(temp),
                        "epsilon": eps, "image": img,
                        "annotator": annotator.get(img, ""),
                        "gt_food": gt_list[gi][0], "gt_weight": gt_list[gi][1],
                        "pred_food": pred_list[pi][0], "pred_weight": grams,
                        "score": score,
                    })
                fp_weights = [
                    w for pi in res.fp_pred_idx
                    if (w := gram_parser(pred_list[pi][1])) is not None
                ]
                abs_err = [abs(a - b) for a, b in zip(img_gt_w, img_pred_w)]
                per_image_rows.append({
                    "mode": mode, "model": model, "temperature": float(temp),
                    "epsilon": eps, "image": img,
                    "annotator": annotator.get(img, ""),
                    "n_gt": len(gt_list), "n_pred": len(pred_list) + n_empty,
                    "tp": res.tp, "fp": res.fp + n_empty, "fn": res.fn,
                    "n_matched_with_weight": len(img_gt_w),
                    "sum_abs_err": float(np.sum(abs_err)) if abs_err else 0.0,
                    "sum_gt_weight_matched": float(np.sum(img_gt_w)) if img_gt_w else 0.0,
                    "sum_gt_weight_total": float(sum(w for _, w in gt_list)),
                    "n_fp_with_weight": len(fp_weights),
                    "sum_fp_pred_weight": float(np.sum(fp_weights)) if fp_weights else 0.0,
                })
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            jaccard = tp / (tp + fp + fn) if tp + fp + fn else 0.0

            gt_arr, pred_arr = np.array(gt_p), np.array(pred_p)
            if len(gt_arr):
                abs_err = np.abs(gt_arr - pred_arr)
                mae = float(np.mean(abs_err))
                medae = float(np.median(abs_err))
                rmse = float(np.sqrt(np.mean((gt_arr - pred_arr) ** 2)))
                mape = float(np.mean(abs_err / gt_arr) * 100)
                # NMAE (Eq. 1 of the paper): fixed 129 g penalty per unmatched GT item
                n_total_gt = sum(len(v) for v in gt.values())
                n_missed = n_total_gt - len(gt_arr)
                nmae_legacy = float((np.sum(abs_err) + 129.0 * n_missed) / n_total_gt)
            else:
                mae = medae = rmse = mape = nmae_legacy = None

            metrics[f"{model}_T{temp}_eps{eps}"] = {
                "model": model, "temperature": float(temp), "epsilon": eps,
                "tp": tp, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1": f1,
                "jaccard": jaccard, "n_matched_with_weight": int(len(gt_arr)),
                "mae": mae, "medae": medae, "rmse": rmse, "mape": mape,
                "nmae_legacy": nmae_legacy,
            }
        print(f"[{mode}] {model} T={temp} done")

    return metrics, pd.DataFrame(per_image_rows), pd.DataFrame(per_item_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="greedy,hungarian,clean")
    parser.add_argument("--pair-scores", default=PAIR_SCORES,
                        help="parquet with (pred_name, gt_name, score)")
    parser.add_argument("--out-suffix", default="",
                        help="suffix for output files, e.g. '_robertalarge'")
    args = parser.parse_args()
    modes = args.modes.split(",")

    runs = complete_runs()
    print(f"{len(runs)} complete (model, T) runs")
    pair_scores = PairScores(args.pair_scores)
    gt = load_ground_truth()
    annotator = load_annotator_map()

    for mode in modes:
        metrics, per_image, per_item = evaluate_mode(
            mode, runs, pair_scores, gt, annotator)
        tag = f"{mode}{args.out_suffix}"
        with open(os.path.join(OUTPUTS_DIR, f"metrics_{tag}.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        per_image.to_parquet(os.path.join(OUTPUTS_DIR, f"per_image_{tag}.parquet"),
                             index=False)
        per_item.to_parquet(os.path.join(OUTPUTS_DIR, f"per_item_{tag}.parquet"),
                            index=False)


if __name__ == "__main__":
    main()
