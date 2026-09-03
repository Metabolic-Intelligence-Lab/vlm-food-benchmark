"""WP1 - Recompute all benchmark metrics from the pair-score cache.

Three evaluation modes, to decompose the effect of each fix:
  paper   : legacy GT mapping (47 wrong images) + legacy matching + legacy FN
            double-count bug -> must reproduce the submitted paper's numbers.
  fixed   : correct GT mapping + legacy matching (no FN bug) -> isolates the
            impact of the GT-mapping bug.
  greedy  : correct GT mapping + one-to-one greedy matching -> new standard.
  hungarian: correct GT mapping + optimal assignment -> robustness check.

Outputs (analysis/outputs/):
  metrics_<mode>.json          aggregate metrics per (model, T, eps)
  per_image_<mode>.parquet     per-image TP/FP/FN and portion errors
  per_item_<mode>.parquet      matched-item detail (gt vs pred weights)
  comparison_paper_vs_published.csv   sanity check vs model_performance/*.json

Usage: analysis_env/bin/python analysis/02_compute_metrics.py [--modes paper,fixed,greedy,hungarian]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import (MATCHERS, OUTPUTS_DIR, REPO, THRESHOLDS, PairScores,
                        complete_runs, extract_grams_legacy,
                        extract_grams_robust, load_annotator_map,
                        load_ground_truth, load_ground_truth_legacy,
                        load_run_predictions)

PAIR_SCORES = os.path.join(OUTPUTS_DIR, "pair_scores.parquet")

MODES = {
    # mode: (gt_loader, matcher_name, legacy_fn_bug, gram_parser, empty_raw_as_empty)
    "paper": ("legacy", "legacy", True, extract_grams_legacy, False),
    "fixed": ("correct", "legacy", False, extract_grams_legacy, False),
    "greedy": ("correct", "greedy", False, extract_grams_robust, False),
    "hungarian": ("correct", "hungarian", False, extract_grams_robust, False),
    # "clean" additionally discards repair-LLM output for images where the VLM
    # produced an EMPTY response (the repair step hallucinated ~3 items in 93%
    # of those 1279 cases): empty raw -> empty prediction list.
    "clean": ("correct", "greedy", False, extract_grams_robust, True),
}


def evaluate_mode(mode, runs, pair_scores, gt_correct, gt_legacy, annotator):
    gt_kind, matcher_name, legacy_fn_bug, gram_parser, empty_raw_as_empty = MODES[mode]
    gt = gt_legacy if gt_kind == "legacy" else gt_correct
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
            last_img_unmatched_fn = 0
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
                last_img_unmatched_fn = res.fn  # survives loop for legacy bug
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
            if legacy_fn_bug:
                # compute_performance.py:200-205 re-added the unmatched GT items
                # of the LAST image a second time, outside the file loop
                fn += last_img_unmatched_fn

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
                # legacy NMAE: fixed 129 g penalty per unmatched GT item
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


def compare_with_published(metrics_paper):
    """Compare 'paper' mode with the JSONs used for the submitted manuscript."""
    rows = []
    for eps in THRESHOLDS:
        path = os.path.join(REPO, "model_performance",
                            f"evaluation_metrics_CrossEncThr{eps}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            published = json.load(f)
        for key, pub in published.items():
            model, temp = key.rsplit("_T", 1)
            ours = metrics_paper.get(f"{model}_T{temp}_eps{eps}")
            if ours is None:
                continue
            rows.append({
                "model": model, "temperature": temp, "epsilon": eps,
                "f1_published": round(pub["f1"], 4),
                "f1_repro": round(ours["f1"], 4),
                "f1_delta": round(ours["f1"] - pub["f1"], 4),
                "mae_published": round(pub["mae"], 2) if pub["mae"] is not None else None,
                "mae_repro": round(ours["mae"], 2) if ours["mae"] is not None else None,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="paper,fixed,greedy,hungarian")
    parser.add_argument("--pair-scores", default=PAIR_SCORES,
                        help="parquet with (pred_name, gt_name, score)")
    parser.add_argument("--out-suffix", default="",
                        help="suffix for output files, e.g. '_robertalarge'")
    args = parser.parse_args()
    modes = args.modes.split(",")

    runs = complete_runs()
    print(f"{len(runs)} complete (model, T) runs")
    pair_scores = PairScores(args.pair_scores)
    gt_correct = load_ground_truth()
    gt_legacy = load_ground_truth_legacy()
    annotator = load_annotator_map()

    for mode in modes:
        metrics, per_image, per_item = evaluate_mode(
            mode, runs, pair_scores, gt_correct, gt_legacy, annotator)
        tag = f"{mode}{args.out_suffix}"
        with open(os.path.join(OUTPUTS_DIR, f"metrics_{tag}.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        per_image.to_parquet(os.path.join(OUTPUTS_DIR, f"per_image_{tag}.parquet"),
                             index=False)
        per_item.to_parquet(os.path.join(OUTPUTS_DIR, f"per_item_{tag}.parquet"),
                            index=False)
        if mode == "paper":
            cmp_df = compare_with_published(metrics)
            cmp_path = os.path.join(OUTPUTS_DIR, "comparison_paper_vs_published.csv")
            cmp_df.to_csv(cmp_path, index=False)
            print(f"\n=== reproduction check (paper mode vs published JSONs) ===")
            print(f"rows: {len(cmp_df)}, max |f1 delta|: {cmp_df['f1_delta'].abs().max()}")
            print(cmp_df["f1_delta"].abs().describe())


if __name__ == "__main__":
    main()
