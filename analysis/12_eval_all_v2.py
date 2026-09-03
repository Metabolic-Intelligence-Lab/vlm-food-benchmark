"""B4 - Batch evaluation of every (model, T, run) in model_outputs_v2/.

One pass: collect all missing (pred, gt) pairs across all v2 runs, score them
once with the CrossEncoder (appended to pair_scores_v2.parquet), then compute
F1/portion metrics per combo with the standard greedy matcher, plus per-image
and per-item tables for bootstrap and portion analyses.

Outputs (analysis/outputs/):
  metrics_v2_all.csv       per (model, T, run, epsilon)
  per_image_v2.parquet     per-image TP/FP/FN + portion sums (bootstrap-ready)
  per_item_v2.parquet      matched items with weights
Usage: analysis_env/bin/python analysis/12_eval_all_v2.py
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import (OUTPUTS_DIR, REPO, THRESHOLDS, extract_grams_robust,
                        load_and_normalize_preds, load_annotator_map,
                        load_ground_truth, match_greedy, normalize)

V2_DIR = os.path.join(REPO, "model_outputs_v2")
V2_CACHE = os.path.join(OUTPUTS_DIR, "pair_scores_v2.parquet")
MAIN_CACHE = os.path.join(OUTPUTS_DIR, "pair_scores.parquet")
PAT = re.compile(r"^(\d{3})_(.+?)_T([\d.]+)_user01_sys01_(r\w+)\.json$")


def load_all_runs():
    runs = {}
    for fname in sorted(os.listdir(V2_DIR)):
        m = PAT.match(fname)
        if not m:
            continue
        img, model, temp, run = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        preds, _ = load_and_normalize_preds(os.path.join(V2_DIR, fname))
        named = [(normalize(p.get("name", "")), str(p.get("portion_estimate", "")))
                 for p in preds if normalize(p.get("name", ""))]
        runs.setdefault((model, temp, run), {})[img] = named
    return runs


def score_map():
    dfs = [pd.read_parquet(MAIN_CACHE)]
    if os.path.exists(V2_CACHE):
        dfs.append(pd.read_parquet(V2_CACHE))
    df = pd.concat(dfs).drop_duplicates(["pred_name", "gt_name"])
    return dict(zip(zip(df.pred_name, df.gt_name), df.score))


def ensure_scores(runs, gt, scores):
    needed = set()
    for combo, by_img in runs.items():
        for img, plist in by_img.items():
            for pname, _ in plist:
                for gname, _ in gt[img]:
                    if (pname, gname) not in scores:
                        needed.add((pname, gname))
    if not needed:
        print("nessuna coppia nuova da valutare")
        return scores
    print(f"scoring di {len(needed)} coppie nuove...")
    import torch
    from sentence_transformers.cross_encoder import CrossEncoder
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CrossEncoder("cross-encoder/stsb-distilroberta-base", device=device)
    pairs = sorted(needed)
    vals = model.predict([list(p) for p in pairs], batch_size=256,
                         show_progress_bar=True)
    new_df = pd.DataFrame(pairs, columns=["pred_name", "gt_name"])
    new_df["score"] = np.asarray(vals, dtype=np.float32)
    if os.path.exists(V2_CACHE):
        new_df = pd.concat([pd.read_parquet(V2_CACHE), new_df]) \
                   .drop_duplicates(["pred_name", "gt_name"])
    new_df.to_parquet(V2_CACHE, index=False)
    scores.update(dict(zip(zip(new_df.pred_name, new_df.gt_name), new_df.score)))
    return scores


def main():
    gt = load_ground_truth()
    annotator = load_annotator_map()
    runs = load_all_runs()
    print(f"{len(runs)} combinazioni (model, T, run)")
    scores = ensure_scores(runs, gt, score_map())

    met_rows, img_rows, item_rows = [], [], []
    for (model, temp, run), by_img in sorted(runs.items()):
        # score matrices per image
        for eps in THRESHOLDS:
            tp = fp = fn = 0
            gt_w_all, pred_w_all = [], []
            for img, plist in sorted(by_img.items()):
                gl = gt[img]
                mat = np.zeros((len(plist), len(gl)), dtype=np.float32)
                for i, (p, _) in enumerate(plist):
                    for j, (g, _) in enumerate(gl):
                        mat[i, j] = scores[(p, g)]
                res = match_greedy(mat, eps)
                tp += res.tp; fp += res.fp; fn += res.fn
                img_gt_w, img_pred_w = [], []
                for (pi, gi, sc) in res.pairs:
                    grams = extract_grams_robust(plist[pi][1])
                    if grams is not None:
                        img_gt_w.append(gl[gi][1]); img_pred_w.append(grams)
                    item_rows.append({
                        "model": model, "temperature": float(temp), "run": run,
                        "epsilon": eps, "image": img,
                        "gt_food": gl[gi][0], "gt_weight": gl[gi][1],
                        "pred_food": plist[pi][0], "pred_weight": grams,
                        "score": sc})
                gt_w_all += img_gt_w; pred_w_all += img_pred_w
                abs_err = [abs(a - b) for a, b in zip(img_gt_w, img_pred_w)]
                img_rows.append({
                    "model": model, "temperature": float(temp), "run": run,
                    "epsilon": eps, "image": img,
                    "annotator": annotator.get(img, ""),
                    "n_gt": len(gl), "n_pred": len(plist),
                    "tp": res.tp, "fp": res.fp, "fn": res.fn,
                    "n_matched_with_weight": len(img_gt_w),
                    "sum_abs_err": float(np.sum(abs_err)) if abs_err else 0.0,
                    "sum_gt_weight_matched": float(np.sum(img_gt_w)) if img_gt_w else 0.0,
                    "sum_gt_weight_total": float(sum(w for _, w in gl)),
                })
            n_tot = sum(len(gt[i]) for i in by_img)
            prec = tp / (tp + fp) if tp + fp else 0
            rec = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            gtA, prA = np.array(gt_w_all), np.array(pred_w_all)
            abs_err = np.abs(gtA - prA) if len(gtA) else np.array([])
            missed_w = sum(sum(w for _, w in gt[i]) for i in by_img) - gtA.sum() if len(gtA) else None
            met_rows.append({
                "model": model, "temperature": float(temp), "run": run,
                "epsilon": eps, "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 4), "recall": round(rec, 4),
                "f1": round(f1, 4),
                "n_matched_w": int(len(gtA)),
                "mae": round(float(abs_err.mean()), 2) if len(abs_err) else None,
                "medae": round(float(np.median(abs_err)), 2) if len(abs_err) else None,
                "rmse": round(float(np.sqrt((abs_err**2).mean())), 2) if len(abs_err) else None,
                "mape": round(float((abs_err / gtA).mean() * 100), 2) if len(abs_err) else None,
                "nmae_legacy": round(float((abs_err.sum() + 129.0 * (n_tot - len(gtA))) / n_tot), 2) if len(abs_err) else None,
                "cpmae": round(float((abs_err.sum() + (sum(sum(w for _, w in gt[i]) for i in by_img) - gtA.sum())) / n_tot), 2) if len(abs_err) else None,
            })
        print(f"{model} T{temp} {run} ok")

    pd.DataFrame(met_rows).to_csv(os.path.join(OUTPUTS_DIR, "metrics_v2_all.csv"), index=False)
    pd.DataFrame(img_rows).to_parquet(os.path.join(OUTPUTS_DIR, "per_image_v2.parquet"), index=False)
    pd.DataFrame(item_rows).to_parquet(os.path.join(OUTPUTS_DIR, "per_item_v2.parquet"), index=False)
    print("salvati metrics_v2_all.csv, per_image_v2.parquet, per_item_v2.parquet")


if __name__ == "__main__":
    main()
