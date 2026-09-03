"""WP8/B4 - Evaluate a model run from model_outputs_v2/ (new-campaign outputs).

Handles the v2 filename pattern NNN_<model>_T<t>_user01_sys01_<runid>.json,
scores any (pred, gt) pairs missing from the cache with the same CrossEncoder,
appends them to pair_scores_v2.parquet (main cache untouched), and reports
F1/portion metrics with the standard greedy matcher across the epsilon grid.

Usage: analysis_env/bin/python analysis/11_eval_v2.py --model claude-sonnet-5 \
           --temp 1.0 --run r1 [--compare gpt-4o]
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import (OUTPUTS_DIR, REPO, THRESHOLDS, load_and_normalize_preds,
                        load_ground_truth, match_greedy, normalize,
                        extract_grams_robust)

V2_DIR = os.path.join(REPO, "model_outputs_v2")
V2_CACHE = os.path.join(OUTPUTS_DIR, "pair_scores_v2.parquet")
MAIN_CACHE = os.path.join(OUTPUTS_DIR, "pair_scores.parquet")
CROSS_ENCODER = "cross-encoder/stsb-distilroberta-base"


def load_v2_predictions(model, temp, run):
    pat = re.compile(rf"^(\d{{3}})_{re.escape(model)}_T{re.escape(temp)}"
                     rf"_user01_sys01_{re.escape(run)}\.json$")
    out = {}
    for fname in sorted(os.listdir(V2_DIR)):
        m = pat.match(fname)
        if not m:
            continue
        preds, _ = load_and_normalize_preds(os.path.join(V2_DIR, fname))
        img = int(m.group(1))
        out[img] = [(normalize(p.get("name", "")), str(p.get("portion_estimate", "")))
                    for p in preds if normalize(p.get("name", ""))]
    return out


def score_map():
    dfs = [pd.read_parquet(MAIN_CACHE)]
    if os.path.exists(V2_CACHE):
        dfs.append(pd.read_parquet(V2_CACHE))
    df = pd.concat(dfs)
    return dict(zip(zip(df.pred_name, df.gt_name), df.score))


def ensure_scores(preds_by_img, gt, scores):
    needed = set()
    for img, plist in preds_by_img.items():
        for pname, _ in plist:
            for gname, _ in gt[img]:
                if (pname, gname) not in scores:
                    needed.add((pname, gname))
    if not needed:
        return scores
    print(f"scoring {len(needed)} new pairs...")
    import torch
    from sentence_transformers.cross_encoder import CrossEncoder
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CrossEncoder(CROSS_ENCODER, device=device)
    pairs = sorted(needed)
    vals = model.predict([list(p) for p in pairs], batch_size=256)
    new_df = pd.DataFrame(pairs, columns=["pred_name", "gt_name"])
    new_df["score"] = np.asarray(vals, dtype=np.float32)
    if os.path.exists(V2_CACHE):
        new_df = pd.concat([pd.read_parquet(V2_CACHE), new_df])
    new_df.to_parquet(V2_CACHE, index=False)
    scores.update(dict(zip(zip(new_df.pred_name, new_df.gt_name), new_df.score)))
    return scores


def evaluate(preds_by_img, gt, scores):
    n_tot = sum(len(v) for v in gt.values())
    rows = []
    for eps in THRESHOLDS:
        tp = fp = fn = 0
        gt_w, pred_w, missed_w = [], [], 0.0
        for img, plist in preds_by_img.items():
            gl = gt[img]
            mat = np.zeros((len(plist), len(gl)), dtype=np.float32)
            for i, (p, _) in enumerate(plist):
                for j, (g, _) in enumerate(gl):
                    mat[i, j] = scores[(p, g)]
            res = match_greedy(mat, eps)
            tp += res.tp
            fp += res.fp
            fn += res.fn
            matched_w = set()
            for (pi, gi, _) in res.pairs:
                grams = extract_grams_robust(plist[pi][1])
                if grams is not None:
                    gt_w.append(gl[gi][1])
                    pred_w.append(grams)
                    matched_w.add(gi)
            missed_w += sum(w for j, (_, w) in enumerate(gl) if j not in matched_w)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        gt_w_a, pred_w_a = np.array(gt_w), np.array(pred_w)
        abs_err = np.abs(gt_w_a - pred_w_a) if len(gt_w_a) else np.array([])
        rows.append({
            "epsilon": eps, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(f1, 3),
            "n_matched_w": len(gt_w_a),
            "mae": round(float(abs_err.mean()), 1) if len(abs_err) else None,
            "medae": round(float(np.median(abs_err)), 1) if len(abs_err) else None,
            "cpmae": round(float((abs_err.sum() + missed_w) / n_tot), 1),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--temp", default="1.0")
    parser.add_argument("--run", default="r1")
    parser.add_argument("--compare", default="gpt-4o")
    args = parser.parse_args()

    gt = load_ground_truth()
    preds = load_v2_predictions(args.model, args.temp, args.run)
    print(f"{args.model} T{args.temp} {args.run}: {len(preds)} images, "
          f"{sum(len(v) for v in preds.values())} predictions")
    if len(preds) < 200:
        print(f"WARNING: incomplete run ({len(preds)}/200)")

    scores = ensure_scores(preds, gt, score_map())
    df = evaluate(preds, gt, scores)
    print(df.to_string(index=False))
    out_csv = os.path.join(OUTPUTS_DIR,
                           f"metrics_v2_{args.model}_T{args.temp}_{args.run}.csv")
    df.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")

    if args.compare:
        with open(os.path.join(OUTPUTS_DIR, "metrics_clean.json")) as f:
            clean = json.load(f)
        print(f"\n=== reference ({args.compare}, clean mode, best T per eps) ===")
        for eps in THRESHOLDS:
            best = max((v for v in clean.values()
                        if v["model"] == args.compare and v["epsilon"] == eps),
                       key=lambda v: v["f1"], default=None)
            if best:
                print(f"  eps={eps}: F1={best['f1']:.3f} "
                      f"(T={best['temperature']}), mae={best['mae'] and round(best['mae'], 1)}")


if __name__ == "__main__":
    main()
