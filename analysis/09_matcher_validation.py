"""WP6 - Matcher validation analysis. Three parts, each
running only when its inputs are available:

A. Human labels vs cross-encoder score: inter-judge agreement (Cohen's kappa),
   ROC AUC of the score against the majority human label, data-driven optimal
   threshold (Youden's J). Inputs: analysis/outputs/human_labels_*.csv
   (from the form built by 07_build_human_eval.py) + human_eval_key.csv.

B. Second cross-encoder: Spearman correlation of model rankings
   (metrics_greedy.json vs metrics_greedy_robertalarge.json).

C. LLM judge (threshold-free evaluator): recompute F1 per (model, T) using
   same/different verdicts (llm_judge_results.jsonl; outside the judged band,
   score < lo -> different, >= hi -> same), then Spearman vs cross-encoder
   rankings. Output: metrics_judge.csv.

Usage: analysis_env/bin/python analysis/09_matcher_validation.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import (OUTPUTS_DIR, complete_runs, load_ground_truth,
                        load_run_predictions)

JUDGE_LO, JUDGE_HI = 0.30, 0.95


# ---------------------------------------------------------------- part A

def part_a():
    files = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "human_labels_*.csv")))
    if not files:
        print("A. human labels: none found (waiting for judges) - skipped")
        return
    key = pd.read_csv(os.path.join(OUTPUTS_DIR, "human_eval_key.csv"))
    labels = pd.concat([pd.read_csv(f) for f in files])
    labels = labels[labels.label.isin(["same", "partial", "different"])]
    print(f"A. {len(files)} judges, {len(labels)} labels")

    # inter-judge agreement (pairwise Cohen's kappa, same/partial/different)
    from itertools import combinations
    piv = labels.pivot_table(index="pair_id", columns="judge", values="label",
                             aggfunc="first")
    from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve
    for j1, j2 in combinations(piv.columns, 2):
        both = piv[[j1, j2]].dropna()
        if len(both) > 10:
            k = cohen_kappa_score(both[j1], both[j2])
            print(f"   kappa({j1}, {j2}) = {k:.3f}  (n={len(both)})")

    # majority vote, binarized: same=1, partial/different=0 (strict) and
    # same/partial=1 (lenient)
    def majority(s):
        return s.mode().iloc[0]

    maj = piv.apply(majority, axis=1).rename("majority").reset_index()
    merged = key.merge(maj, on="pair_id")
    for scheme, positive in [("strict (same only)", ["same"]),
                             ("lenient (same+partial)", ["same", "partial"])]:
        y = merged.majority.isin(positive).astype(int)
        if y.nunique() < 2:
            continue
        auc = roc_auc_score(y, merged.score)
        fpr, tpr, thr = roc_curve(y, merged.score)
        youden = thr[np.argmax(tpr - fpr)]
        print(f"   {scheme}: AUC = {auc:.3f}, optimal threshold (Youden) = {youden:.2f}")


# ---------------------------------------------------------------- part B

def ranking_from_metrics(path, eps):
    with open(path) as f:
        metrics = json.load(f)
    best = {}
    for v in metrics.values():
        if v["epsilon"] != eps:
            continue
        m = v["model"]
        if m not in best or v["f1"] > best[m]:
            best[m] = v["f1"]
    return best


def part_b():
    p1 = os.path.join(OUTPUTS_DIR, "metrics_greedy.json")
    p2 = os.path.join(OUTPUTS_DIR, "metrics_greedy_robertalarge.json")
    if not (os.path.exists(p1) and os.path.exists(p2)):
        print("B. roberta-large metrics not found - skipped")
        return
    from scipy.stats import spearmanr
    print("B. ranking agreement distilroberta vs roberta-large (best-T F1):")
    for eps in [0.5, 0.6, 0.7, 0.8, 0.9]:
        r1 = ranking_from_metrics(p1, eps)
        r2 = ranking_from_metrics(p2, eps)
        models = sorted(set(r1) & set(r2))
        rho = spearmanr([r1[m] for m in models], [r2[m] for m in models]).statistic
        print(f"   eps={eps}: Spearman rho = {rho:.3f}  ({len(models)} models)")


# ---------------------------------------------------------------- part C

def part_c():
    jl = os.path.join(OUTPUTS_DIR, "llm_judge_results.jsonl")
    if not os.path.exists(jl):
        print("C. LLM judge results not found - skipped")
        return
    verdicts = {}
    with open(jl) as f:
        for line in f:
            r = json.loads(line)
            verdicts[(r["pred_name"], r["gt_name"])] = r["verdict"] == "same"
    scores = pd.read_parquet(os.path.join(OUTPUTS_DIR, "pair_scores.parquet"))
    score_map = dict(zip(zip(scores.pred_name, scores.gt_name), scores.score))
    print(f"C. LLM judge: {len(verdicts)} judged pairs")

    def is_same(p, g):
        s = score_map.get((p, g), 0.0)
        if s < JUDGE_LO:
            return False, s
        if s >= JUDGE_HI:
            return True, s
        v = verdicts.get((p, g))
        return (v if v is not None else False), s

    gt = load_ground_truth()
    rows = []
    for model, temp in complete_runs():
        preds = load_run_predictions(model, temp, empty_raw_as_empty=True)
        tp = fp = fn = 0
        for img, (plist, n_empty) in preds.items():
            gt_names = [n for n, _ in gt[img]]
            cands = []
            for i, (pname, _) in enumerate(plist):
                for j, gname in enumerate(gt_names):
                    same, s = is_same(pname, gname)
                    if same:
                        cands.append((s, i, j))
            used_p, used_g = set(), set()
            for s, i, j in sorted(cands, reverse=True):
                if i not in used_p and j not in used_g:
                    used_p.add(i)
                    used_g.add(j)
            tp += len(used_p)
            fp += len(plist) - len(used_p) + n_empty
            fn += len(gt_names) - len(used_g)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        rows.append({"model": model, "temperature": temp,
                     "precision": prec, "recall": rec, "f1": f1})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUTS_DIR, "metrics_judge.csv"), index=False)

    from scipy.stats import spearmanr
    best_judge = df.groupby("model")["f1"].max()
    print("   top 8 by judge-F1 (threshold-free):")
    print(best_judge.sort_values(ascending=False).head(8).round(3).to_string())
    for eps in [0.5, 0.6, 0.7]:
        ce = ranking_from_metrics(os.path.join(OUTPUTS_DIR, "metrics_greedy.json"), eps)
        models = sorted(set(ce) & set(best_judge.index))
        rho = spearmanr([ce[m] for m in models],
                        [best_judge[m] for m in models]).statistic
        print(f"   Spearman(judge, cross-encoder eps={eps}) = {rho:.3f}")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
