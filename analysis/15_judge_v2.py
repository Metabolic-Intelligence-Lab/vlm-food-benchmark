"""WP6-ext - LLM-judge metrics for the 2026 campaign (model_outputs_v2) and
combined threshold-free ranking correlation over ALL models.

Mirrors 09_matcher_validation.part_c: the judge replaces the epsilon threshold
(scores < 0.30 -> different, >= 0.95 -> same, in-between -> LLM verdict) and
matching stays greedy one-to-one. v1 judge metrics are reused from
metrics_judge.csv; v2 combos are computed here with the same loader as the
official v2 evaluation (12_eval_all_v2.load_all_runs).

Outputs: analysis/outputs/metrics_judge_v2.csv, plus the combined Spearman
rho (judge-F1 vs Cross Encoder F1, best-T per model, r1) printed for the paper.

Usage: analysis_env/bin/python analysis/15_judge_v2.py
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluation import OUTPUTS_DIR, load_ground_truth

JUDGE_LO, JUDGE_HI = 0.30, 0.95

spec = importlib.util.spec_from_file_location(
    "eval_v2_mod", os.path.join(HERE, "12_eval_all_v2.py"))
m12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m12)


def main():
    verdicts = {}
    with open(os.path.join(OUTPUTS_DIR, "llm_judge_results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            verdicts[(r["pred_name"], r["gt_name"])] = r["verdict"] == "same"
    scores = m12.score_map()
    print(f"{len(verdicts)} verdetti, {len(scores)} coppie con score")

    def is_same(p, g):
        s = scores.get((p, g), 0.0)
        if s < JUDGE_LO:
            return False, s
        if s >= JUDGE_HI:
            return True, s
        v = verdicts.get((p, g))
        return (v if v is not None else False), s

    gt = load_ground_truth()
    runs = m12.load_all_runs()
    rows = []
    n_missing = 0
    for (model, temp, run), by_img in sorted(runs.items()):
        tp = fp = fn = 0
        for img, plist in by_img.items():
            gl = gt[img]
            cands = []
            for i, (pname, _) in enumerate(plist):
                for j, (gname, _) in enumerate(gl):
                    if (pname, gname) not in scores:
                        n_missing += 1
                    same, s = is_same(pname, gname)
                    if same:
                        cands.append((s, i, j))
            used_p, used_g = set(), set()
            for s, i, j in sorted(cands, reverse=True):
                if i not in used_p and j not in used_g:
                    used_p.add(i)
                    used_g.add(j)
            tp += len(used_p)
            fp += len(plist) - len(used_p)
            fn += len(gl) - len(used_g)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        rows.append({"model": model, "temperature": float(temp), "run": run,
                     "precision": round(prec, 4), "recall": round(rec, 4),
                     "f1": round(f1, 4)})
    dfv2 = pd.DataFrame(rows)
    dfv2.to_csv(os.path.join(OUTPUTS_DIR, "metrics_judge_v2.csv"), index=False)
    print(f"salvato metrics_judge_v2.csv ({len(dfv2)} righe, "
          f"{n_missing} coppie senza score)")

    # ---- combined ranking: judge-F1 (best T, r1) per model, v1 + v2 ----
    j1 = pd.read_csv(os.path.join(OUTPUTS_DIR, "metrics_judge.csv"))
    judge_best = j1.groupby("model")["f1"].max().to_dict()
    for m, f1 in dfv2[dfv2.run == "r1"].groupby("model")["f1"].max().items():
        judge_best.setdefault(m, f1)  # v1 value wins for overlapping models

    # Cross Encoder rankings: v1 clean + v2 (r1), best T per model
    clean = json.load(open(os.path.join(OUTPUTS_DIR, "metrics_clean.json")))
    from scipy.stats import spearmanr
    for eps in [0.5, 0.6, 0.7]:
        ce = {}
        for v in clean.values():
            if v["epsilon"] == eps:
                ce[v["model"]] = max(ce.get(v["model"], 0), v["f1"])
        v2m = pd.read_csv(os.path.join(OUTPUTS_DIR, "metrics_v2_all.csv"))
        sel = v2m[(v2m.run == "r1") & (v2m.epsilon == eps)]
        for m, f1 in sel.groupby("model")["f1"].max().items():
            ce.setdefault(m, f1)
        models = sorted(set(ce) & set(judge_best))
        rho = spearmanr([ce[m] for m in models],
                        [judge_best[m] for m in models]).statistic
        print(f"Spearman(judge, cross-encoder eps={eps}) = {rho:.3f} "
              f"({len(models)} modelli)")


if __name__ == "__main__":
    main()
