"""Primary threshold and nature of the accepted matches.

1. Human-optimal thresholds (Youden's J vs the panel majority, strict = "same",
   lenient = "same" + "partial") with bootstrap 95% CIs over the 300 pairs.
2. For every model (Table 2 configurations), the estimated composition of the
   matches accepted at the primary threshold eps=0.5: exact textual matches, and
   the shares of same / partial / different foods, obtained by weighting the
   majority-label distribution of each similarity-score interval of the human
   validation sample by the model's matches in that interval (assumes that the
   label distribution given the score does not depend on the model).
3. Strict F1 (only expected same-food matches count as TP), and F1 at eps =
   0.55 / 0.6, with Spearman correlations against the primary ranking.

Usage: python analysis/21_matcher_composition.py
Outputs: analysis/outputs/matcher_thresholds.csv, matcher_composition.csv
"""
import glob
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluation import (N_IMAGES, OUTPUTS_DIR, load_ground_truth,
                        load_run_predictions, match_greedy)

spec = importlib.util.spec_from_file_location(
    "m12", os.path.join(HERE, "12_eval_all_v2.py"))
m12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m12)

BIN_EDGES = [-10, .3, .4, .5, .6, .7, .8, .9, 10]
SEED = 20260901


def human_majority():
    key = pd.read_csv(os.path.join(OUTPUTS_DIR, "human_eval_key.csv"))
    labels = pd.concat(pd.read_csv(f) for f in
                       sorted(glob.glob(os.path.join(OUTPUTS_DIR, "human_labels_*.csv"))))
    piv = labels.pivot_table(index="pair_id", columns="judge", values="label", aggfunc="first")
    # same majority rule as 09_matcher_validation.py
    maj = piv.apply(lambda r: r.mode().iloc[0], axis=1).rename("majority")
    return key.merge(maj, left_on="pair_id", right_index=True)


def youden_thresholds(k):
    rng = np.random.default_rng(SEED)
    rows = []
    for scheme, positive in [("strict", ["same"]), ("lenient", ["same", "partial"])]:
        y = k.majority.isin(positive).astype(int).to_numpy()
        sc = k.score.to_numpy()
        fpr, tpr, thr = roc_curve(y, sc)
        boot = []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() == y[i].max():
                continue
            f, t, th = roc_curve(y[i], sc[i])
            boot.append(th[np.argmax(t - f)])
        rows.append({"scheme": scheme, "auc": roc_auc_score(y, sc),
                     "youden_threshold": thr[np.argmax(tpr - fpr)],
                     "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5)})
    return pd.DataFrame(rows)


def main():
    k = human_majority()
    thr = youden_thresholds(k)
    thr.to_csv(os.path.join(OUTPUTS_DIR, "matcher_thresholds.csv"), index=False)
    print(thr.round(3).to_string(index=False))

    k["bin"] = pd.cut(k.score, BIN_EDGES)
    comp = pd.crosstab(k.bin, k.majority, normalize="index")
    p_same = comp["same"].to_dict()
    p_partial = comp["partial"].to_dict()

    gt = load_ground_truth()
    scores = m12.score_map()
    runs_v2 = m12.load_all_runs()
    acc = pd.read_csv(os.path.join(OUTPUTS_DIR, "portion_accounting.csv"))
    v1_models = set(pd.read_parquet(os.path.join(OUTPUTS_DIR, "per_image_clean.parquet"),
                                    columns=["model"]).model)
    rows = []
    for r in acc.itertuples():
        temp = f"{r.temperature:.1f}"
        if r.model in v1_models:
            preds = load_run_predictions(r.model, temp, empty_raw_as_empty=True)
            by_img = {i: pl for i, (pl, _) in preds.items()}
            n_empty = sum(e for _, e in preds.values())
        else:
            by_img, n_empty = runs_v2[(r.model, temp, "r1")], 0
        out = {"model": r.model, "temperature": r.temperature}
        for eps in (0.5, 0.55, 0.6):
            tp = fp = fn = exact = 0
            e_same = e_partial = 0.0
            for img in range(1, N_IMAGES + 1):
                pl, gl = by_img.get(img, []), gt[img]
                mat = np.array([[scores[(p, g)] for g, _ in gl] for p, _ in pl],
                               dtype=np.float32).reshape(len(pl), len(gl))
                res = match_greedy(mat, eps)
                tp, fp, fn = tp + res.tp, fp + res.fp, fn + res.fn
                if eps == 0.5:
                    for pi, gi, sc in res.pairs:
                        exact += pl[pi][0] == gl[gi][0]
                        b = pd.cut([sc], BIN_EDGES)[0]
                        e_same += p_same[b]
                        e_partial += p_partial[b]
            fp += n_empty
            out[f"f1_eps{eps}"] = 2 * tp / (2 * tp + fp + fn)
            if eps == 0.5:
                out.update({
                    "tp": tp, "share_exact": exact / tp, "share_same": e_same / tp,
                    "share_partial": e_partial / tp,
                    "share_different": 1 - (e_same + e_partial) / tp,
                    # strict: expected non-same matches become FP + FN
                    "f1_strict": 2 * e_same / (2 * e_same + (fp + tp - e_same) + (fn + tp - e_same)),
                })
        rows.append(out)
    df = pd.DataFrame(rows).sort_values("f1_eps0.5", ascending=False)
    df.to_csv(os.path.join(OUTPUTS_DIR, "matcher_composition.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    for col in ["f1_eps0.55", "f1_eps0.6", "f1_strict"]:
        print(f"Spearman({col}, f1_eps0.5) = {spearmanr(df[col], df['f1_eps0.5']).statistic:.3f}")


if __name__ == "__main__":
    main()
