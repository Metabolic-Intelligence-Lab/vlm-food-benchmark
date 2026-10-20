"""Do portion estimates carry visual information beyond food identity?

For every model with at least 100 numerical estimates (Table 2 configurations,
eps=0.5), on the SAME matched items:
  - MAE and median absolute percentage error (MedAPE) of the model and of the
    typical-serving reference (per-food median weight, leave-one-image-out,
    global median for foods seen in a single image)
  - LOIO-CV linear combination GT ~ typical-serving + model estimate, and its
    MAE gain over the typical-serving reference (paired image bootstrap CI)
  - within-food correlation: for foods with >= 5 matched instances, Pearson
    correlation between food-centered predicted and true weights, i.e. whether
    the model tracks how large THIS portion is relative to other portions of the
    same food (image bootstrap CI). This is the cleanest test of visual size
    sensitivity, since food identity is held fixed.

Usage: analysis_env/bin/python analysis/20_visual_vs_identity.py
Output: analysis/outputs/visual_vs_identity.csv
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluation import OUTPUTS_DIR, load_ground_truth

EPS = 0.5
MIN_ESTIMATES = 100
MIN_FOOD_INSTANCES = 5
SEED, B = 20260901, 1000


def typical_serving_lookup(gt):
    """(image, food) -> per-food median weight over the OTHER images."""
    items = [(i, f, w) for i, lst in gt.items() for f, w in lst]
    global_median = float(np.median([w for *_, w in items]))
    by_food = defaultdict(list)
    for i, f, w in items:
        by_food[f].append((i, w))

    def ts(img, food):
        others = [w for i, w in by_food[food] if i != img]
        return float(np.median(others)) if others else global_median
    return ts


def loio_combined(y, ts, pred, img):
    """Leave-one-image-out OLS prediction of y from [1, ts, pred]."""
    out = np.empty(len(y))
    for i in np.unique(img):
        te = img == i
        X = np.c_[np.ones((~te).sum()), ts[~te], pred[~te]]
        b = np.linalg.lstsq(X, y[~te], rcond=None)[0]
        out[te] = b[0] + b[1] * ts[te] + b[2] * pred[te]
    return out


def within_food_r(d):
    d = d[d.gt_food.map(d.gt_food.value_counts()) >= MIN_FOOD_INSTANCES]
    if len(d) < 10:
        return np.nan, 0
    gc = d.gt_weight - d.groupby("gt_food").gt_weight.transform("mean")
    pc = d.pred_weight - d.groupby("gt_food").pred_weight.transform("mean")
    if pc.std() == 0:
        return np.nan, len(d)
    return float(np.corrcoef(gc, pc)[0, 1]), len(d)


def main():
    gt = load_ground_truth()
    ts = typical_serving_lookup(gt)
    v1 = pd.read_parquet(os.path.join(OUTPUTS_DIR, "per_item_clean.parquet")).assign(run="r1")
    v2 = pd.read_parquet(os.path.join(OUTPUTS_DIR, "per_item_v2.parquet"))
    acc = pd.read_csv(os.path.join(OUTPUTS_DIR, "portion_accounting.csv"))
    rng = np.random.default_rng(SEED)

    rows = []
    for r in acc.itertuples():
        if r.TP_with_estimate < MIN_ESTIMATES:
            continue
        df = v1 if r.model in set(v1.model) else v2
        d = df[(df.model == r.model) & (df.temperature == r.temperature) & (df.run == "r1")
               & (df.epsilon == EPS) & df.pred_weight.notna()].copy()
        d["ts"] = [ts(i, f) for i, f in zip(d.image, d.gt_food)]
        y, p, s, img = (d.gt_weight.to_numpy(float), d.pred_weight.to_numpy(float),
                        d.ts.to_numpy(float), d.image.to_numpy())
        comb = loio_combined(y, s, p, img)
        e_model, e_ts, e_comb = np.abs(y - p), np.abs(y - s), np.abs(y - comb)
        r_within, n_within = within_food_r(d)

        # paired image bootstrap
        groups = {i: np.flatnonzero(img == i) for i in np.unique(img)}
        keys = np.array(list(groups))
        gains, rws = [], []
        for _ in range(B):
            sel = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
            gains.append(e_ts[sel].mean() - e_comb[sel].mean())
            rws.append(within_food_r(d.iloc[sel])[0])
        rows.append({
            "model": r.model, "n_items": len(d),
            "MAE_model": e_model.mean(), "MAE_typical_same_items": e_ts.mean(),
            "MedAPE_model": float(np.median(e_model / y)) * 100,
            "MedAPE_typical_same_items": float(np.median(e_ts / y)) * 100,
            "MAE_combined_cv": e_comb.mean(),
            "gain_combined_vs_typical": e_ts.mean() - e_comb.mean(),
            "gain_lo": np.percentile(gains, 2.5), "gain_hi": np.percentile(gains, 97.5),
            "r_within_food": r_within, "r_within_lo": np.nanpercentile(rws, 2.5),
            "r_within_hi": np.nanpercentile(rws, 97.5), "n_items_within_food": n_within,
        })
    out = pd.DataFrame(rows).sort_values("MAE_model")
    out.to_csv(os.path.join(OUTPUTS_DIR, "visual_vs_identity.csv"), index=False)
    pd.set_option("display.width", 250)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
