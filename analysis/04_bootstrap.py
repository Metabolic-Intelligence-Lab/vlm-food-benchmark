"""WP3 - Image-level bootstrap CIs, paired comparisons, ranking stability.

Resamples the 200 images with replacement (B=2000, fixed seed). The SAME
resample indices are used for every (model, T, eps), so differences between
models are paired by construction (answers robustness analysis, point 3: clustering of
items within an image, CIs, paired statistical comparisons, rank uncertainty).

Outputs (analysis/outputs/):
  bootstrap_ci_<mode>.csv        95% CI for f1/precision/recall/jaccard/mae/cpmae
  paired_comparisons_<mode>.csv  paired deltas vs reference models
  rank_probabilities_<mode>.csv  P(model has best F1 / best cpmae among open models)

Usage: analysis_env/bin/python analysis/04_bootstrap.py [--mode greedy] [--B 2000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import N_IMAGES, OUTPUTS_DIR

OPENAI_MODELS = {"gpt-4o", "gpt-4.1", "gpt-5.2", "gpt-4o-mini"}
SEED = 20260901


def per_image_arrays(per_image):
    """dict (model, T, eps) -> dict of np arrays indexed by image 1..200."""
    combos = {}
    for key, g in per_image.groupby(["model", "temperature", "epsilon"]):
        g = g.set_index("image").reindex(range(1, N_IMAGES + 1))
        if g["tp"].isna().any():
            continue  # incomplete run, skip
        combos[key] = {
            c: g[c].to_numpy(dtype=float)
            for c in ["tp", "fp", "fn", "sum_abs_err", "n_matched_with_weight",
                      "n_gt", "sum_gt_weight_total", "sum_gt_weight_matched",
                      "sum_fp_pred_weight"]
        }
    return combos


def boot_metrics(arr, idx):
    """Vectorized metrics for all bootstrap replicates. idx: (B, 200) ints 0-199."""
    s = {k: v[idx].sum(axis=1) for k, v in arr.items()}
    tp, fp, fn = s["tp"], s["fp"], s["fn"]
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), 0.0)
        jaccard = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)
        mae = np.where(s["n_matched_with_weight"] > 0,
                       s["sum_abs_err"] / s["n_matched_with_weight"], np.nan)
        missed_w = s["sum_gt_weight_total"] - s["sum_gt_weight_matched"]
        cpmae = np.where(s["n_gt"] > 0,
                         (s["sum_abs_err"] + missed_w) / s["n_gt"], np.nan)
        cpmae_fp = np.where(s["n_gt"] > 0,
                            (s["sum_abs_err"] + missed_w + s["sum_fp_pred_weight"])
                            / s["n_gt"], np.nan)
    return {"precision": precision, "recall": recall, "f1": f1,
            "jaccard": jaccard, "mae": mae, "cpmae": cpmae, "cpmae_fp": cpmae_fp}


def ci(x):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    return (float(np.mean(x)), float(np.percentile(x, 2.5)),
            float(np.percentile(x, 97.5)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="greedy")
    parser.add_argument("--B", type=int, default=2000)
    args = parser.parse_args()

    per_image = pd.read_parquet(
        os.path.join(OUTPUTS_DIR, f"per_image_{args.mode}.parquet"))
    combos = per_image_arrays(per_image)
    print(f"{len(combos)} (model, T, eps) combos")

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, N_IMAGES, size=(args.B, N_IMAGES))

    # ---- CIs per combo, keep bootstrap F1/cpmae for pairing/ranking ----
    boot_f1, boot_cpmae = {}, {}
    ci_rows = []
    for (model, temp, eps), arr in sorted(combos.items()):
        m = boot_metrics(arr, idx)
        boot_f1[(model, temp, eps)] = m["f1"]
        boot_cpmae[(model, temp, eps)] = m["cpmae"]
        row = {"model": model, "temperature": temp, "epsilon": eps}
        for name, vals in m.items():
            mean, lo, hi = ci(vals)
            row[f"{name}_mean"] = mean
            row[f"{name}_lo"] = lo
            row[f"{name}_hi"] = hi
        ci_rows.append(row)
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(os.path.join(OUTPUTS_DIR, f"bootstrap_ci_{args.mode}.csv"),
                 index=False)

    # ---- best-T selection per (model, eps) by point-estimate F1 ----
    point = ci_df.loc[ci_df.groupby(["model", "epsilon"])["f1_mean"].idxmax(),
                      ["model", "epsilon", "temperature"]]
    best_T = {(r.model, r.epsilon): r.temperature for r in point.itertuples()}

    # ---- paired comparisons vs reference models ----
    pair_rows = []
    for eps in sorted({k[2] for k in combos}):
        models = sorted({k[0] for k in combos if k[2] == eps})
        refs = [m for m in ("qwen2.5vl:7b", "gpt-4o") if m in models]
        for ref in refs:
            fr = boot_f1[(ref, best_T[(ref, eps)], eps)]
            for m in models:
                if m == ref:
                    continue
                fm = boot_f1[(m, best_T[(m, eps)], eps)]
                delta = fr - fm
                pair_rows.append({
                    "epsilon": eps, "reference": ref, "model": m,
                    "ref_T": best_T[(ref, eps)], "model_T": best_T[(m, eps)],
                    "delta_f1_mean": float(np.mean(delta)),
                    "delta_f1_lo": float(np.percentile(delta, 2.5)),
                    "delta_f1_hi": float(np.percentile(delta, 97.5)),
                    "p_ref_better": float(np.mean(delta > 0)),
                })
    pd.DataFrame(pair_rows).to_csv(
        os.path.join(OUTPUTS_DIR, f"paired_comparisons_{args.mode}.csv"),
        index=False)

    # ---- ranking stability among open-source models ----
    rank_rows = []
    for eps in sorted({k[2] for k in combos}):
        open_models = sorted({k[0] for k in combos
                              if k[2] == eps and k[0] not in OPENAI_MODELS})
        F = np.vstack([boot_f1[(m, best_T[(m, eps)], eps)] for m in open_models])
        C = np.vstack([boot_cpmae[(m, best_T[(m, eps)], eps)] for m in open_models])
        best_f1_idx = np.argmax(F, axis=0)
        best_cp_idx = np.nanargmin(np.where(np.isnan(C), np.inf, C), axis=0)
        for i, m in enumerate(open_models):
            rank_rows.append({
                "epsilon": eps, "model": m,
                "p_best_f1_open": float(np.mean(best_f1_idx == i)),
                "p_best_cpmae_open": float(np.mean(best_cp_idx == i)),
            })
    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(os.path.join(OUTPUTS_DIR, f"rank_probabilities_{args.mode}.csv"),
                   index=False)

    print("\n=== P(best open-source model), by epsilon ===")
    for eps, g in rank_df.groupby("epsilon"):
        top = g.sort_values("p_best_f1_open", ascending=False).head(3)
        line = ", ".join(f"{r.model}: {r.p_best_f1_open:.2f}"
                         for r in top.itertuples())
        print(f"  eps={eps}: {line}")


if __name__ == "__main__":
    main()
