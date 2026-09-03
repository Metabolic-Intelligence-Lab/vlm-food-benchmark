"""B3/B4 - Image-level bootstrap CIs for the 2026-campaign runs (per_image_v2).

Same resampling scheme as 04_bootstrap (200 images, B=2000, fixed seed, shared
indices for paired comparisons). Covers the 6 new open models (r1) and the
replicate runs. Outputs: bootstrap_ci_v2.csv + paired vs qwen2.5vl:7b (clean).

Usage: analysis_env/bin/python analysis/13_bootstrap_v2.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import N_IMAGES, OUTPUTS_DIR

SEED = 20260901
B = 2000


def arrays_for(df):
    combos = {}
    for key, g in df.groupby(["model", "temperature", "run", "epsilon"]):
        g = g.set_index("image").reindex(range(1, N_IMAGES + 1))
        if g["tp"].isna().any():
            continue
        combos[key] = {c: g[c].to_numpy(dtype=float) for c in
                       ["tp", "fp", "fn", "sum_abs_err", "n_matched_with_weight",
                        "n_gt", "sum_gt_weight_total", "sum_gt_weight_matched"]}
    return combos


def boot(arr, idx):
    s = {k: v[idx].sum(axis=1) for k, v in arr.items()}
    tp, fp, fn = s["tp"], s["fp"], s["fn"]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
        mae = np.where(s["n_matched_with_weight"] > 0,
                       s["sum_abs_err"] / s["n_matched_with_weight"], np.nan)
        cpmae = (s["sum_abs_err"] + s["sum_gt_weight_total"] - s["sum_gt_weight_matched"]) / s["n_gt"]
    return f1, mae, cpmae


def main():
    v2 = pd.read_parquet(os.path.join(OUTPUTS_DIR, "per_image_v2.parquet"))
    combos = arrays_for(v2)
    print(f"{len(combos)} combinazioni v2")
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, N_IMAGES, size=(B, N_IMAGES))

    boot_f1 = {}
    rows = []
    for (model, temp, run, eps), arr in sorted(combos.items()):
        f1, mae, cpmae = boot(arr, idx)
        boot_f1[(model, temp, run, eps)] = f1
        rows.append({
            "model": model, "temperature": temp, "run": run, "epsilon": eps,
            "f1_mean": float(np.mean(f1)),
            "f1_lo": float(np.percentile(f1, 2.5)),
            "f1_hi": float(np.percentile(f1, 97.5)),
            "mae_mean": float(np.nanmean(mae)),
            "mae_lo": float(np.nanpercentile(mae, 2.5)),
            "mae_hi": float(np.nanpercentile(mae, 97.5)),
            "cpmae_mean": float(np.nanmean(cpmae)),
            "cpmae_lo": float(np.nanpercentile(cpmae, 2.5)),
            "cpmae_hi": float(np.nanpercentile(cpmae, 97.5)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUTS_DIR, "bootstrap_ci_v2.csv"), index=False)

    # paired: nuovi modelli (best-T, eps=0.5) vs qwen2.5vl:7b della campagna clean
    old = pd.read_parquet(os.path.join(OUTPUTS_DIR, "per_image_clean.parquet"))
    ref = old[(old.model == "qwen2.5vl:7b") & (old.temperature == 0.0)
              & (old.epsilon == 0.5)]
    ref_arr = arrays_for(ref.assign(run="r1"))[("qwen2.5vl:7b", 0.0, "r1", 0.5)] \
        if len(ref) else None
    # per_image_clean lacks 'run': assign; also lacks fp-weight cols handled above
    print("\n=== CI 95% nuovi modelli (best-T, eps=0.5) ===")
    new_models = ["gemma4:12b", "gemma4:e4b", "minicpm-v4.5:8b", "minicpm-v4.6",
                  "qwen3.5:4b", "qwen3.5:9b"]
    sel = df[(df.epsilon == 0.5) & (df.run == "r1") & df.model.isin(new_models)]
    best = sel.loc[sel.groupby("model")["f1_mean"].idxmax()]
    if ref_arr is not None:
        f1_ref, _, _ = boot(ref_arr, idx)
    for r in best.sort_values("f1_mean", ascending=False).itertuples():
        line = (f"{r.model:18s} F1 {r.f1_mean:.3f} [{r.f1_lo:.3f}-{r.f1_hi:.3f}] "
                f"MAE {r.mae_mean:.1f} [{r.mae_lo:.1f}-{r.mae_hi:.1f}] "
                f"cpMAE {r.cpmae_mean:.0f} [{r.cpmae_lo:.0f}-{r.cpmae_hi:.0f}]")
        if ref_arr is not None:
            delta = f1_ref - boot_f1[(r.model, r.temperature, "r1", 0.5)]
            line += f" | Δ vs qwen2.5vl:7b: {np.mean(delta):+.3f} [P(qwen>) {np.mean(delta>0):.2f}]"
        print(line)


if __name__ == "__main__":
    main()
