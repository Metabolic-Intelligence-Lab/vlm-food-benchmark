"""Direction and structure of the portion-estimation bias.

For a representative set of models (best configuration of Table 2, eps=0.5),
characterizes the signed error pred - GT on matched items with a parsed weight:
  - median pred/GT ratio and share of under-estimates
  - median signed error by GT-weight quartile (pooled quartile edges)
  - share of estimates that are multiples of 50 g (output quantization)
  - per food category (pooled over the models)
  - repaired / truncated raw outputs of the same configurations (rules out
    output-length constraints as a cause)

Usage: python analysis/16_portion_bias.py
Outputs: analysis/outputs/portion_bias_models.csv, portion_bias_categories.csv
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluation import OUTPUTS_DIR, REPO

spec = importlib.util.spec_from_file_location(
    "m10", os.path.join(HERE, "10_dataset_and_stratified.py"))
m10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m10)

EPS = 0.5
# (model, T, run, campaign) - best-F1 configuration of Table 2
CONFIGS = [
    ("qwen2.5vl:7b", 0.0, "r1", "v1"), ("qwen3.5:9b", 0.0, "r1", "v2"),
    ("minicpm-v4.5:8b", 0.0, "r1", "v2"), ("gpt-4o", 0.2, "r1", "v1"),
    ("gpt-4.1", 0.4, "r1", "v1"), ("gpt-5.2", 0.0, "r1", "v1"),
    ("gpt-5.5", 1.0, "r1", "v2"), ("claude-sonnet-5", 1.0, "r1", "v2"),
]


def raw_stats(model, temp, run, campaign):
    """(n repaired raw outputs, n with unbalanced braces = likely truncated)."""
    d = os.path.join(REPO, "model_outputs" if campaign == "v1" else "model_outputs_v2")
    suffix = (f"_{model}_T{temp}_user01_sys01_raw.txt" if campaign == "v1"
              else f"_{model}_T{temp}_user01_sys01_{run}_raw.txt")
    n = trunc = 0
    for f in os.listdir(d):
        if f.endswith(suffix):
            n += 1
            s = open(os.path.join(d, f), errors="replace").read().strip()
            trunc += bool(s) and s.count("{") > s.count("}")
    return n, trunc


def main():
    v1 = pd.read_parquet(os.path.join(OUTPUTS_DIR, "per_item_clean.parquet"))
    v1["run"] = "r1"
    v2 = pd.read_parquet(os.path.join(OUTPUTS_DIR, "per_item_v2.parquet"))
    items = []
    for model, temp, run, campaign in CONFIGS:
        df = v1 if campaign == "v1" else v2
        d = df[(df.model == model) & (df.temperature == temp) & (df.run == run)
               & (df.epsilon == EPS) & df.pred_weight.notna()].copy()
        d["err"] = d.pred_weight - d.gt_weight
        d["ratio"] = d.pred_weight / d.gt_weight
        d["category"] = d.gt_food.map(m10.categorize)
        items.append(d)
    pooled = pd.concat(items)
    edges = np.percentile(pooled.gt_weight, [25, 50, 75])
    bins = [-np.inf, *edges, np.inf]
    labels = ["Q1", "Q2", "Q3", "Q4"]

    rows = []
    for (model, temp, run, campaign), d in zip(CONFIGS, items):
        q = pd.cut(d.gt_weight, bins, labels=labels)
        n_raw, n_trunc = raw_stats(model, f"{temp:.1f}", run, campaign)
        rows.append({
            "model": model, "temperature": temp, "n_items": len(d),
            "median_ratio": d.ratio.median(),
            "frac_under": (d.err < 0).mean(),
            **{f"median_err_{k}": g.err.median()
               for k, g in d.groupby(q, observed=True)},
            "frac_multiple_of_50": (d.pred_weight % 50 == 0).mean(),
            "n_repaired_outputs": n_raw, "n_truncated_outputs": n_trunc,
        })
    models_df = pd.DataFrame(rows)
    models_df.to_csv(os.path.join(OUTPUTS_DIR, "portion_bias_models.csv"), index=False)

    cat_df = (pooled.groupby("category")
              .agg(n=("err", "size"), median_ratio=("ratio", "median"),
                   frac_under=("err", lambda e: (e < 0).mean()),
                   median_gt_weight=("gt_weight", "median"))
              .sort_values("median_ratio").reset_index())
    cat_df.to_csv(os.path.join(OUTPUTS_DIR, "portion_bias_categories.csv"), index=False)

    print(f"GT-weight quartile edges (pooled matched items): {edges}")
    print(models_df.round(2).to_string(index=False))
    print("\nby category (pooled):")
    print(cat_df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
