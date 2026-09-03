"""WP7 - Dataset characterization + stratified error analyses.

1. Macro-category mapping of the 314 unique GT food names (keyword rules,
   exported to food_categories.csv for manual review).
2. Per-category portion-error analysis (R1.3: the appearance->weight relation
   differs across food types) + per-category GT weight variability.
3. Per-annotator analysis (R1.1: 6 people = 6 different phones/conditions):
   F1 and portion error per annotator, for the top models.
4. Dataset statistics for the manuscript (R2.1: instances vs unique names).

Usage: analysis_env/bin/python analysis/10_dataset_and_stratified.py [--mode clean]
Outputs: food_categories.csv, category_errors.csv, annotator_errors.csv,
         dataset_stats.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import OUTPUTS_DIR, load_annotator_map, load_ground_truth

# ordered keyword rules: first match wins
CATEGORY_RULES = [
    ("beverage", ["water", "juice", "tea", "coffee", "coke", "soda"]),
    ("soup", ["soup", "minestrone", "broth", "velvety", "cream of", "passato"]),
    ("bread", ["bread", "roll", "focaccia", "crackers", "breadstick", "grissini"]),
    ("pasta_rice_grains", ["pasta", "spaghetti", "penne", "fusilli", "rice",
                           "risotto", "gnocchi", "lasagna", "couscous", "barley",
                           "spelt", "cereal", "tortellini", "ravioli", "noodle"]),
    ("dessert_yogurt", ["yogurt", "cake", "dessert", "pudding", "ice cream",
                        "cookie", "biscuit", "tart", "chocolate", "tiramisu"]),
    ("fruit", ["apple", "banana", "orange", "pear", "peach", "plum", "kiwi",
               "melon", "watermelon", "apricot", "grapes", "cherries", "fruit",
               "strawberr", "pineapple", "nectarine"]),
    ("cheese_dairy", ["cheese", "mozzarella", "ricotta", "parmesan", "burrata",
                      "stracchino", "scamorza", "milk"]),
    ("fish", ["fish", "tuna", "salmon", "cod", "seafood", "swordfish", "octopus",
              "squid", "anchov", "sea bream", "sea bass", "shrimp"]),
    ("meat_protein", ["chicken", "beef", "pork", "turkey", "meat", "ham",
                      "sausage", "meatball", "cutlet", "egg", "omelet",
                      "frittata", "burger", "bresaola", "speck", "salami",
                      "roast", "veal", "rabbit"]),
    ("legumes", ["beans", "chickpea", "lentil", "peas", "legume", "hummus",
                 "edamame", "soy"]),
    ("potatoes", ["potato", "fries"]),
    ("vegetables", ["salad", "vegetable", "zucchini", "courgette", "eggplant",
                    "aubergine", "spinach", "broccoli", "cauliflower", "carrot",
                    "tomato", "pepper", "chicory", "chard", "fennel", "greens",
                    "artichoke", "pumpkin", "cabbage", "lettuce", "rocket",
                    "arugula", "mushroom", "onion", "beet", "green bean",
                    "escarole", "endive", "radicchio", "corn", "cucumber"]),
]


def categorize(name):
    for cat, kws in CATEGORY_RULES:
        if any(k in name for k in kws):
            return cat
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="clean")
    parser.add_argument("--epsilon", type=float, default=0.6)
    args = parser.parse_args()

    gt = load_ground_truth()
    annotator = load_annotator_map()
    items = pd.DataFrame(
        [(img, food, w) for img, lst in gt.items() for food, w in lst],
        columns=["image", "food", "gt_weight"])
    items["category"] = items.food.map(categorize)
    items["annotator"] = items.image.map(annotator)

    # 1. category mapping for manual review
    mapping = (items.groupby(["category", "food"]).size().rename("n")
               .reset_index().sort_values(["category", "n"], ascending=[True, False]))
    mapping.to_csv(os.path.join(OUTPUTS_DIR, "food_categories.csv"), index=False)
    n_other = (items.category == "other").sum()
    print(f"1. categories assigned; 'other': {n_other}/{len(items)} instances "
          f"({items[items.category == 'other'].food.nunique()} unique names)")

    # 2. per-category errors for top models + GT variability
    per_item = pd.read_parquet(
        os.path.join(OUTPUTS_DIR, f"per_item_{args.mode}.parquet"))
    per_item = per_item[(per_item.epsilon == args.epsilon)
                        & per_item.pred_weight.notna()].copy()
    per_item["category"] = per_item.gt_food.map(categorize)
    per_item["abs_err"] = (per_item.gt_weight - per_item.pred_weight).abs()
    per_item["rel_err"] = per_item.abs_err / per_item.gt_weight

    top_models = ["qwen2.5vl:7b", "gpt-4o", "gpt-4.1", "llama3.2-vision:11b"]
    cat_rows = []
    gt_var = items.groupby("category").gt_weight.agg(["count", "median", "mean", "std"])
    gt_var["cv"] = gt_var["std"] / gt_var["mean"]
    for model in top_models:
        sub = per_item[per_item.model == model]
        for cat, g in sub.groupby("category"):
            cat_rows.append({
                "model": model, "category": cat, "n_matched": len(g),
                "medae": float(g.abs_err.median()),
                "mae": float(g.abs_err.mean()),
                "median_rel_err": float(g.rel_err.median()),
            })
    cat_df = pd.DataFrame(cat_rows)
    cat_df = cat_df.merge(gt_var["cv"].rename("gt_weight_cv").reset_index(),
                          on="category")
    cat_df.to_csv(os.path.join(OUTPUTS_DIR, "category_errors.csv"), index=False)
    print("\n2. median relative portion error by category (pooled top models):")
    pooled = (per_item[per_item.model.isin(top_models)]
              .groupby("category")
              .agg(n=("abs_err", "size"), medae=("abs_err", "median"),
                   med_rel_err=("rel_err", "median")))
    print(pooled.sort_values("med_rel_err", ascending=False).round(2).to_string())

    # 3. per-annotator analysis
    per_image = pd.read_parquet(
        os.path.join(OUTPUTS_DIR, f"per_image_{args.mode}.parquet"))
    per_image = per_image[per_image.epsilon == args.epsilon]
    ann_rows = []
    for model in top_models:
        sub = per_image[per_image.model == model]
        # pool over temperatures for stability
        for ann, g in sub.groupby("annotator"):
            tp, fp, fn = g.tp.sum(), g.fp.sum(), g.fn.sum()
            p = tp / (tp + fp) if tp + fp else 0
            r = tp / (tp + fn) if tp + fn else 0
            ann_rows.append({
                "model": model, "annotator": ann,
                "n_images": g.image.nunique(),
                "f1": 2 * p * r / (p + r) if p + r else 0,
                "cpmae": float((g.sum_abs_err.sum()
                                + (g.sum_gt_weight_total - g.sum_gt_weight_matched).sum())
                               / g.n_gt.sum()),
            })
    ann_df = pd.DataFrame(ann_rows)
    ann_df.to_csv(os.path.join(OUTPUTS_DIR, "annotator_errors.csv"), index=False)
    print(f"\n3. F1 by annotator (eps={args.epsilon}, pooled over T):")
    print(ann_df.pivot(index="annotator", columns="model", values="f1")
          .round(3).to_string())

    # 4. dataset stats
    stats = {
        "n_images": int(items.image.nunique()),
        "n_instances": len(items),
        "n_unique_food_names": int(items.food.nunique()),
        "items_per_image": items.groupby("image").size().describe().round(2).to_dict(),
        "instances_per_category": items.category.value_counts().to_dict(),
        "gt_weight_by_category": gt_var.round(1).to_dict("index"),
        "images_per_annotator": items.groupby("annotator").image.nunique().to_dict(),
    }
    with open(os.path.join(OUTPUTS_DIR, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("\n4. dataset_stats.json saved")


if __name__ == "__main__":
    main()
