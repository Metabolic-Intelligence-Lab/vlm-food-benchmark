"""Dataset composition and robustness across participants.

1. Composition: items/images/unique names/median weight per food category,
   images and items per participant, recurrence of food descriptions.
2. Robustness: recognition F1 (eps=0.5, Table 2 configurations) per participant,
   and leave-one-participant-out (LOPO) rankings compared with the full ranking
   (Spearman), with the best open-source and best overall model in every split.
   Participants photographed their own meals with their own smartphones, so the
   split approximates a change of photographer, device and acquisition habits.

Usage: analysis_env/bin/python analysis/19_dataset_composition.py
Outputs: analysis/outputs/dataset_composition_{categories,participants}.csv,
         participant_robustness.csv
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluation import OUTPUTS_DIR, load_annotator_map, load_ground_truth

spec = importlib.util.spec_from_file_location(
    "m10", os.path.join(HERE, "10_dataset_and_stratified.py"))
m10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m10)

PROPRIETARY = ("gpt", "claude")


def f1(g):
    tp, fp, fn = g.tp.sum(), g.fp.sum(), g.fn.sum()
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main():
    gt = load_ground_truth()
    participant = load_annotator_map()
    items = pd.DataFrame([(i, f, w) for i, lst in gt.items() for f, w in lst],
                         columns=["image", "food", "weight"])
    items["category"] = items.food.map(m10.categorize)
    items["participant"] = items.image.map(participant)

    # ---- 1. composition
    cats = (items.groupby("category")
            .agg(n_items=("weight", "size"), n_images=("image", "nunique"),
                 n_unique_names=("food", "nunique"), median_weight=("weight", "median"))
            .sort_values("n_items", ascending=False).reset_index())
    cats.to_csv(os.path.join(OUTPUTS_DIR, "dataset_composition_categories.csv"), index=False)
    parts = (items.groupby("participant")
             .agg(n_images=("image", "nunique"), n_items=("weight", "size"),
                  median_weight=("weight", "median")).reset_index())
    parts["items_per_image"] = parts.n_items / parts.n_images
    parts.to_csv(os.path.join(OUTPUTS_DIR, "dataset_composition_participants.csv"), index=False)
    counts = items.food.value_counts()
    occ = items.food.map(counts)
    print(cats.to_string(index=False))
    print(parts.round(2).to_string(index=False))
    print(f"unique names {len(counts)}; names occurring once {int((counts == 1).sum())} "
          f"({(occ == 1).mean():.0%} of items); items whose name occurs >=5 times "
          f"{(occ >= 5).mean():.0%}, >=10 times {(occ >= 10).mean():.0%}")
    print("most frequent:", counts.head(8).to_dict())

    # ---- 2. robustness across participants (Table 2 configurations)
    acc = pd.read_csv(os.path.join(OUTPUTS_DIR, "portion_accounting.csv"))
    v1 = pd.read_parquet(os.path.join(OUTPUTS_DIR, "per_image_clean.parquet")).assign(run="r1")
    v2 = pd.read_parquet(os.path.join(OUTPUTS_DIR, "per_image_v2.parquet"))
    frames = []
    for r in acc.itertuples():
        df = v1 if r.model in set(v1.model) else v2
        d = df[(df.model == r.model) & (df.temperature == r.temperature)
               & (df.run == "r1") & (df.epsilon == 0.5)]
        assert len(d) == 200, r.model
        frames.append(d.assign(participant=d.image.map(participant)))
    per_image = pd.concat(frames)
    overall = per_image.groupby("model").apply(f1)
    rows = []
    for p in sorted(per_image.participant.unique()):
        within = per_image[per_image.participant == p].groupby("model").apply(f1)
        lopo = per_image[per_image.participant != p].groupby("model").apply(f1)
        open_lopo = lopo[[m for m in lopo.index if not m.startswith(PROPRIETARY)]]
        rows.append({
            "participant": p,
            "n_images": int(parts.set_index("participant").n_images[p]),
            "rho_within_vs_full": spearmanr(within, overall).statistic,
            "rho_lopo_vs_full": spearmanr(lopo, overall).statistic,
            "best_open_lopo": open_lopo.idxmax(), "best_overall_lopo": lopo.idxmax(),
            "best_open_within": within[open_lopo.index].idxmax(),
        })
    rob = pd.DataFrame(rows)
    rob.to_csv(os.path.join(OUTPUTS_DIR, "participant_robustness.csv"), index=False)
    print(rob.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
