"""WP1 - Score every unique (pred_name, gt_name) pair once with the CrossEncoder.

Collects prediction names from ALL model outputs and pairs each with the GT
names of its image under BOTH the correct mapping and the legacy (buggy)
mapping, so that downstream scripts can reproduce the submitted paper's numbers
and compute the corrected ones from the same cache.

Usage: analysis_env/bin/python analysis/01_score_pairs.py [--count-only]
Output: analysis/outputs/pair_scores.parquet  (pred_name, gt_name, score)
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import (OUTPUTS_DIR, iter_run_files, list_runs,
                        load_and_normalize_preds, load_ground_truth,
                        load_ground_truth_legacy, normalize)

CROSS_ENCODER = "cross-encoder/stsb-distilroberta-base"
OUT_PATH = os.path.join(OUTPUTS_DIR, "pair_scores.parquet")


def collect_pairs():
    gt_correct = load_ground_truth()
    gt_legacy = load_ground_truth_legacy()
    gt_names = {
        i: sorted({name for name, _ in gt_correct[i]} | {name for name, _ in gt_legacy[i]})
        for i in gt_correct
    }
    pairs = set()
    n_files = n_bad = 0
    for (model, temp) in sorted(list_runs()):
        for img, path in iter_run_files(model, temp):
            n_files += 1
            try:
                preds, _ = load_and_normalize_preds(path)
            except Exception:
                n_bad += 1
                continue
            for pred in preds:
                pname = normalize(pred.get("name", ""))
                if not pname:
                    continue
                for gname in gt_names[img]:
                    pairs.add((pname, gname))
    print(f"scanned {n_files} files ({n_bad} unreadable), {len(pairs)} unique pairs")
    return sorted(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cross-encoder", default=CROSS_ENCODER)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    pairs = collect_pairs()
    if args.count_only:
        return

    import torch
    from sentence_transformers.cross_encoder import CrossEncoder

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"scoring on device: {device} with {args.cross_encoder}")
    model = CrossEncoder(args.cross_encoder, device=device)

    t0 = time.time()
    scores = model.predict(
        [list(p) for p in pairs],
        batch_size=args.batch_size,
        show_progress_bar=True,
    )
    print(f"scored {len(pairs)} pairs in {time.time() - t0:.0f}s")

    df = pd.DataFrame(pairs, columns=["pred_name", "gt_name"])
    df["score"] = np.asarray(scores, dtype=np.float32)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
