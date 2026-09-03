"""WP1 - Shared evaluation utilities for the BYO-ME VLM benchmark.

This module is the single source of truth for:
  - ground-truth loading (with the CORRECT image -> GT mapping, plus a helper
    reproducing the legacy buggy lookup for impact quantification)
  - prediction loading/normalization (same tolerant logic as the submitted paper)
  - gram extraction from portion strings (legacy + robust variants)
  - matching algorithms (legacy first-come-argmax, greedy one-to-one, Hungarian)

Known bugs of the submitted pipeline (compute_performance.py), reproduced here
only behind explicit *legacy* flags so their impact can be measured:
  1. GT lookup via endswith("NN.jpg"): 47/200 images matched the wrong GT
     (e.g. image 4 scored against the labels of image 104).
  2. False negatives of the last image double-counted outside the file loop.
  3. Predictions matched via argmax over all GT items; a prediction whose best
     GT was already taken counted as FP even if another GT above threshold
     was available (order-dependent, not one-to-one optimal).
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_OUTPUT_DIR = os.path.join(REPO, "model_outputs")
GT_CSV = os.path.join(REPO, "structured_food_labels.csv")
MAPPING_CSV = os.path.join(REPO, "GT_Mensa_2025", "image_participants.csv")
OUTPUTS_DIR = os.path.join(REPO, "analysis", "outputs")

N_IMAGES = 200
TEMPERATURES = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

FNAME_RE = re.compile(
    r"^(?P<img>\d{1,3})_(?P<model>.+?)_T(?P<temp>\d(?:\.\d+)?)_user01_sys01\.json$"
)


def normalize(text: str) -> str:
    return str(text).strip().lower()


# ---------------------------------------------------------------- ground truth

def load_ground_truth() -> dict[int, list[tuple[str, float]]]:
    """Correct mapping: image i <-> GT path ending in -%03d.jpg."""
    gt_df = pd.read_csv(GT_CSV)
    gt_df["food"] = gt_df["food"].astype(str).map(normalize)
    by_path: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for _, row in gt_df.iterrows():
        by_path[row["image"]].append((row["food"], float(row["portion"])))
    gt: dict[int, list[tuple[str, float]]] = {}
    for i in range(1, N_IMAGES + 1):
        suffix = f"-{i:03d}.jpg"
        matches = [p for p in by_path if p.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"image {i}: {len(matches)} GT paths match {suffix}")
        gt[i] = by_path[matches[0]]
    return gt


def load_ground_truth_legacy() -> dict[int, list[tuple[str, float]]]:
    """Reproduce the buggy legacy lookup (endswith on 2-digit id, first sorted path)."""
    gt_df = pd.read_csv(GT_CSV)
    gt_df["food"] = gt_df["food"].astype(str).map(normalize)
    by_path: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for _, row in gt_df.iterrows():
        by_path[row["image"]].append((row["food"], float(row["portion"])))
    sorted_paths = sorted(by_path)  # pandas groupby sorted keys
    gt: dict[int, list[tuple[str, float]]] = {}
    for i in range(1, N_IMAGES + 1):
        image_id = str(i).zfill(2)  # what basename.split("_")[0].zfill(2) produced
        key = next(p for p in sorted_paths if p.endswith(f"{image_id}.jpg"))
        gt[i] = by_path[key]
    return gt


def wrong_gt_images() -> list[int]:
    correct, legacy = load_ground_truth(), load_ground_truth_legacy()
    return [i for i in range(1, N_IMAGES + 1) if correct[i] != legacy[i]]


def load_annotator_map() -> dict[int, str]:
    df = pd.read_csv(MAPPING_CSV)
    out = {}
    for _, r in df.iterrows():
        out[int(r["resized_filename"].split(".")[0])] = r["participant_id"]
    return out


# ----------------------------------------------------------------- predictions

def load_and_normalize_preds(path: str) -> tuple[list[dict], list[str]]:
    """Same tolerant logic as the submitted pipeline."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        items = data.get("food_items", [])
    elif isinstance(data, list):
        items = data
    else:
        return [], ["root_not_dict_or_list"]
    if not isinstance(items, list):
        return [], ["food_items_not_list"]
    normalized, issues = [], []
    for item in items:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, str):
            issues.append("item_is_string")
            normalized.append({"name": item, "portion_estimate": ""})
        else:
            issues.append("item_invalid_type")
    return normalized, issues


def load_run_predictions(model, temp, empty_raw_as_empty=False):
    """image -> (list of (name, portion_string), n_empty_names).

    Predictions with an empty name are counted separately (automatic FPs).
    With empty_raw_as_empty=True, images whose raw model output was EMPTY are
    treated as empty predictions, discarding the repair-LLM's hallucinations."""
    out = {}
    for img, path in iter_run_files(model, temp):
        if empty_raw_as_empty:
            raw_path = path.replace(".json", "_raw.txt")
            if os.path.exists(raw_path) and os.path.getsize(raw_path) == 0:
                out[img] = ([], 0)
                continue
        preds, _ = load_and_normalize_preds(path)
        named = [
            (normalize(p.get("name", "")), str(p.get("portion_estimate", "")))
            for p in preds
            if normalize(p.get("name", ""))
        ]
        out[img] = (named, len(preds) - len(named))
    return out


def iter_run_files(model: str, temp: str):
    """Yield (image_index, path) for one (model, T) run, sorted by image."""
    for fname in sorted(os.listdir(MODEL_OUTPUT_DIR)):
        m = FNAME_RE.match(fname)
        if m and m.group("model") == model and m.group("temp") == temp:
            yield int(m.group("img")), os.path.join(MODEL_OUTPUT_DIR, fname)


def list_runs() -> dict[tuple[str, str], int]:
    """All (model, temp) -> number of output files."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for fname in os.listdir(MODEL_OUTPUT_DIR):
        m = FNAME_RE.match(fname)
        if m:
            counts[(m.group("model"), m.group("temp"))] += 1
    return dict(counts)


def complete_runs() -> list[tuple[str, str]]:
    return sorted(k for k, n in list_runs().items() if n == N_IMAGES)


# -------------------------------------------------------------- gram extraction

def extract_grams_legacy(text: str) -> float | None:
    """Exactly the submitted paper's parser."""
    text = str(text).lower().replace("grams", "").replace("gram", "").replace("g", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


_RANGE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[-–~]\s*(\d+(?:\.\d+)?)\s*$")
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")


def extract_grams_robust(text: str) -> float | None:
    """More tolerant parser: handles 'about 120 g', '100-150 g', '120g (approx)'.

    Deliberately conservative: returns the midpoint for ranges, the first
    number otherwise, None when no digits are present (e.g. 'uncertain').
    """
    t = str(text).lower().replace("grams", "").replace("gram", "").replace("g", "").strip()
    m = _RANGE_RE.match(t)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    m = _NUM_RE.search(t)
    return float(m.group(1)) if m else None


# -------------------------------------------------------------------- matching

@dataclass
class MatchResult:
    # pred index -> gt index for matched pairs
    pairs: list[tuple[int, int, float]]  # (pred_idx, gt_idx, score)
    fp_pred_idx: list[int]
    fn_gt_idx: list[int]

    @property
    def tp(self) -> int:
        return len(self.pairs)

    @property
    def fp(self) -> int:
        return len(self.fp_pred_idx)

    @property
    def fn(self) -> int:
        return len(self.fn_gt_idx)


def match_legacy(scores: np.ndarray, threshold: float) -> MatchResult:
    """Submitted paper's algorithm: iterate predictions in output order, take the
    argmax GT over ALL GT items; count FP if below threshold OR argmax GT already
    taken (even when another GT above threshold is free)."""
    n_pred, n_gt = scores.shape
    matched_gt: set[int] = set()
    pairs, fps = [], []
    for p in range(n_pred):
        if n_gt == 0:
            fps.append(p)
            continue
        best = int(np.argmax(scores[p]))
        best_score = float(scores[p, best])
        if best_score >= threshold and best not in matched_gt:
            matched_gt.add(best)
            pairs.append((p, best, best_score))
        else:
            fps.append(p)
    fns = [g for g in range(n_gt) if g not in matched_gt]
    return MatchResult(pairs, fps, fns)


def match_greedy(scores: np.ndarray, threshold: float) -> MatchResult:
    """One-to-one greedy assignment by descending similarity score."""
    n_pred, n_gt = scores.shape
    if n_pred == 0 or n_gt == 0:
        return MatchResult([], list(range(n_pred)), list(range(n_gt)))
    order = np.argsort(scores, axis=None)[::-1]
    used_p: set[int] = set()
    used_g: set[int] = set()
    pairs = []
    for flat in order:
        p, g = divmod(int(flat), n_gt)
        s = float(scores[p, g])
        if s < threshold:
            break
        if p in used_p or g in used_g:
            continue
        used_p.add(p)
        used_g.add(g)
        pairs.append((p, g, s))
    fps = [p for p in range(n_pred) if p not in used_p]
    fns = [g for g in range(n_gt) if g not in used_g]
    return MatchResult(pairs, fps, fns)


def match_hungarian(scores: np.ndarray, threshold: float) -> MatchResult:
    """Optimal one-to-one assignment maximizing total score, then threshold."""
    from scipy.optimize import linear_sum_assignment

    n_pred, n_gt = scores.shape
    if n_pred == 0 or n_gt == 0:
        return MatchResult([], list(range(n_pred)), list(range(n_gt)))
    rows, cols = linear_sum_assignment(-scores)
    pairs, used_p, used_g = [], set(), set()
    for p, g in zip(rows, cols):
        s = float(scores[p, g])
        if s >= threshold:
            pairs.append((int(p), int(g), s))
            used_p.add(int(p))
            used_g.add(int(g))
    fps = [p for p in range(n_pred) if p not in used_p]
    fns = [g for g in range(n_gt) if g not in used_g]
    return MatchResult(pairs, fps, fns)


MATCHERS = {"legacy": match_legacy, "greedy": match_greedy, "hungarian": match_hungarian}


# ------------------------------------------------------------------ pair cache

class PairScores:
    """Lookup of cross-encoder scores for (pred_name, gt_name) pairs."""

    def __init__(self, parquet_path: str):
        df = pd.read_parquet(parquet_path)
        self._scores = dict(zip(zip(df["pred_name"], df["gt_name"]), df["score"]))

    def matrix(self, pred_names: list[str], gt_names: list[str]) -> np.ndarray:
        out = np.zeros((len(pred_names), len(gt_names)), dtype=np.float32)
        for i, p in enumerate(pred_names):
            for j, g in enumerate(gt_names):
                try:
                    out[i, j] = self._scores[(p, g)]
                except KeyError:
                    raise KeyError(f"missing pair score: {(p, g)!r}")
        return out
