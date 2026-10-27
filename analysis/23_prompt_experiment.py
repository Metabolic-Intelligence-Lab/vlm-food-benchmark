"""Concise-name prompt experiment.

Two older/newer pairs (gpt-4o / gpt-5.5, qwen2.5vl:7b / qwen3.5:9b) were re-run at
their Table 2 configuration with system_prompts/02.txt, which differs from the
original prompt only in asking for the shortest common food name (examples not
present among the ground-truth labels). Outputs: model_outputs_concise/.

For each model and prompt: words per predicted name, items per image, precision,
recall and F1 (Cross Encoder, eps=0.5, greedy matching), F1 under the LLM judge
(same rules as 15_judge_v2.py), matched-item MAE and NMAE; paired image-bootstrap
95% CI of the concise-minus-original F1 difference (CE and judge).

  --export-band  write the (pred, gt) pairs of the concise outputs falling in the
                 judge band and not yet judged to outputs/prompt_experiment_band.parquet,
                 to be judged with: 08_llm_judge.py --pairs-file <that file>

Usage: python analysis/23_prompt_experiment.py [--export-band]
Output: analysis/outputs/prompt_experiment.csv
"""
import argparse
import importlib.util
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluation import (N_IMAGES, OUTPUTS_DIR, REPO, extract_grams_robust,
                        load_and_normalize_preds, load_ground_truth,
                        load_run_predictions, match_greedy, normalize)

spec = importlib.util.spec_from_file_location("m12", os.path.join(HERE, "12_eval_all_v2.py"))
m12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m12)

CONCISE_DIR = os.path.join(REPO, "model_outputs_concise")
EPS = 0.5
JUDGE_LO, JUDGE_HI = 0.30, 0.95
SEED, B = 20260901, 2000
# model -> (campaign of the original run, temperature label)
MODELS = {"gpt-4o": ("v1", "0.2"), "gpt-5.5": ("v2", "1.0"),
          "qwen2.5vl:7b": ("v1", "0.0"), "qwen3.5:9b": ("v2", "0.0")}


def load_concise(model, temp):
    pat = re.compile(rf"^(\d{{3}})_{re.escape(model)}_T{re.escape(temp)}_user01_sys02_r1\.json$")
    out = {}
    for f in sorted(os.listdir(CONCISE_DIR)):
        m = pat.match(f)
        if m:
            preds, _ = load_and_normalize_preds(os.path.join(CONCISE_DIR, f))
            out[int(m.group(1))] = [(normalize(p.get("name", "")), str(p.get("portion_estimate", "")))
                                    for p in preds if normalize(p.get("name", ""))]
    return out


def load_original(model, campaign, temp, runs_v2):
    if campaign == "v1":
        preds = load_run_predictions(model, temp, empty_raw_as_empty=True)
        return {i: pl for i, (pl, _) in preds.items()}
    return runs_v2[(model, temp, "r1")]


def load_verdicts():
    verdicts = {}
    with open(os.path.join(OUTPUTS_DIR, "llm_judge_results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            verdicts[(r["pred_name"], r["gt_name"])] = r["verdict"] == "same"
    return verdicts


def per_image(by_img, gt, scores, verdicts):
    """Per-image arrays: CE tp/fp/fn, judge tp/fp/fn, portion sums, name stats."""
    keys = ["tp", "fp", "fn", "jtp", "jfp", "jfn", "abs_err", "n_w", "n_gt", "n_pred", "words"]
    arr = {k: np.zeros(N_IMAGES) for k in keys}
    for idx, img in enumerate(range(1, N_IMAGES + 1)):
        pl, gl = by_img.get(img, []), gt[img]
        mat = np.array([[scores[(p, g)] for g, _ in gl] for p, _ in pl],
                       dtype=np.float32).reshape(len(pl), len(gl))
        res = match_greedy(mat, EPS)
        arr["tp"][idx], arr["fp"][idx], arr["fn"][idx] = res.tp, res.fp, res.fn
        for pi, gi, _ in res.pairs:
            w = extract_grams_robust(pl[pi][1])
            if w is not None:
                arr["abs_err"][idx] += abs(w - gl[gi][1])
                arr["n_w"][idx] += 1
        # judge: same greedy rule as 15_judge_v2.py
        cands = []
        for i, (p, _) in enumerate(pl):
            for j, (g, _) in enumerate(gl):
                s = scores[(p, g)]
                same = s >= JUDGE_HI or (s >= JUDGE_LO and verdicts.get((p, g), False))
                if same:
                    cands.append((s, i, j))
        used_p, used_g = set(), set()
        for s, i, j in sorted(cands, reverse=True):
            if i not in used_p and j not in used_g:
                used_p.add(i)
                used_g.add(j)
        arr["jtp"][idx], arr["jfp"][idx], arr["jfn"][idx] = len(used_p), len(pl) - len(used_p), len(gl) - len(used_g)
        arr["n_gt"][idx], arr["n_pred"][idx] = len(gl), len(pl)
        arr["words"][idx] = sum(len(p.split()) for p, _ in pl)
    return arr


def f1(tp, fp, fn):
    return np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-band", action="store_true")
    args = parser.parse_args()

    gt = load_ground_truth()
    runs_v2 = m12.load_all_runs()
    data = {}
    for model, (campaign, temp) in MODELS.items():
        conc = load_concise(model, temp)
        if len(conc) < N_IMAGES:
            print(f"WARNING: {model} concise run incomplete ({len(conc)}/200)")
        data[model] = {"original": load_original(model, campaign, temp, runs_v2),
                       "concise": conc}
    pseudo = {(m, p, "x"): d[p] for m, d in data.items() for p in d}
    scores = m12.ensure_scores(pseudo, gt, m12.score_map())
    verdicts = load_verdicts()

    if args.export_band:
        band = {(p, g) for by in pseudo.values() for i, pl in by.items() for p, _ in pl
                for g, _ in gt[i] if JUDGE_LO <= scores[(p, g)] < JUDGE_HI and (p, g) not in verdicts}
        df = pd.DataFrame(sorted(band), columns=["pred_name", "gt_name"])
        df["score"] = [scores[k] for k in sorted(band)]
        path = os.path.join(OUTPUTS_DIR, "prompt_experiment_band.parquet")
        df.to_parquet(path, index=False)
        print(f"{len(df)} pairs to judge -> {path}")
        return

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, N_IMAGES, size=(B, N_IMAGES))
    rows = []
    for model, d in data.items():
        a = {p: per_image(d[p], gt, scores, verdicts) for p in d}
        for p, arr in a.items():
            tp, fp, fn = arr["tp"].sum(), arr["fp"].sum(), arr["fn"].sum()
            n_tot, P = arr["n_gt"].sum(), arr["n_w"].sum()
            rows.append({
                "model": model, "prompt": p,
                "words_per_name": arr["words"].sum() / arr["n_pred"].sum(),
                "items_per_image": arr["n_pred"].mean(),
                "precision": tp / (tp + fp), "recall": tp / (tp + fn), "f1": float(f1(tp, fp, fn)),
                "judge_precision": arr["jtp"].sum() / (arr["jtp"].sum() + arr["jfp"].sum()),
                "judge_recall": arr["jtp"].sum() / (arr["jtp"].sum() + arr["jfn"].sum()),
                "judge_f1": float(f1(arr["jtp"].sum(), arr["jfp"].sum(), arr["jfn"].sum())),
                "mae": arr["abs_err"].sum() / P if P else np.nan,
                "nmae": (arr["abs_err"].sum() + 129.0 * (n_tot - P)) / n_tot,
            })
        o, c = a["original"], a["concise"]
        for key, (t, fp_, fn_) in {"f1": ("tp", "fp", "fn"), "judge_f1": ("jtp", "jfp", "jfn")}.items():
            boot = (f1(*(c[k][idx].sum(axis=1) for k in (t, fp_, fn_)))
                    - f1(*(o[k][idx].sum(axis=1) for k in (t, fp_, fn_))))
            rows[-1][f"delta_{key}"] = rows[-1][key] - rows[-2][key]
            rows[-1][f"delta_{key}_lo"], rows[-1][f"delta_{key}_hi"] = np.percentile(boot, [2.5, 97.5])
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUTS_DIR, "prompt_experiment.csv"), index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
