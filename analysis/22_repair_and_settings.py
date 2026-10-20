"""Output repair and inference settings as confounders.

1. JSON failure rates per model, over all official runs (first campaign: all
   temperatures; 2026 campaign: run r1 and replicates). An output is "empty" when
   the raw response was empty, "repaired" when a non-empty raw response could not
   be parsed and was converted by the repair LLM, "valid" otherwise. Raw responses
   were logged only from August 2025, i.e. not for images 1-84 of the earliest
   first-campaign runs: rates are therefore computed on images 85-200 for the
   first campaign (logging complete for every model) and on all images for the
   2026 campaign.
2. Repair sensitivity on images 85-200 (Table 2 configurations, eps=0.5): F1
   as published, with every repaired output treated as an empty prediction, and
   with the repair LLM replaced by the deterministic parser of 06_repair_validation.
3. Temperature selection: best-T vs mean-over-T ranking, best open model at every
   temperature, proprietary models at the OpenAI default temperature (T=1).

Usage: analysis_env/bin/python analysis/22_repair_and_settings.py
Outputs: analysis/outputs/repair_rates.csv, repair_sensitivity_v2.csv,
         temperature_robustness.csv
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluation import (FNAME_RE, OUTPUTS_DIR, REPO, load_ground_truth,
                        load_run_predictions, match_greedy)


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


m06 = load_module("m06", "06_repair_validation.py")
m12 = load_module("m12", "12_eval_all_v2.py")

LOGGED_FROM_IMAGE_V1 = 85
EXCLUDED = {"gpt-4o-mini"}
PROPRIETARY = ("gpt", "claude")


def repair_rates():
    rows = []
    for d, campaign in [("model_outputs", "v1"), ("model_outputs_v2", "v2")]:
        path = os.path.join(REPO, d)
        files = os.listdir(path)
        raws = {f[:-len("_raw.txt")] for f in files if f.endswith("_raw.txt")}
        for f in files:
            if not f.endswith(".json") or f.startswith("run_"):
                continue
            m = FNAME_RE.match(f) if campaign == "v1" else m12.PAT.match(f)
            if not m:
                continue
            model = m.group("model") if campaign == "v1" else m.group(2)
            img = int(m.group("img") if campaign == "v1" else m.group(1))
            if model in EXCLUDED or (campaign == "v1" and img < LOGGED_FROM_IMAGE_V1):
                continue
            stem = f[:-len(".json")]
            status = "valid"
            if stem in raws:
                status = ("empty" if os.path.getsize(os.path.join(path, stem + "_raw.txt")) == 0
                          else "repaired")
            rows.append((model, status))
    df = pd.DataFrame(rows, columns=["model", "status"])
    rates = (df.groupby("model").status.value_counts(normalize=True)
             .unstack(fill_value=0) * 100).reindex(columns=["valid", "empty", "repaired"], fill_value=0)
    rates["n_outputs"] = df.groupby("model").size()
    overall = df.status.value_counts(normalize=True) * 100
    return rates.reset_index(), overall


def repair_sensitivity(gt):
    acc = pd.read_csv(os.path.join(OUTPUTS_DIR, "portion_accounting.csv"))
    runs_v2 = m12.load_all_runs()
    v1_models = set(pd.read_parquet(os.path.join(OUTPUTS_DIR, "per_image_clean.parquet"),
                                    columns=["model"]).model)
    images = range(LOGGED_FROM_IMAGE_V1, 201)
    scenarios = {}
    for r in acc.itertuples():
        t = f"{r.temperature:.1f}"
        if r.model in v1_models:
            preds = load_run_predictions(r.model, t, empty_raw_as_empty=True)
            official = {i: pl for i, (pl, _) in preds.items()}
            raw_path = lambda i: os.path.join(  # noqa: E731
                REPO, "model_outputs", f"{i:02d}_{r.model}_T{t}_user01_sys01_raw.txt")
        else:
            official = runs_v2[(r.model, t, "r1")]
            raw_path = lambda i: os.path.join(  # noqa: E731
                REPO, "model_outputs_v2", f"{i:03d}_{r.model}_T{t}_user01_sys01_r1_raw.txt")
        no_repair, deterministic, n_rep = {}, {}, 0
        for i in images:
            rp = raw_path(i)
            if os.path.exists(rp) and os.path.getsize(rp) > 0:
                n_rep += 1
                no_repair[i] = []
                det = m06.deterministic_parse(open(rp, errors="replace").read())
                deterministic[i] = [(x["name"], x["portion"]) for x in det] if det else []
            else:
                no_repair[i] = deterministic[i] = official.get(i, [])
        scenarios[r.model] = ({i: official.get(i, []) for i in images},
                              no_repair, deterministic, n_rep)
    pseudo_runs = {(m, s, "x"): v[k] for m, v in scenarios.items()
                   for k, s in [(1, "no_repair"), (2, "deterministic")]}
    scores = m12.ensure_scores(pseudo_runs, gt, m12.score_map())

    def f1(by_img):
        tp = fp = fn = 0
        for i, pl in by_img.items():
            gl = gt[i]
            mat = np.array([[scores[(p, g)] for g, _ in gl] for p, _ in pl],
                           dtype=np.float32).reshape(len(pl), len(gl))
            res = match_greedy(mat, 0.5)
            tp, fp, fn = tp + res.tp, fp + res.fp, fn + res.fn
        return 2 * tp / (2 * tp + fp + fn)

    rows = [{"model": m, "n_repaired_img85_200": v[3], "f1_official": f1(v[0]),
             "f1_repaired_as_empty": f1(v[1]), "f1_deterministic_only": f1(v[2])}
            for m, v in scenarios.items()]
    return pd.DataFrame(rows).sort_values("f1_official", ascending=False)


def temperature_robustness():
    clean = json.load(open(os.path.join(OUTPUTS_DIR, "metrics_clean.json")))
    v1 = pd.DataFrame([v for v in clean.values()
                       if v["epsilon"] == 0.5 and v["model"] not in EXCLUDED])[["model", "temperature", "f1"]]
    v2 = pd.read_csv(os.path.join(OUTPUTS_DIR, "metrics_v2_all.csv"))
    v2 = v2[(v2.epsilon == 0.5) & (v2.run == "r1") & ~v2.model.isin(v1.model)][["model", "temperature", "f1"]]
    piv = pd.concat([v1, v2]).pivot_table(index="model", columns="temperature", values="f1")
    out = piv.copy()
    out.columns = [f"f1_T{c}" for c in out.columns]
    out["f1_best_T"] = piv.max(axis=1)
    out["f1_mean_T"] = piv.mean(axis=1)
    out = out.reset_index()
    open_piv = piv[[not m.startswith(PROPRIETARY) for m in piv.index]]
    summary = {
        "rho_mean_vs_best": spearmanr(out.f1_mean_T, out.f1_best_T).statistic,
        "best_open_by_T": {float(t): open_piv[t].idxmax() for t in open_piv.columns},
    }
    return out, summary


def main():
    gt = load_ground_truth()
    rates, overall = repair_rates()
    rates.to_csv(os.path.join(OUTPUTS_DIR, "repair_rates.csv"), index=False)
    print(rates.round(1).sort_values("repaired", ascending=False).to_string(index=False))
    print("overall (%):", overall.round(1).to_dict())

    sens = repair_sensitivity(gt)
    sens.to_csv(os.path.join(OUTPUTS_DIR, "repair_sensitivity_v2.csv"), index=False)
    print(sens.round(3).to_string(index=False))
    for c in ["f1_repaired_as_empty", "f1_deterministic_only"]:
        print(f"Spearman({c}, f1_official) = {spearmanr(sens[c], sens.f1_official).statistic:.3f}")

    temp, summary = temperature_robustness()
    temp.to_csv(os.path.join(OUTPUTS_DIR, "temperature_robustness.csv"), index=False)
    print(summary)
    print(temp[temp.model.str.startswith(PROPRIETARY)][["model", "f1_T1.0", "f1_best_T"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
