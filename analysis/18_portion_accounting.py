"""Outcome accounting and sensitivity of the portion metrics.

For every model at its Table 2 configuration (eps=0.5, best-F1 temperature; first
campaign in clean mode, 2026 campaign run r1), accounts for every prediction
outcome and recomputes the portion metrics under alternative penalty schemes:

  P        matched items with a numerical estimate        -> |error| in every metric
  TP_noW   matched items without a numerical estimate     -> penalized as missed
  FN       ground-truth items not matched                 -> penalized as missed
  FP       unmatched predictions (incl. empty names)      -> ignored by NMAE/cpMAE,
                                                             predicted mass added by cpMAE+FP
  NMAE(w0)  (sum|err| + w0 * Q) / N_tot,  Q = N_tot - P,  w0 in {64.5, 129, 258}
  cpMAE     missed items penalized by their true weight
  cpMAE+FP  cpMAE + predicted mass of the false positives
  meal error  |sum of all predicted weights - sum of GT weights| per image (mean, in g)
              and its median relative value

Wrong matches: matched pairs that the LLM judge (llm_judge_results.jsonl, same
rules as 15_judge_v2.py) rates as different foods are re-scored as FN + FP.
Image-level bootstrap CIs (B=2000, seed 20260901) for cpMAE+FP and meal error.

Usage: python analysis/18_portion_accounting.py
Outputs: analysis/outputs/portion_accounting.csv, nmae_sensitivity.csv
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
from evaluation import (N_IMAGES, OUTPUTS_DIR, extract_grams_robust,
                        load_ground_truth, load_run_predictions, match_greedy)

spec = importlib.util.spec_from_file_location(
    "m12", os.path.join(HERE, "12_eval_all_v2.py"))
m12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m12)

EPS = 0.5
W0_VALUES = [64.5, 129.0, 258.0]
JUDGE_LO, JUDGE_HI = 0.30, 0.95
SEED, B = 20260901, 2000


def table2_configs():
    """model -> (campaign, temperature string) at the best-F1 temperature, eps=0.5."""
    clean = json.load(open(os.path.join(OUTPUTS_DIR, "metrics_clean.json")))
    best = {}
    for v in clean.values():
        if v["epsilon"] == EPS and (v["model"] not in best or v["f1"] > best[v["model"]][0]):
            best[v["model"]] = (v["f1"], f"{v['temperature']:.1f}")
    cfgs = {m: ("v1", t) for m, (_, t) in best.items()}
    v2 = pd.read_csv(os.path.join(OUTPUTS_DIR, "metrics_v2_all.csv"))
    v2 = v2[(v2.epsilon == EPS) & (v2.run == "r1")]
    for m, g in v2.groupby("model"):
        if m not in cfgs:
            cfgs[m] = ("v2", f"{g.loc[g.f1.idxmax()].temperature:.1f}")
    return cfgs


def load_verdicts():
    verdicts = {}
    with open(os.path.join(OUTPUTS_DIR, "llm_judge_results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            verdicts[(r["pred_name"], r["gt_name"])] = r["verdict"] == "same"
    return verdicts


def judge_same(p, g, score, verdicts):
    if score < JUDGE_LO:
        return False
    if score >= JUDGE_HI:
        return True
    return verdicts.get((p, g), False)


def account_model(by_img, n_empty, gt, scores, verdicts):
    """Per-image arrays of every outcome, for the CE matching and the judge-filtered one."""
    cols = ["n_gt", "tp", "tp_w", "fn", "fp", "fp_w", "abs_err", "rel_err_list",
            "missed_w", "fp_mass", "meal_abs", "gt_total",
            "j_abs_err", "j_p", "j_missed_w", "j_fp_mass", "n_wrong"]
    rows = {c: [] for c in cols}
    for img in range(1, N_IMAGES + 1):
        plist, gl = by_img.get(img, []), gt[img]
        mat = np.array([[scores[(p, g)] for g, _ in gl] for p, _ in plist],
                       dtype=np.float32).reshape(len(plist), len(gl))
        res = match_greedy(mat, EPS)
        weights = [extract_grams_robust(w) for _, w in plist]
        abs_err, rel, matched_w, j_abs, j_matched_w = 0.0, [], set(), 0.0, set()
        j_fp_mass, n_wrong, tp_w = 0.0, 0, 0
        for pi, gi, sc in res.pairs:
            w = weights[pi]
            same = judge_same(plist[pi][0], gl[gi][0], sc, verdicts)
            n_wrong += not same
            if w is None:
                continue
            tp_w += 1
            e = abs(w - gl[gi][1])
            abs_err += e
            rel.append(e / gl[gi][1])
            matched_w.add(gi)
            if same:
                j_abs += e
                j_matched_w.add(gi)
            else:
                j_fp_mass += w
        fp_mass = sum(weights[pi] for pi in res.fp_pred_idx if weights[pi] is not None)
        gt_total = sum(w for _, w in gl)
        rows["n_gt"].append(len(gl))
        rows["tp"].append(res.tp)
        rows["tp_w"].append(tp_w)
        rows["fn"].append(res.fn)
        rows["fp"].append(res.fp)
        rows["fp_w"].append(sum(weights[pi] is not None for pi in res.fp_pred_idx))
        rows["abs_err"].append(abs_err)
        rows["rel_err_list"].append(rel)
        rows["missed_w"].append(sum(w for j, (_, w) in enumerate(gl) if j not in matched_w))
        rows["fp_mass"].append(fp_mass)
        rows["meal_abs"].append(abs(sum(w for w in weights if w is not None) - gt_total))
        rows["gt_total"].append(gt_total)
        rows["j_abs_err"].append(j_abs)
        rows["j_p"].append(len(j_matched_w))
        rows["j_missed_w"].append(sum(w for j, (_, w) in enumerate(gl) if j not in j_matched_w))
        rows["j_fp_mass"].append(fp_mass + j_fp_mass)
        rows["n_wrong"].append(n_wrong)
    arr = {k: np.array(v, dtype=float) for k, v in rows.items() if k != "rel_err_list"}
    arr["fp"][0] += n_empty  # empty-name predictions are automatic FPs (clean mode)
    rel_all = np.array([x for lst in rows["rel_err_list"] for x in lst])
    return arr, rel_all


def summarize(model, arr, rel_all, idx):
    n_tot = arr["n_gt"].sum()
    P = arr["tp_w"].sum()
    sae = arr["abs_err"].sum()
    out = {
        "model": model, "TP_with_estimate": int(P),
        "TP_without_estimate": int(arr["tp"].sum() - P),
        "FN": int(arr["fn"].sum()), "FP": int(arr["fp"].sum()),
        "FP_with_estimate": int(arr["fp_w"].sum()),
        "MAE": sae / P if P else np.nan,
        "median_rel_err": float(np.median(rel_all)) if len(rel_all) else np.nan,
    }
    for w0 in W0_VALUES:
        out[f"NMAE_w0_{w0:g}"] = (sae + w0 * (n_tot - P)) / n_tot
    out["cpMAE"] = (sae + arr["missed_w"].sum()) / n_tot
    out["cpMAE_FP"] = (sae + arr["missed_w"].sum() + arr["fp_mass"].sum()) / n_tot
    out["meal_MAE"] = arr["meal_abs"].mean()
    out["meal_median_rel_err"] = float(np.median(arr["meal_abs"] / arr["gt_total"]))
    # bootstrap CIs (shared resample indices across models)
    s = {k: arr[k][idx].sum(axis=1) for k in ["abs_err", "missed_w", "fp_mass", "n_gt", "meal_abs"]}
    cpfp = (s["abs_err"] + s["missed_w"] + s["fp_mass"]) / s["n_gt"]
    meal = s["meal_abs"] / N_IMAGES
    out["cpMAE_FP_lo"], out["cpMAE_FP_hi"] = np.percentile(cpfp, [2.5, 97.5])
    out["meal_MAE_lo"], out["meal_MAE_hi"] = np.percentile(meal, [2.5, 97.5])
    # judge-filtered: CE matches rated different by the LLM judge -> FN + FP
    jP = arr["j_p"].sum()
    out["n_matches_judged_wrong"] = int(arr["n_wrong"].sum())
    out["frac_matches_judged_wrong"] = arr["n_wrong"].sum() / arr["tp"].sum() if arr["tp"].sum() else np.nan
    out["NMAE_judge"] = (arr["j_abs_err"].sum() + 129.0 * (n_tot - jP)) / n_tot
    out["cpMAE_judge"] = (arr["j_abs_err"].sum() + arr["j_missed_w"].sum()) / n_tot
    return out


def main():
    gt = load_ground_truth()
    scores = m12.score_map()
    verdicts = load_verdicts()
    runs_v2 = m12.load_all_runs()
    idx = np.random.default_rng(SEED).integers(0, N_IMAGES, size=(B, N_IMAGES))

    rows = []
    for model, (campaign, temp) in sorted(table2_configs().items()):
        if campaign == "v1":
            preds = load_run_predictions(model, temp, empty_raw_as_empty=True)
            by_img = {i: pl for i, (pl, _) in preds.items()}
            n_empty = sum(e for _, e in preds.values())
        else:
            by_img, n_empty = runs_v2[(model, temp, "r1")], 0
        arr, rel_all = account_model(by_img, n_empty, gt, scores, verdicts)
        row = summarize(model, arr, rel_all, idx)
        row["temperature"] = float(temp)
        assert row["TP_with_estimate"] + row["TP_without_estimate"] + row["FN"] == 870
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("NMAE_w0_129")
    df.to_csv(os.path.join(OUTPUTS_DIR, "portion_accounting.csv"), index=False)

    # rank agreement of every variant with the paper's NMAE (w0 = 129 g), restricted
    # to models with enough numerical estimates for portion metrics to be meaningful
    usable = df[df.TP_with_estimate > 50]
    sens = []
    for col in ["MAE", "NMAE_w0_64.5", "NMAE_w0_258", "cpMAE", "cpMAE_FP",
                "meal_MAE", "NMAE_judge", "cpMAE_judge"]:
        sens.append({"metric": col, "n_models": len(usable),
                     "spearman_vs_NMAE129": spearmanr(usable[col], usable["NMAE_w0_129"]).statistic})
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv(os.path.join(OUTPUTS_DIR, "nmae_sensitivity.csv"), index=False)

    pd.set_option("display.width", 250)
    show = ["model", "temperature", "TP_with_estimate", "TP_without_estimate", "FN", "FP",
            "MAE", "NMAE_w0_64.5", "NMAE_w0_129", "NMAE_w0_258", "cpMAE", "cpMAE_FP",
            "meal_MAE", "frac_matches_judged_wrong", "NMAE_judge"]
    print(df[show].round(2).to_string(index=False))
    print("\n" + sens_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
