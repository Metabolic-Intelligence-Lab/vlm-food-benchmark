"""WP5 - Validate the gemma3:4b JSON-repair step.

For every output that went through the second-LLM repair (a *_raw.txt exists),
re-parse the raw text with a deterministic tolerant parser and compare with the
repaired JSON actually used in the benchmark:
  - how many raw outputs the deterministic parser can recover
  - agreement between repaired and deterministically-parsed items
    (food names via difflib similarity >= 0.85, portion values exactly)
  - items added or dropped by the repair LLM
Additionally, a sensitivity analysis: F1 per (model, T, eps) computed on
non-repaired images only vs all images.

Usage: analysis_env/bin/python analysis/06_repair_validation.py
Outputs: analysis/outputs/repair_validation.csv, repair_sensitivity.csv
"""
import difflib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import (MODEL_OUTPUT_DIR, OUTPUTS_DIR, THRESHOLDS,
                        load_and_normalize_preds, normalize)

RAW_RE = re.compile(
    r"^(?P<img>\d{1,3})_(?P<model>.+?)_T(?P<temp>\d(?:\.\d+)?)_user01_sys01_raw\.txt$"
)


def deterministic_parse(text):
    """Tolerant parser: markdown fences, sub-object extraction, trailing commas,
    single quotes, truncated arrays. Returns list of dicts or None."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()

    candidates = [t]
    # substring from first { to last }
    if "{" in t and "}" in t:
        candidates.append(t[t.index("{"): t.rindex("}") + 1])
    # substring from first [ to last ]
    if "[" in t and "]" in t:
        candidates.append(t[t.index("["): t.rindex("]") + 1])

    def attempts(s):
        yield s
        yield re.sub(r",\s*([}\]])", r"\1", s)              # trailing commas
        yield re.sub(r",\s*([}\]])", r"\1", s).replace("'", '"')
        # truncated output: close open brackets
        open_b = s.count("{") - s.count("}")
        open_a = s.count("[") - s.count("]")
        if open_b >= 0 and open_a >= 0 and (open_b or open_a):
            fixed = re.sub(r",\s*$", "", s.rstrip())
            fixed = re.sub(r':\s*"?[^",}\]]*$', ': ""', fixed)
            yield fixed + "}" * open_b + "]" * open_a + "}" * 0

    for cand in candidates:
        for att in attempts(cand):
            try:
                data = json.loads(att)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                items = data.get("food_items", [])
            elif isinstance(data, list):
                items = data
            else:
                continue
            if isinstance(items, list):
                out = []
                for it in items:
                    if isinstance(it, dict) and normalize(it.get("name", "")):
                        out.append({
                            "name": normalize(it.get("name", "")),
                            "portion": str(it.get("portion_estimate", "")),
                        })
                    elif isinstance(it, str) and normalize(it):
                        out.append({"name": normalize(it), "portion": ""})
                return out
    return None


def sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def compare_items(det_items, rep_items):
    """Greedy name alignment by similarity; returns match stats."""
    used = set()
    n_name_match = n_portion_match = 0
    for d in det_items:
        best_j, best_s = None, 0.0
        for j, r in enumerate(rep_items):
            if j in used:
                continue
            s = sim(d["name"], r["name"])
            if s > best_s:
                best_j, best_s = j, s
        if best_j is not None and best_s >= 0.85:
            used.add(best_j)
            n_name_match += 1
            dp = re.sub(r"[^\d.]", "", d["portion"])
            rp = re.sub(r"[^\d.]", "", rep_items[best_j]["portion"])
            if dp and dp == rp:
                n_portion_match += 1
    return n_name_match, n_portion_match


def main():
    rows = []
    for fname in sorted(os.listdir(MODEL_OUTPUT_DIR)):
        m = RAW_RE.match(fname)
        if not m:
            continue
        raw_path = os.path.join(MODEL_OUTPUT_DIR, fname)
        json_path = raw_path.replace("_raw.txt", ".json")
        with open(raw_path, errors="replace") as f:
            raw_text = f.read()
        det = deterministic_parse(raw_text)
        if os.path.exists(json_path):
            try:
                preds, _ = load_and_normalize_preds(json_path)
                rep = [{"name": normalize(p.get("name", "")),
                        "portion": str(p.get("portion_estimate", ""))}
                       for p in preds if normalize(p.get("name", ""))]
            except Exception:
                rep = None
        else:
            rep = None

        row = {
            "model": m.group("model"), "temperature": m.group("temp"),
            "image": int(m.group("img")),
            "det_parse_ok": det is not None,
            "n_det_items": len(det) if det is not None else 0,
            "repaired_exists": rep is not None,
            "n_rep_items": len(rep) if rep is not None else 0,
            "n_name_match": 0, "n_portion_match": 0,
        }
        if det is not None and rep is not None:
            nm, pm = compare_items(det, rep)
            row["n_name_match"] = nm
            row["n_portion_match"] = pm
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUTS_DIR, "repair_validation.csv"), index=False)

    both = df[df.det_parse_ok & df.repaired_exists]
    tot_det = both.n_det_items.sum()
    tot_rep = both.n_rep_items.sum()
    print(f"raw files: {len(df)}")
    print(f"deterministic parser recovers: {df.det_parse_ok.mean() * 100:.1f}%")
    print(f"repaired json exists:          {df.repaired_exists.mean() * 100:.1f}%")
    print(f"\nOn {len(both)} files parsed by both:")
    print(f"  items (deterministic): {tot_det}, items (repaired): {tot_rep}")
    print(f"  name preservation:    {both.n_name_match.sum() / tot_det * 100:.1f}% "
          f"of deterministic items found in repaired output")
    print(f"  portion preservation: {both.n_portion_match.sum() / max(both.n_name_match.sum(), 1) * 100:.1f}% "
          f"of name-matched items keep the exact number")
    print(f"  items added by repair LLM: {max(tot_rep - both.n_name_match.sum(), 0)} "
          f"({(tot_rep - both.n_name_match.sum()) / max(tot_rep, 1) * 100:.1f}% of repaired)")

    # ---- sensitivity: F1 with vs without repaired images ----
    repaired_flags = {(r.model, r.temperature, r.image) for r in df.itertuples()}
    per_image = pd.read_parquet(os.path.join(OUTPUTS_DIR, "per_image_greedy.parquet"))
    per_image["repaired"] = [
        (r.model, f"{r.temperature:.1f}", r.image) in repaired_flags
        for r in per_image.itertuples()
    ]
    sens = []
    for (model, temp, eps), g in per_image.groupby(["model", "temperature", "epsilon"]):
        n_rep = int(g.repaired.sum())
        if n_rep == 0:
            continue

        def f1_of(sub):
            tp, fp, fn = sub.tp.sum(), sub.fp.sum(), sub.fn.sum()
            p = tp / (tp + fp) if tp + fp else 0
            r = tp / (tp + fn) if tp + fn else 0
            return 2 * p * r / (p + r) if p + r else 0

        sens.append({
            "model": model, "temperature": temp, "epsilon": eps,
            "n_images_repaired": n_rep,
            "f1_all": f1_of(g),
            "f1_nonrepaired_only": f1_of(g[~g.repaired]),
            "f1_repaired_only": f1_of(g[g.repaired]),
        })
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv(os.path.join(OUTPUTS_DIR, "repair_sensitivity.csv"), index=False)
    sens_df["delta"] = sens_df.f1_repaired_only - sens_df.f1_nonrepaired_only
    print(f"\nsensitivity rows: {len(sens_df)}")
    print("F1(repaired images) - F1(non-repaired images), per combo:")
    print(sens_df["delta"].describe().round(3))


if __name__ == "__main__":
    main()
