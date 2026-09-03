"""WP0 - Full inventory of model outputs and ground-truth integrity checks.

Stdlib-only so it can run with any Python. Produces analysis/outputs/inventory.json
and prints a human-readable report. Read-only: never touches model_outputs content.
"""
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_OUTPUT_DIR = os.path.join(REPO, "model_outputs")
GT_CSV = os.path.join(REPO, "structured_food_labels.csv")
MAPPING_CSV = os.path.join(REPO, "GT_Mensa_2025", "image_participants.csv")
OUT_JSON = os.path.join(REPO, "analysis", "outputs", "inventory.json")

N_IMAGES = 200

# Output filenames look like: 01_gemma3:12b_T0.0_user01_sys01.json  (image index NOT zero-padded to 3)
FNAME_RE = re.compile(
    r"^(?P<img>\d{1,3})_(?P<model>.+?)_T(?P<temp>\d(?:\.\d+)?)_user(?P<usr>\d+)_sys(?P<sys>\d+)(?P<raw>_raw)?\.(?P<ext>json|txt)$"
)


def inventory_outputs():
    runs = defaultdict(lambda: {"images": set(), "raw_images": set(), "unparsed": []})
    unmatched = []
    for fname in sorted(os.listdir(MODEL_OUTPUT_DIR)):
        m = FNAME_RE.match(fname)
        if not m:
            unmatched.append(fname)
            continue
        key = (m.group("model"), m.group("temp"))
        img = int(m.group("img"))
        if m.group("raw"):
            runs[key]["raw_images"].add(img)
        else:
            runs[key]["images"].add(img)
    return runs, unmatched


def check_json_validity(runs):
    """Count output files that fail json.load or have unexpected structure."""
    invalid = Counter()
    empty_list = Counter()
    for (model, temp), info in runs.items():
        for img in info["images"]:
            # reconstruct filename (image index as written: 1-3 digits, no padding rule -> try both)
            for pat in (f"{img:02d}", f"{img:03d}", str(img)):
                path = os.path.join(MODEL_OUTPUT_DIR, f"{pat}_{model}_T{temp}_user01_sys01.json")
                if os.path.exists(path):
                    break
            try:
                with open(path) as f:
                    data = json.load(f)
                items = data.get("food_items", data) if isinstance(data, dict) else data
                if isinstance(items, list) and len(items) == 0:
                    empty_list[(model, temp)] += 1
            except (json.JSONDecodeError, UnicodeDecodeError):
                invalid[(model, temp)] += 1
    return invalid, empty_list


def inventory_ground_truth():
    with open(GT_CSV) as f:
        rows = list(csv.DictReader(f))
    foods = [r["food"].strip().lower() for r in rows]
    images = sorted({r["image"] for r in rows})
    portions = [float(r["portion"]) for r in rows]
    per_image = Counter(r["image"] for r in rows)
    portions_sorted = sorted(portions)
    n = len(portions_sorted)
    median = (portions_sorted[n // 2] if n % 2 else
              (portions_sorted[n // 2 - 1] + portions_sorted[n // 2]) / 2)
    return {
        "n_instances": len(rows),
        "n_unique_food_names": len(set(foods)),
        "n_images_with_labels": len(images),
        "items_per_image_min": min(per_image.values()),
        "items_per_image_max": max(per_image.values()),
        "items_per_image_mean": round(len(rows) / len(images), 2),
        "portion_median_g": median,
        "portion_mean_g": round(sum(portions) / len(portions), 1),
        "portion_min_g": min(portions),
        "portion_max_g": max(portions),
    }


def inventory_annotators():
    with open(MAPPING_CSV) as f:
        rows = list(csv.DictReader(f))
    per_annotator = Counter()
    img_to_annotator = {}
    for r in rows:
        annotator = r["participant_id"]
        per_annotator[annotator] += 1
        img_to_annotator[r["resized_filename"]] = annotator
    return dict(per_annotator), img_to_annotator


def main():
    runs, unmatched = inventory_outputs()
    gt = inventory_ground_truth()
    annotators, img_map = inventory_annotators()

    expected = set(range(1, N_IMAGES + 1))
    report = {}
    for (model, temp), info in sorted(runs.items()):
        missing = sorted(expected - info["images"])
        report[f"{model}_T{temp}"] = {
            "model": model,
            "temperature": temp,
            "n_outputs": len(info["images"]),
            "n_missing": len(missing),
            "missing_images": missing,
            "n_raw_repairs": len(info["raw_images"]),
        }

    complete = {k for k, v in report.items() if v["n_missing"] == 0}
    incomplete = {k: v for k, v in report.items() if v["n_missing"] > 0}

    models = sorted({v["model"] for v in report.values()})
    summary = {
        "n_model_temp_combinations": len(report),
        "n_complete_combinations": len(complete),
        "n_incomplete_combinations": len(incomplete),
        "n_models": len(models),
        "models": models,
        "total_raw_repair_files": sum(v["n_raw_repairs"] for v in report.values()),
        "unmatched_filenames": unmatched,
        "ground_truth": gt,
        "images_per_annotator": annotators,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"summary": summary, "runs": report,
                   "image_to_annotator": img_map}, f, indent=2)

    print("=== GROUND TRUTH ===")
    for k, v in gt.items():
        print(f"  {k}: {v}")
    print("\n=== IMAGES PER ANNOTATOR ===")
    for k, v in sorted(annotators.items()):
        print(f"  {k}: {v}")
    print(f"\n=== MODEL RUNS ===")
    print(f"  {len(report)} (model, T) combinations, {len(complete)} complete, "
          f"{len(incomplete)} incomplete")
    if incomplete:
        print("\n  INCOMPLETE:")
        for k, v in sorted(incomplete.items()):
            print(f"    {k}: {v['n_outputs']}/200 outputs "
                  f"(missing {v['n_missing']})")
    if unmatched:
        print(f"\n  Unmatched filenames ({len(unmatched)}):")
        for u in unmatched[:10]:
            print(f"    {u}")
    print("\n=== RAW REPAIRS PER MODEL (JSON fixed by gemma3:4b) ===")
    per_model_raw = defaultdict(int)
    per_model_tot = defaultdict(int)
    for v in report.values():
        per_model_raw[v["model"]] += v["n_raw_repairs"]
        per_model_tot[v["model"]] += v["n_outputs"]
    for mdl in sorted(per_model_raw, key=lambda m: -per_model_raw[m]):
        tot = per_model_tot[mdl]
        pct = 100 * per_model_raw[mdl] / tot if tot else 0
        print(f"  {mdl}: {per_model_raw[mdl]} repairs ({pct:.1f}% of outputs)")
    print(f"\nSaved full inventory to {OUT_JSON}")


if __name__ == "__main__":
    main()
