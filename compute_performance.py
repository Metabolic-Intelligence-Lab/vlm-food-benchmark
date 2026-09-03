import json
import os
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from glob import glob
import re
from collections import defaultdict, Counter
import sys
from sentence_transformers.cross_encoder import CrossEncoder

issue_counter = Counter()
bad_files = []

# --- Parameters ---
GROUND_TRUTH_CSV = "structured_food_labels.csv"
MODEL_OUTPUT_DIR = "model_outputs"

#if sys.argv[1]:
#    FUZZY_THRESHOLD = int(sys.argv[1])
#else:
#    FUZZY_THRESHOLD = 50  # 0-100

if sys.argv[1]:
    CrossEnc_thr = float(sys.argv[1])
else:
    CrossEnc_thr = 0.7  # 0-1


# Load a pretrained CrossEncoder model
CrossEnc_model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

# --- Here we select only the models/T combinations which produced an output for all the images ---
# Set the output directory and expected number of images
expected_images = {f"{i:02d}" for i in range(1, 201)}  # {"001", "002", ..., "200"}
# Regular expression to extract parts from the filename
pattern = re.compile(r"(?P<image>\d{2,3})_(?P<model>.+?)_T(?P<temp>\d\.\d)_user01_sys01\.json")
# Dictionary to track available image outputs for each (model, temperature) pair
output_tracker = defaultdict(set)
# Scan files
for fname in os.listdir(MODEL_OUTPUT_DIR):
    match = pattern.match(fname)
    if match:
        image = match.group("image")
        model = match.group("model")
        temp = match.group("temp")
        output_tracker[(model, temp)].add(image)
# Find only the (model, temp) pairs that have all the images
complete_runs = [(model, temp) for (model, temp), images in output_tracker.items() if images == expected_images]
# Print or use the result
print("Valid (model, temperature) combinations with all the images:")
for model, temp in sorted(complete_runs):
    print(f"Model: {model}, Temperature: {temp}")


# --- Load Ground Truth ---
gt_df = pd.read_csv(GROUND_TRUTH_CSV)


def load_and_normalize_preds(file):
    with open(file) as f:
        data = json.load(f)

    # Step 1: get food_items safely
    if isinstance(data, dict):
        items = data.get("food_items", [])
    elif isinstance(data, list):
        items = data
    else:
        return [], ["root_not_dict_or_list"]

    if not isinstance(items, list):
        return [], ["food_items_not_list"]

    normalized = []
    issues = []

    for item in items:
        if isinstance(item, dict):
            normalized.append(item)

        elif isinstance(item, str):
            issues.append("item_is_string")
            normalized.append({
                "name": item,
                "portion_estimate": ""
            })

        else:
            issues.append("item_invalid_type")

    return normalized, issues

# Normalize ground truth food names and group per image
def normalize(text):
    return text.strip().lower()

gt_df['food'] = gt_df['food'].astype(str).apply(normalize)
gt_df['image'] = gt_df['image'].astype(str).str.zfill(2)

gt_dict = {
    image: [(normalize(row['food']), float(row['portion'])) for _, row in group.iterrows()]
    for image, group in gt_df.groupby("image")
}

# --- Helper to extract numeric grams from prediction ---
def extract_grams(text):
    text = text.lower().replace("grams", "").replace("gram", "").replace("g", "").strip()
    try:
        return float(text)
    except ValueError:
        return None

'''
# --- Fuzzy matching function ---
def best_gt_match(pred_name, gt_names, threshold=FUZZY_THRESHOLD):
    best_score = 0
    best_index = None
    for i, gt_name in enumerate(gt_names):
        score = fuzz.token_sort_ratio(pred_name, gt_name)
        if score > best_score and score >= threshold:
            best_score = score
            best_index = i
    return best_index
'''

def best_semantic_match(pred_name, gt_names, threshold=CrossEnc_thr):

    sentence_combinations = [[pred_name, sentence] for sentence in gt_names]
    scores = CrossEnc_model.predict(sentence_combinations) 

    best_score = np.max(scores)
    best_index = np.argmax(scores)

    return best_index if best_score >= threshold else None, best_score


all_metrics = {}
all_matches = []
all_mismatches = []

for TARGET_MODEL, TARGET_TEMP in complete_runs:
    #if TARGET_MODEL == "llava-phi3:3.8b":
    #    continue
    #if TARGET_MODEL == "bakllava:7b":
    #    continue
    # --- Collect model output files ---
    file_pattern = os.path.join(MODEL_OUTPUT_DIR, f"*_{TARGET_MODEL}_T{TARGET_TEMP}_*.json")
    files = glob(file_pattern)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    gt_portions = []
    pred_portions = []

    for file in files:
        image_id = os.path.basename(file).split("_")[0].zfill(2)
        gt_key = next((key for key in gt_dict if key.endswith(f"{image_id}.jpg")), None) # type: ignore
        gt_items = gt_dict[gt_key] # type: ignore

        gt_names = [name for name, _ in gt_items]
        matched_gt = set()

        #with open(file) as f:
        #    data = json.load(f)
        preds, issues = load_and_normalize_preds(file)

        if issues:
            issue_counter.update(issues)
            bad_files.append(file)

        #preds = data.get("food_items", [])
        for pred in preds:
            pred_name = normalize(pred.get("name", ""))
            portion = extract_grams(str(pred.get("portion_estimate", "")))

            #match_idx = best_gt_match(pred_name, gt_names, threshold=FUZZY_THRESHOLD)
            match_idx, best_score = best_semantic_match(pred_name, gt_names, threshold=CrossEnc_thr)
            if match_idx is not None and match_idx not in matched_gt:
                matched_gt.add(match_idx)
                true_positives += 1

                if portion is not None:
                    gt_portions.append(gt_items[match_idx][1])
                    pred_portions.append(portion)

                all_matches.append(
                    f"[{TARGET_MODEL} | T={TARGET_TEMP} | image {image_id}] GT: {gt_items[match_idx][0]} (portion: {gt_items[match_idx][1]})  <--->  PRED: {pred_name} (portion: {portion}), score: {str(best_score)}"
                )

            else:
                false_positives += 1
                all_mismatches.append(
                    f"[{TARGET_MODEL} | T={TARGET_TEMP} | image {image_id}] Unmatched prediction: {pred_name} (portion: {portion})"
                )

        false_negatives += len(gt_items) - len(matched_gt)

    for i, (name, _) in enumerate(gt_items):
        if i not in matched_gt:
            false_negatives += 1
            all_mismatches.append(
                f"[{TARGET_MODEL} | T={TARGET_TEMP} | image {image_id}] Missed GT: {name}"
    )

    # --- Metrics ---
    accuracy = true_positives / (true_positives + false_positives + false_negatives) if true_positives + false_positives + false_negatives else 0
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0
    recall = true_positives /   (true_positives + false_negatives) if true_positives + false_negatives else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    if gt_portions:
        gt_portions = np.array(gt_portions)
        pred_portions = np.array(pred_portions)
        mae = np.mean(np.abs(gt_portions - pred_portions))
        rmse = np.sqrt(np.mean((gt_portions - pred_portions) ** 2))
        mape = np.mean(np.abs((gt_portions - pred_portions) / gt_portions)) * 100
        default_penalty = 129 # grams
        n_total_gt = sum(len(v) for v in gt_dict.values())
        n_matched = len(gt_portions)
        # IMPORTANT: the "misses" are not just the false negatives! There might be cases in which the item is correctly identified, but then the VLM fails at providing
        # an estimate of the portion size. Therefore, the "misses" are the difference between the total number of food and the predicted items (with a portion estimate) 
        n_missed = n_total_gt - n_matched

        if n_total_gt > 0:
            total_error = np.sum(np.abs(gt_portions - pred_portions)) + default_penalty * n_missed
            normalized_mae = total_error / n_total_gt
        else:
            normalized_mae = None

        # Save the true and predicted values in a table, for further examination
        filename_csv = "GT_PRED_" + TARGET_MODEL + "_" + TARGET_TEMP + "_" + str(CrossEnc_thr) + ".csv"
        np.savetxt("model_performance/" + filename_csv, 
                   np.column_stack((gt_portions, pred_portions)), 
                   delimiter=",", header="GT,pred")


    else:
        mae = rmse = mape = normalized_mae = None 

    key = f"{TARGET_MODEL}_T{TARGET_TEMP}"
    all_metrics[key] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae": mae,
        "normalized_mae": normalized_mae,
        "rmse": rmse,
        "mape": mape
    }

    # --- Report ---
    print(f"---- Evaluation for {TARGET_MODEL} at T={TARGET_TEMP} ----")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
    if mae is not None:
        print(f"MAE: {mae:.2f} g, RMSE: {rmse:.2f} g, MAPE: {mape:.2f}%\n")
    else:
        print("No matched items for portion error metrics.\n")
        
        
# Save metrics as JSON
with open(f"model_performance/evaluation_metrics_CrossEncThr{str(CrossEnc_thr)}.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

# Save matches
with open(f"model_performance/matches_CrossEncThr{str(CrossEnc_thr)}.txt", "w") as f:
    f.write("\n".join(all_matches))

# Save mismatches
with open(f"model_performance/mismatches_CrossEncThr{str(CrossEnc_thr)}.txt", "w") as f:
    f.write("\n".join(all_mismatches))

print("\n=== JSON Schema Issues ===")
for issue, count in issue_counter.most_common():
    print(f"{issue}: {count}")

print(f"\nFiles with issues: {len(bad_files)}")

with open("bad_prediction_files.txt", "w") as f:
    for bf in bad_files:
        f.write(bf + "\n")
