import json
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_food_items(label_data):
    parsed = []
    for task in label_data:
        image = task["data"]["image"]
        raw_text = task["annotations"][0]["result"][0]["value"]["text"][0]
        lines = raw_text.strip().split("\n")
        for line in lines:
            match = re.match(r"(.+):\s*([\d.]+)", line)
            if match:
                food = match.group(1).strip()
                grams = float(match.group(2))
                parsed.append({"image": image, "food": food, "portion": grams})
    return parsed

# Load and parse the exported file
with open("GroundTruth_GTMensa.json", "r") as f:
    data = json.load(f)

parsed_data = parse_food_items(data)

# Optional: save to CSV
import csv
with open("structured_food_labels.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "food", "portion"])
    writer.writeheader()
    writer.writerows(parsed_data)


# Make histogram of portion sizes

portion_values = [entry['portion'] for entry in parsed_data]

print('There are %i food items' %len(portion_values))

print('Median portion size: %s grams' %np.median(portion_values))
print('Mean portion size: %s grams' %np.mean(portion_values))
print('Min portion size: %s grams' %np.min(portion_values))
print('Max portion size: %s grams' %np.max(portion_values))


# Plot histogram
plt.figure(figsize=(8, 7))
plt.hist(portion_values, bins=30, histtype='step', color='k', lw=2)
plt.axvline(np.median(portion_values), c='r', ls='--', lw=2.5)
plt.axvline(np.mean(portion_values), c='b', ls=':', lw=3)
plt.xticks(ticks=range(0, int(max(portion_values)) + 50, 50))  # every 25 grams
plt.xlabel('Portion size [g]', fontsize=15)
plt.ylabel('Counts', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.tight_layout()
plt.savefig('GT_portions_histo.png')