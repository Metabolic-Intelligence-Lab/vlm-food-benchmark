import os
import re
from collections import defaultdict

OUTPUT_DIR = "model_outputs"
EXPECTED_IMAGES = {f"{i:03d}" for i in range(1, 201)}

FILENAME_RE = re.compile(
    r"(?P<img>\d+?)_(?P<model>.+)_T(?P<temp>[0-9.]+)_user\d+_sys\d+\.json$"
)

results = defaultdict(set)

for root, _, files in os.walk(OUTPUT_DIR):
    for fname in files:
        match = FILENAME_RE.match(fname)
        if not match:
            continue

        img = match.group("img").zfill(3)  # <--- zero padding corretto
        model = match.group("model")
        temp = match.group("temp")

        results[(model, temp)].add(img)

print("=== MISSING OUTPUT REPORT ===\n")

for (model, temp), imgs_present in sorted(results.items()):
    missing = sorted(EXPECTED_IMAGES - imgs_present)
    if missing:
        print(f"Model: {model}")
        print(f"Temperature: {temp}")
        print(f"Missing {len(missing)} images:")
        print(", ".join(missing))
        print("-" * 60)
