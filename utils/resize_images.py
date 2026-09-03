from PIL import Image
import os
import csv

input_dir = "./GT_Mensa_2025"
output_dir = "./GT_Mensa_2025/resized_images/"
log_file = "./GT_Mensa_2025/processed_images.txt"
mapping_file = "./GT_Mensa_2025/image_mapping.csv"


os.makedirs(output_dir, exist_ok=True)

# Set target resolution (width, height)
max_size = (1024, 1024)


# Load already processed images
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        processed = set(line.strip() for line in f)
else:
    processed = set()


# Collect all image paths recursively
image_paths = []
for root, dirs, files in os.walk(input_dir):

    # Skip the output_dir if it's in the list of subdirs
    dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(root, d)) != os.path.abspath(output_dir)]

    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg')):
            full_path = os.path.join(root, file)
            if full_path not in processed:
                image_paths.append(full_path)


# Get the next available index for output filenames
existing_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.jpg')]
existing_indices = [int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()]
next_idx = max(existing_indices + [0]) + 1

# Check if we need to write a CSV header
write_header = not os.path.exists(mapping_file)

# Process and log new images
with open(log_file, 'a') as log, open(mapping_file, 'a', newline='') as map_csv:
    writer = csv.writer(map_csv)
    if write_header:
        writer.writerow(["original_path", "resized_filename"])

    for img_path in sorted(image_paths):
        try:
            img = Image.open(img_path)
            img.thumbnail(max_size)

            output_filename = f"{next_idx:03d}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            img.save(output_path, quality=90)

            # Log processed path
            log.write(img_path + '\n')

            # Write mapping entry
            writer.writerow([img_path, output_filename])

            next_idx += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")