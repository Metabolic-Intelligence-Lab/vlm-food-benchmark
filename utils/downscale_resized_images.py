from pathlib import Path
from PIL import Image


def main():
    src = Path("GT_Mensa_2025/resized_images")
    #dst = Path("GT_Mensa_2025/resized_images_748")  # point to src to overwrite in place
    dst = src
    dst.mkdir(parents=True, exist_ok=True)

    max_size = (748, 748)

    for img_path in sorted(src.glob("*.jpg")):
        with Image.open(img_path) as img:
            img.thumbnail(max_size)
            target = dst / img_path.name
            img.save(target, quality=90)


if __name__ == "__main__":
    main()
