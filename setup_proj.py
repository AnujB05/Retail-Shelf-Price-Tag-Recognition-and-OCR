import os
import random
import shutil
from pathlib import Path

# -------- USER CONFIG --------
SAMPLE_DATASET_DIR = "Sample Dataset"
PROJECT_ROOT = "project"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
# -----------------------------


def make_dirs():
    dirs = [
        f"{PROJECT_ROOT}/data/raw",
        f"{PROJECT_ROOT}/data/splits",
        f"{PROJECT_ROOT}/data/annotations",
        f"{PROJECT_ROOT}/data/crops",
        f"{PROJECT_ROOT}/data/results",

        f"{PROJECT_ROOT}/models/detector",
        f"{PROJECT_ROOT}/logs",

        f"{PROJECT_ROOT}/detection",
        f"{PROJECT_ROOT}/ocr",
        f"{PROJECT_ROOT}/pipeline",
        f"{PROJECT_ROOT}/evaluation",
        f"{PROJECT_ROOT}/configs",
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    print("✔ Directory structure created")


def get_image_list():
    sample_path = Path(SAMPLE_DATASET_DIR)

    images = sorted([
        str(p) for p in sample_path.glob("*.jpg")
    ])

    if len(images) != 20:
        print(f"⚠ Warning: Expected 20 images, found {len(images)}")

    print(f"Found {len(images)} images in Sample Dataset")
    return images


def copy_images_to_raw(images):
    for img in images:
        shutil.copy(img, f"{PROJECT_ROOT}/data/raw/")

    print("✔ Images copied to project/data/raw")


def create_splits(images):
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = images[:n_train]
    val = images[n_train:n_train + n_val]
    test = images[n_train + n_val:]

    def write_split(name, subset):
        out_path = f"{PROJECT_ROOT}/data/splits/{name}.txt"
        with open(out_path, "w") as f:
            for img in subset:
                fname = os.path.basename(img)
                f.write(f"{fname}\n")

    write_split("train", train)
    write_split("val", val)
    write_split("test", test)

    print("✔ Train / Val / Test splits created")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")


def main():
    print("\n=== PROJECT SETUP STARTED ===\n")

    make_dirs()
    images = get_image_list()
    copy_images_to_raw(images)
    create_splits(images)

    print("\n=== SETUP COMPLETE ===")
    print("Next step: annotation + detection pipeline\n")


if __name__ == "__main__":
    main()
