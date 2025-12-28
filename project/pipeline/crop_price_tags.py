import os
import json
import cv2
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path("project")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CROPS_DIR = PROJECT_ROOT / "data" / "crops"

MODEL_PATH = PROJECT_ROOT / "models" / "price_tag_yolo" / "exp_price_tag3" / "weights" / "best.pt"

CONF_THRESHOLD = 0.25      
IOU_THRESHOLD = 0.45       
MAX_DETS = 300

def ensure_dirs():
    (CROPS_DIR / "images").mkdir(parents=True, exist_ok=True)
    (CROPS_DIR / "metadata").mkdir(parents=True, exist_ok=True)


def list_images_in_split(split_name):


    split_dir = RAW_DIR / split_name

    if split_dir.exists():
        return list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))

    # fallback — flat directory
    return list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png"))



def crop_and_save(img_path, detections, split_name):
    img = cv2.imread(str(img_path))

    if img is None:
        print(f"⚠ Could not read image: {img_path}")
        return []

    h, w, _ = img.shape
    crops_meta = []
    tag_index = 0

    for det in detections:
        conf = float(det["confidence"])
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, det["bbox"])

        # clamp to image bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]

        crop_name = f"{img_path.stem}_tag{tag_index}.jpg"
        crop_path = CROPS_DIR / "images" / split_name / crop_name

        crop_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop_path), crop)

        crops_meta.append({
            "crop_name": crop_name,
            "crop_path": str(crop_path),
            "source_image": img_path.name,
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "split": split_name
        })

        tag_index += 1

    return crops_meta



def detect_price_tags(model, img_path):
    results = model.predict(
        source=str(img_path),
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        max_det=MAX_DETS,
        verbose=False,
        device="cpu"
    )

    detections = []

    for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf[0])
            })

    return detections


def process_split(model, split_name):
    print(f"\n=== Processing {split_name} split ===")

    images = list_images_in_split(split_name)

    if not images:
        print(f"No images found for split: {split_name}")
        return

    all_meta = []

    for img_path in images:
        dets = detect_price_tags(model, img_path)
        meta = crop_and_save(img_path, dets, split_name)
        all_meta.extend(meta)

    meta_out = CROPS_DIR / "metadata" / f"{split_name}_crops.json"
    with open(meta_out, "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"Saved metadata → {meta_out}")
    print(f"Crops generated: {len(all_meta)}")


def main():
    print("\n=== CROPPING PRICE TAG REGIONS (YOLO detector) ===\n")

    ensure_dirs()

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # runs even if only some splits exist
    for split in ["train", "val", "test"]:
        process_split(model, split)



if __name__ == "__main__":
    main()
