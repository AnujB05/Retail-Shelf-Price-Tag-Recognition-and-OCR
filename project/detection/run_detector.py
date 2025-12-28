import os
import json
from pathlib import Path

from ultralytics import YOLO
import cv2


PROJECT_ROOT = Path("project")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "detection"


# Pretrained YOLO model (COCO)
MODEL_NAME = "yolov8n.pt"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


def load_split(split_name):
    split_file = SPLITS_DIR / f"{split_name}.txt"
    with open(split_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR / "visualized", exist_ok=True)


def load_model():
    print("\nLoading pretrained YOLO model...")
    model = YOLO(MODEL_NAME)
    print("✔ Model loaded")
    return model


def run_inference_on_image(model, image_path):
    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )

    detections = []
    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id
            })

    return detections


def save_visualization(image_path, detections, out_path):
    img = cv2.imread(str(image_path))

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["confidence"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(
            img,
            f"{conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            2
        )

    cv2.imwrite(str(out_path), img)


def run_split(model, split_name):
    print(f"\n=== Running detection on {split_name} set ===")

    image_files = load_split(split_name)

    results_json = {}

    for fname in image_files:
        image_path = RAW_DIR / fname

        detections = run_inference_on_image(model, image_path)

        results_json[fname] = detections

        # Save visualization
        vis_path = RESULTS_DIR / "visualized" / f"{fname}"
        save_visualization(image_path, detections, vis_path)

    # Save prediction file
    out_json_path = RESULTS_DIR / f"{split_name}_detections.json"
    with open(out_json_path, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"Saved predictions to {out_json_path}")
    print(f"Visualizations stored in {RESULTS_DIR/'visualized'}")


def main():
    print("\n=== YOLO Detection Inference ===\n")

    ensure_dirs()
    model = load_model()

    # Run on all 3 splits for now
    for split in ["train", "val", "test"]:
        run_split(model, split)

    print("\n=== Detection inference complete ===\n")


if __name__ == "__main__":
    main()
