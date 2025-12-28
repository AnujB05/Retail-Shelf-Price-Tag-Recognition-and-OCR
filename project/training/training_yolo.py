from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path("project")
DATA_YAML = PROJECT_ROOT / "config" / "price_tag.yaml"
RESULTS_DIR = PROJECT_ROOT / "models" / "price_tag_yolo"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("\n=== Starting YOLO Price Tag Training ===\n")

model = YOLO("yolov8n.pt")  # small model for small dataset


model.train(
    data=str(DATA_YAML),
    project=str(RESULTS_DIR),
    name="exp_price_tag",
    epochs=60,
    batch=8,
    imgsz=640,

    # Let YOLO automatically create validation split
    val=True,
    fraction=0.2,

    # On-the-fly augmentations = effective 800–1500 samples
    hsv_h=0.02,
    hsv_s=0.5,
    hsv_v=0.5,
    degrees=5,
    translate=0.05,
    scale=0.1,
    shear=2,
    perspective=0.0005,
    flipud=0.0,
    fliplr=0.2,
    mosaic=1.0,
    mixup=0.1,
)

print("\n=== Training complete ===\n")
print(f"Model weights saved under:\n{RESULTS_DIR}\n")
