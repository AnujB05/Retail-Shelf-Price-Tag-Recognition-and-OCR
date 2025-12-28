import os
import json
from pathlib import Path

from pipeline.crop_price_tags import is_likely_price_tag  # reuse filtering
from ocr.run_ocr import run_ocr_with_variants
from ocr.price_postprocess import extract_price


PROJECT_ROOT = Path("project")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
DETECTION_RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "detection"
OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "end_to_end"


CONF_THRESHOLD = 0.25  # same as cropping stage


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_detections(split_name):
    det_path = DETECTION_RESULTS_DIR / f"{split_name}_detections.json"

    if not det_path.exists():
        raise FileNotFoundError(
            f"Detection file not found: {det_path}\n"
            "Run detection/run_detector.py first."
        )

    with open(det_path, "r") as f:
        return json.load(f)


def run_pipeline_on_image(image_name, detections):
    """
    Runs full pipeline on a single image:
    detection → crop → OCR → price parsing
    """

    results = []

    for det in detections:

        conf = det["confidence"]
        x1, y1, x2, y2 = map(int, det["bbox"])

        if conf < CONF_THRESHOLD:
            continue

        # OCR pipeline operates on crop path — but we already created crops earlier
        # So instead we reuse crop metadata later; for now assume bbox-level inference

        ocr_out = run_ocr_with_variants(  # expects crop image path
            det.get("crop_path", None)
        )

        price = extract_price(ocr_out["text"])

        results.append({
            "bbox": det["bbox"],
            "confidence": conf,
            "ocr_text": ocr_out["text"],
            "ocr_conf": ocr_out["confidence"],
            "variant_used": ocr_out["variant_used"],
            "lang": ocr_out.get("lang"),
            "price": price
        })

    return results


def process_split(split_name):
    print(f"\n=== Running end-to-end pipeline on {split_name} split ===")

    detections = load_detections(split_name)

    image_level_results = {}

    for image_name, dets in detections.items():
        image_results = run_pipeline_on_image(image_name, dets)
        image_level_results[image_name] = image_results

    out_path = OUTPUT_DIR / f"{split_name}_pipeline_output.json"

    with open(out_path, "w") as f:
        json.dump(image_level_results, f, indent=2, ensure_ascii=False)

    print(f"✔ Saved: {out_path}")


def main():
    print("\n=== END-TO-END PRICE PIPELINE ===\n")

    ensure_dirs()

    for split in ["train", "val", "test"]:
        process_split(split)

    print("\n=== Pipeline execution complete ===\n")


if __name__ == "__main__":
    main()
