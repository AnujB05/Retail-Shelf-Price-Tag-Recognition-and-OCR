import json
import csv
from pathlib import Path
from collections import defaultdict

OCR_RESULTS_PATH = Path("project/data/results/ocr_structured.json")
GT_CSV = Path("project/data/eval/price_ground_truth_template.csv")

OUT_REPORT = Path("project/data/eval/eval_summary.json")


def load_ocr_results():
    with open(OCR_RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {r["crop_name"]: r for r in data}


def load_ground_truth():
    gt = {}

    with open(GT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            crop = row["crop_name"]
            gt_price = row["gt_price"].strip()

            if gt_price == "":
                continue

            gt[crop] = {
                "source_image": row["source_image"],
                "gt_price": gt_price
            }

    return gt


def normalize_price(v):
    if v is None:
        return None

    v = str(v).strip()

    # strip currency, commas, leading zeros
    v = v.replace("¥", "").replace("₹", "").replace("Rs", "")
    v = v.replace(",", "")

    if v.startswith("0") and v != "0":
        v = v.lstrip("0")

    return v


def main():
    ocr = load_ocr_results()
    gt = load_ground_truth()

    per_image_stats = defaultdict(lambda: {
        "gt_tags": 0,
        "detected_tags": 0,
        "ocr_correct": 0
    })

    total_gt = len(gt)
    detected = 0
    ocr_correct = 0

    errors = []

    for crop, entry in gt.items():
        img = entry["source_image"]
        per_image_stats[img]["gt_tags"] += 1

        gt_price = normalize_price(entry["gt_price"])

        if crop not in ocr:
            # missed detection (no OCR because no crop or lost crop)
            errors.append({
                "crop_name": crop,
                "source_image": img,
                "type": "missed_detection"
            })
            continue

        per_image_stats[img]["detected_tags"] += 1
        detected += 1

        pred_price = normalize_price(ocr[crop].get("value"))

        if pred_price == gt_price:
            per_image_stats[img]["ocr_correct"] += 1
            ocr_correct += 1
        else:
            errors.append({
                "crop_name": crop,
                "source_image": img,
                "gt_price": gt_price,
                "predicted": pred_price,
                "type": "ocr_mismatch"
            })

    detection_coverage = detected / total_gt if total_gt else 0
    ocr_accuracy = ocr_correct / detected if detected else 0
    end_to_end_accuracy = ocr_correct / total_gt if total_gt else 0

    summary = {
        "totals": {
            "gt_price_tags": total_gt,
            "detected_price_tags": detected,
            "ocr_correct": ocr_correct
        },
        "metrics": {
            "detection_coverage": round(detection_coverage, 3),
            "ocr_price_accuracy": round(ocr_accuracy, 3),
            "end_to_end_accuracy": round(end_to_end_accuracy, 3)
        },
        "per_image": per_image_stats,
        "errors": errors
    }

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nEvaluation complete")
    print(f"Report saved to:\n{OUT_REPORT}\n")
    print("Key metrics:")
    print(f"- Detection Coverage  = {summary['metrics']['detection_coverage']}")
    print(f"- OCR Price Accuracy  = {summary['metrics']['ocr_price_accuracy']}")
    print(f"- End-to-End Accuracy = {summary['metrics']['end_to_end_accuracy']}\n")


if __name__ == "__main__":
    main()
