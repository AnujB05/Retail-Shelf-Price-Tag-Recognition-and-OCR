import json
from pathlib import Path

GT_PATH = Path("project/eval/ground_truth_5.json")
PRED_PATH = Path("project/eval/pipeline_outputs/all_predictions.json")

with open(GT_PATH, "r", encoding="utf-8") as f:
    GT = json.load(f)

with open(PRED_PATH, "r", encoding="utf-8") as f:
    PRED = json.load(f)


def normalize_price(x):
    if x is None:
        return None
    return "".join(ch for ch in str(x) if ch.isdigit())


tot_gt = 0
tot_detected = 0
tot_correct = 0

per_image_stats = {}


for image_name, gt_tags in GT.items():

    gt_prices = [normalize_price(t["price"]) for t in gt_tags]
    tot_gt += len(gt_prices)

    pred_tags = PRED.get(image_name, [])
    pred_prices = [normalize_price(t.get("value")) for t in pred_tags]

    tot_detected += len(pred_prices)

    # count matches regardless of order
    correct = sum(p in gt_prices for p in pred_prices)
    tot_correct += correct

    per_image_stats[image_name] = {
        "gt_tags": len(gt_prices),
        "detected": len(pred_prices),
        "correct_prices": correct
    }


metrics = {
    "totals": {
        "gt_tags": tot_gt,
        "detected_tags": tot_detected,
        "correct_prices": tot_correct
    },
    "metrics": {
        "detection_recall": round(tot_detected / tot_gt, 3) if tot_gt else 0,
        "ocr_price_accuracy": round(tot_correct / tot_detected, 3) if tot_detected else 0,
        "end_to_end_accuracy": round(tot_correct / tot_gt, 3) if tot_gt else 0
    },
    "per_image": per_image_stats
}

OUT = Path("project/eval/eval_results.json")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\n=== Evaluation Results ===\n")
print(json.dumps(metrics, indent=2))
print("\nSaved to", OUT)
