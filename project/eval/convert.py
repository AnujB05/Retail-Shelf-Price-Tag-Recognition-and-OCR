import json
from pathlib import Path

PRED_DIR = Path("project/eval/predictions")
OUT = Path("project/eval/pipeline_outputs/all_predictions.json")

grouped = {}

for pred_file in PRED_DIR.glob("*_pred.json"):

    preds = json.load(open(pred_file, encoding="utf-8"))

    for tag in preds:
        if tag is None:
            continue

        img = tag["source_image"]

        grouped.setdefault(img, [])
        grouped[img].append(tag)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(grouped, f, indent=2, ensure_ascii=False)

print("grouped predictions written to", OUT)
