import json
import csv
from pathlib import Path

CROPS_META_DIR = Path("project/data/crops/metadata")
OCR_RESULTS_PATH = Path("project/data/results/ocr_structured.json")

OUT_CSV = Path("project/data/eval/price_ground_truth_template.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def load_all_crops():
    crops = []

    for split in ["train", "val", "test"]:
        path = CROPS_META_DIR / f"{split}_crops.json"
        if not path.exists():
            continue

        with open(path, "r") as f:
            crops.extend(json.load(f))

    return {c["crop_name"]: c for c in crops}


def load_ocr_results():
    if not OCR_RESULTS_PATH.exists():
        print("⚠ OCR results not found — run OCR pipeline first.")
        return {}

    with open(OCR_RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {r["crop_name"]: r for r in data}


def main():
    crops = load_all_crops()
    ocr = load_ocr_results()

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "source_image",
            "crop_name",
            "predicted_price",
            "gt_price",             # <-- YOU fill this column
            "notes"
        ])

        for crop_name, meta in crops.items():
            pred_price = None

            if crop_name in ocr:
                pred_price = ocr[crop_name].get("value")

            writer.writerow([
                meta["source_image"],
                crop_name,
                pred_price,
                "",        # ground-truth price to fill manually
                ""
            ])

    print(f"\n✔ Ground-truth template written to:\n{OUT_CSV}\n")
    print("Fill in the 'gt_price' column where valid price is visible.\n")


if __name__ == "__main__":
    main()
