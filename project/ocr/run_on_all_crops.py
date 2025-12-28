import json
from pathlib import Path

from run_ocr_structured import ocr_crop_to_lines
from structure_price_tag import structure_price_tag


CROPS_META_DIR = Path("project/data/crops/metadata")
OUTPUT_PATH = Path("project/data/results/ocr_structured.json")


def run_split(split_name):
    meta_path = CROPS_META_DIR / f"{split_name}_crops.json"

    if not meta_path.exists():
        return []

    with open(meta_path, "r") as f:
        crops = json.load(f)

    results = []

    for crop_meta in crops:
        crop_path = crop_meta["crop_path"]

        ocr_result = ocr_crop_to_lines(crop_path)

        # skip if OCR completely failed
        if not ocr_result or ocr_result.get("lines") is None:
            continue

        structured = structure_price_tag(ocr_result, crop_meta)
        if structured:
            results.append(structured)


    return results


def main():
    all_results = []

    for split in ["train", "val", "test"]:
        all_results.extend(run_split(split))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nStructured OCR results saved to:\n{OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
