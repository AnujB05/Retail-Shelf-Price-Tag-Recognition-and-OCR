import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from run_pipeline_on_shelf_image import run_pipeline


EVAL_IMAGES_DIR = Path("project/data/raw")
PRED_OUT_DIR = Path("project/eval/predictions")
PRED_OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_eval_pipeline():
    all_outputs = 0

    for img_path in sorted(EVAL_IMAGES_DIR.glob("*.jpg")):
        print(f"\n=== Processing {img_path.name} ===")

        # run pipeline and GET structured output
        structured_tags = run_pipeline(
            img_path,
            return_results=True,   # <-- IMPORTANT
            eval_mode=True
        )

        if not structured_tags:
            print("No structured OCR outputs")
            structured_tags = []

        # --- save JSON properly ---
        out_json = PRED_OUT_DIR / f"{img_path.stem}_pred.json"

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(structured_tags, f, indent=2, ensure_ascii=False)

        print(f"Saved → {out_json}")
        all_outputs += len(structured_tags)

    print(f"Total structured price tags saved: {all_outputs}")


if __name__ == "__main__":
    run_eval_pipeline()
