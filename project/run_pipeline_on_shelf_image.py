import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from ultralytics import YOLO
import easyocr

from ocr.structure_price_tag import structure_price_tag  # SAME AS EVAL
from ocr.price_postprocess import extract_price          # SAME AS EVAL


MODEL_PATH = Path("project/models/price_tag_yolo/exp_price_tag3/weights/best.pt")

OUTPUT_ROOT = Path("project/final_outputs")
CROPS_DIR = OUTPUT_ROOT / "crops"
VIZ_DIR = OUTPUT_ROOT / "viz"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 640
CONF_THRES = 0.10
NMS_IOU = 0.55

# multi-scale pyramid for tiny tags
SCALES = [0.4, 0.6, 1.0, 1.25, 1.5] 

CROP_EXPAND = 0.20


print("\nLoading YOLO detector…")
detector = YOLO(str(MODEL_PATH))

print("Loading EasyOCR (en + ja)…")
reader_en = easyocr.Reader(["en"], gpu=False)
reader_ja = easyocr.Reader(["ja"], gpu=False)



def load_auto_oriented(path):
    pil = Image.open(path)
    pil = ImageOps.exif_transpose(pil)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def resize_stretch_640(img):
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


def run_dual_ocr(img):
    lines = []

    for bbox, text, conf in reader_en.readtext(img, detail=1):
        lines.append({"text": text, "conf": conf, "bbox": bbox, "lang": "en"})

    for bbox, text, conf in reader_ja.readtext(img, detail=1):
        lines.append({"text": text, "conf": conf, "bbox": bbox, "lang": "ja"})

    return lines


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0: return 0
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (areaA + areaB - inter)


def nms(dets, thr=NMS_IOU):
    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
    out = []
    for d in dets:
        if all(iou(d["bbox"], x["bbox"]) < thr for x in out):
            out.append(d)
    return out

def bbox_height(line):
    try:
            (x1,y1),(x2,y2),(x3,y3),(x4,y4) = line.get("bbox", [(0,0)]*4)
            return max(y1,y2,y3,y4) - min(y1,y2,y3,y4)
    except:
            return 0


def run_pipeline(image_path, return_results=False, eval_mode=False):

    image_path = Path(image_path)
    print(f"\n=== RUNNING PIPELINE ON: {image_path.name} ===")

    orig = load_auto_oriented(image_path)
    oh, ow, _ = orig.shape

    all_detections = []



    print(f"Running multi-scale detection ({SCALES})")

    for scale in SCALES:

        scaled = cv2.resize(orig, None, fx=scale, fy=scale)
        sh, sw, _ = scaled.shape

        resized = resize_stretch_640(scaled)

        det = detector.predict(
            resized,
            conf=CONF_THRES,
            iou=NMS_IOU,
            verbose=False
        )[0]

        if det.boxes is None:
            continue

        sx = (sw / IMG_SIZE) / scale
        sy = (sh / IMG_SIZE) / scale

        for b in det.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0])

            gx1 = int(x1 * sx)
            gy1 = int(y1 * sy)
            gx2 = int(x2 * sx)
            gy2 = int(y2 * sy)

            all_detections.append({
                "bbox": [gx1, gy1, gx2, gy2],
                "conf": float(b.conf[0])
            })

    if not all_detections:
        print("⚠ No detections found")
        return [] if return_results else None

    print(f"✔ Raw detections: {len(all_detections)}")

    merged = nms(all_detections)
    print(f"✔ After merge: {len(merged)} price tags")

    results = []

    for i, det in enumerate(merged):

        x1, y1, x2, y2 = det["bbox"]

        bw = x2 - x1
        bh = y2 - y1

        x1 = max(0, x1 - int(bw * CROP_EXPAND))
        y1 = max(0, y1 - int(bh * CROP_EXPAND))
        x2 = min(ow - 1, x2 + int(bw * CROP_EXPAND))
        y2 = min(oh - 1, y2 + int(bh * CROP_EXPAND))

        crop = orig[y1:y2, x1:x2]

        if crop.shape[0] < 80 or crop.shape[1] < 80:
            crop = cv2.resize(
            crop, None,
            fx=1.8, fy=1.8,
            interpolation=cv2.INTER_CUBIC
            )
        crop_name = f"{image_path.stem}_tag{i}.jpg"
        crop_path = CROPS_DIR / crop_name
        cv2.imwrite(str(crop_path), crop)

        lines = run_dual_ocr(crop)

        # height-based weighting (OCR re-ranking)
        max_h = max([bbox_height(l) for l in lines] or [1])
        for ln in lines:
            ln["height_bonus"] = bbox_height(ln) / max_h

        ocr_result = {
            "lines": lines,
            "variant": "final_multiscale_eval_consistent"
        }

        structured = structure_price_tag(ocr_result, {
            "source_image": image_path.name,
            "crop_name": crop_name
        })

        results.append(structured)

  
        print("\n--- PRICE TAG ---")
        print(structured)


    if not eval_mode:
        out_json = OUTPUT_ROOT / f"{image_path.stem}_final_output.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nCrops saved → {CROPS_DIR}")
        print(f"Output JSON → {out_json}")
        print(f"Viz saved   → {VIZ_DIR}\n")


    if return_results:
        return results

    


if __name__ == "__main__":
    path = input("\nEnter path to shelf image: ")
    run_pipeline(path)
