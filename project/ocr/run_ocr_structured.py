import json
from pathlib import Path
import easyocr

from ocr_preprocess import generate_ocr_variants
from price_postprocess import extract_price


reader_en = easyocr.Reader(["en"], gpu=False)
reader_ja = easyocr.Reader(["ja"], gpu=False)


def run_easyocr(reader, img):
    results = reader.readtext(img, detail=1)

    lines = []
    for (bbox, text, conf) in results:
        lines.append({
            "text": text.strip(),
            "conf": float(conf),
            "bbox": bbox
        })
    return lines


def run_dual_language_ocr(img):
    lines_en = run_easyocr(reader_en, img)
    lines_ja = run_easyocr(reader_ja, img)

    mixed = []

    for ln in lines_en:
        mixed.append({**ln, "lang": "en"})

    for ln in lines_ja:
        mixed.append({**ln, "lang": "ja"})

    return mixed


def best_variant_ocr(variants):
    best_variant = None
    best_conf = -1
    best_result = None

    for name, img in variants.items():
        lines = run_dual_language_ocr(img)

        if not lines:
            continue

        conf_avg = sum(l["conf"] for l in lines) / len(lines)

        if conf_avg > best_conf:
            best_conf = conf_avg
            best_variant = name
            best_result = lines

    return best_variant, best_result


def ocr_crop_to_lines(crop_path):
    variants = generate_ocr_variants(crop_path)

    if not variants:
        return None

    variant_used, lines = best_variant_ocr(variants)

    return {
        "variant": variant_used,
        "lines": lines
    }
