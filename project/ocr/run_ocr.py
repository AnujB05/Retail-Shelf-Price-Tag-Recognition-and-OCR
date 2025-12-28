import easyocr
from ocr_preprocess import generate_ocr_variants

print("\nLoading OCR models...")
reader_en = easyocr.Reader(["en"], gpu=False)
reader_ja = easyocr.Reader(["ja"], gpu=False)


def run_easyocr(reader, image):
    results = reader.readtext(image, detail=1)
    if not results:
        return "", 0.0
    best = max(results, key=lambda x: x[2])
    return best[1], float(best[2])


def run_ocr_with_variants(crop_path):
    variants = generate_ocr_variants(crop_path)

    best_text = ""
    best_conf = -1
    best_variant = None
    best_lang = None

    for variant_name, img in variants.items():

        text_en, conf_en = run_easyocr(reader_en, img)

        if any(c.isdigit() for c in text_en):
            if conf_en > best_conf:
                best_conf = conf_en
                best_text = text_en
                best_variant = variant_name
                best_lang = "en"
            continue

        # Otherwise try Japanese
        text_ja, conf_ja = run_easyocr(reader_ja, img)

        if conf_ja > best_conf:
            best_conf = conf_ja
            best_text = text_ja
            best_variant = variant_name
            best_lang = "ja"

    return {
        "crop_path": str(crop_path),
        "text": best_text,
        "confidence": best_conf,
        "variant_used": best_variant,
        "lang": best_lang,
    }


def main():
    print("\n=== OCR TEST UTILITY (EasyOCR) ===\n")

    crop_path = input("Enter crop image path: ").strip()

    result = run_ocr_with_variants(crop_path)

    print("\nOCR RESULT")
    print("------------------------")
    print(f"Variant Used : {result['variant_used']}")
    print(f"Text         : {result['text']}")
    print(f"Confidence   : {result['confidence']:.3f}")


if __name__ == "__main__":
    main()
