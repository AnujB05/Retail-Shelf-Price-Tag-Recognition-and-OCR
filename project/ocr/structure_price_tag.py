from ocr.price_postprocess import extract_price
import re

def is_probably_code(text):
    t = text.strip()

    #very short fragments - ignore
    if len(t) <= 1:
        return True
    # reject only pure numeric/ascii code line
    # allow japanese chars
    if re.fullmatch(r"[0-9\-\s./()A-Za-z]+", t):
        return True
    return False

def pick_price_line(lines):
    candidates = []
    for ln in lines:
        text = ln.get("text","").strip()

        if not text:
            continue
        price = extract_price(text)

        if price:
            candidates.append((ln, price))

    if not candidates:
        return None, None
    #choose highest confidence price
    candidates.sort(key=lambda x: x[0].get("conf",0), reverse = True)

    return candidates[0]

def pick_product_name(lines, price_line):
    remaining = [l for l in lines if l is not price_line]

    candidates = [l for l in remaining
                  if not is_probably_code(l.get("text", ""))
                  ] or remaining
    candidates.sort(key=lambda x: (min(pt[1] for pt in x.get("bbox", [])), -len(x.get("text", "")), -x.get("conf", 0)))

    return candidates[0].get("text")

def collect_extra_info(lines, used_lines):
    extra = []

    for ln in lines:
        if ln in used_lines:
            continue
        t = ln.get("text", "").strip()

        if len(t)<2:
            continue
        extra.append(t)
    return "|".join(extra) if extra else ""

def structure_price_tag(ocr_result, crop_meta):

    if not ocr_result:
        return None
    lines = ocr_result.get("lines")

    if not lines:
        return None
    
    price_line, price_info = pick_price_line(lines)

    product_name = pick_product_name(lines, price_line)

    extra_info = collect_extra_info(lines, used_lines=[price_line] if price_line else [])

    return {"source_image": crop_meta.get("source_image"),
            "crop_name": crop_meta.get("crop_name"),
            "variant_used": ocr_result.get("variant"),
            "product_name": product_name,
            "price": price_info.get("matched_text") if price_info else None,
            "currency": price_info.get("currency") if price_info else None,
            "value": price_info.get("value") if price_info else None,
            "extra_info": extra_info}