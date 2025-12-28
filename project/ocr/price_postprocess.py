import re


CURRENCY_SYMBOLS = {
    "₹": "INR",
    "Rs": "INR",
    "¥": "JPY",
    "￥": "JPY",
    "円": "JPY"
}


PRICE_PATTERNS = [

    # INR examples → ₹129 / Rs 59 / ₹ 1,299.00
    r"(₹\s*\d[\d,]*(?:\.\d{1,2})?)",
    r"(Rs\.?\s*\d[\d,]*(?:\.\d{1,2})?)",

    # JPY examples → ¥120 / ￥480 / 128円
    r"([¥￥]\s*\d[\d,]*)",
    r"(\d[\d,]*\s*円)",

    # Pure numeric fallback (useful when symbol missing)
    r"(\d[\d,]*(?:\.\d{1,2})?)",
]


def normalize_number(num_str: str):
    """
    Removes commas / spaces and normalizes number format.
    """

    num_str = num_str.replace(",", "").strip()

    # Remove trailing 円
    num_str = re.sub(r"円$", "", num_str)

    return num_str


def detect_currency(text: str):
    for symbol, code in CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code
    return None


def extract_price(text: str):

    if not text or not text.strip():
        return None

    for pattern in PRICE_PATTERNS:
        match = re.search(pattern, text)

        if match:
            matched = match.group(1)
            currency = detect_currency(matched)
            value = normalize_number(matched)

            return {
                "raw_text": text,
                "matched_text": matched,
                "currency": currency,
                "value": value,
            }

    return None

if __name__ == "__main__":

    samples = [
        "¥120",
        "￥ 480",
        "128円",
        "₹ 1,299.00",
        "Rs 59",
        "Price: 199",
        "MRP ₹499 SAVE 20%",
    ]

    for s in samples:
        print(s, "→", extract_price(s))
