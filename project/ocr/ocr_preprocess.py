import cv2
import numpy as np


def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def unsharp_mask(image):
    blur = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, blur, -0.5, 0)


def adaptive_thresh(gray):
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 8
    )


def normalize_lighting(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def generate_ocr_variants(image_path):
    img = cv2.imread(str(image_path))

    if img is None:
        return {}

    img = normalize_lighting(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    variants = {
        "orig": gray,
        "clahe": apply_clahe(gray),
        "sharp": unsharp_mask(gray),
        "thr": adaptive_thresh(gray)
    }

    return variants
