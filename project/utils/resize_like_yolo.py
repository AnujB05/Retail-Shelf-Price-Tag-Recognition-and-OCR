import cv2
import numpy as np
from pathlib import Path


def letterbox_resize(img, new_size=640, color=(0, 0, 0)):
    """
    Resize image with unchanged aspect ratio using padding (same as YOLO / Roboflow).
    Returns:
        resized_image, scale, padding (pad_x, pad_y)
    """
    h, w = img.shape[:2]

    # scale using min(w,h) to fit inside new_size
    scale = min(new_size / w, new_size / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # resize keeping aspect ratio
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # compute padding
    pad_x = (new_size - new_w) // 2
    pad_y = (new_size - new_h) // 2

    padded = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return padded, scale, (pad_x, pad_y)


def save_letterboxed_image(input_path, output_path, size=640):
    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")

    resized, scale, padding = letterbox_resize(img, new_size=size)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), resized)

    return {
        "output_path": str(output_path),
        "scale": scale,
        "padding": padding
    }
