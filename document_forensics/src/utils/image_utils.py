"""
Shared image / document loading and preprocessing helpers used across
the forensics pipeline (tampering detection, OCR, version comparison).
"""
from __future__ import annotations

import io
import os
from typing import List

import cv2
import numpy as np
from PIL import Image


def load_pages(path: str, dpi: int = 200) -> List[np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _load_pdf_pages(path, dpi=dpi)

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        pil_img = Image.open(path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return [img]


def _load_pdf_pages(path: str, dpi: int = 200) -> List[np.ndarray]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF (fitz) is required to rasterize PDF pages. "
            "Install it with `pip install PyMuPDF`."
        ) from exc

    pages = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            img_bytes = pix.tobytes("png")
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pages.append(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
    return pages


def to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_max_dim(image: np.ndarray, max_dim: int = 1600) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_dim / float(max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def draw_boxes(image: np.ndarray, boxes, color=(0, 0, 255), thickness=2, labels=None) -> np.ndarray:
    out = image.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
        if labels and i < len(labels):
            cv2.putText(out, str(labels[i]), (int(x), max(0, int(y) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def save_image(image: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    cv2.imwrite(path, image)
