"""
Classifies differences between two document versions: text-level (via
difflib + numeric/date typing), image-level (pixel diff + signature vs
generic-image heuristic), and page-level (perceptual hashing).
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from src.ocr.ocr_extraction import classify_token

try:
    import imagehash
except ImportError:
    imagehash = None


@dataclass
class TextChange:
    change_type: str
    token_type: str
    old_value: str
    new_value: str


@dataclass
class ImageRegionChange:
    bbox: Tuple[int, int, int, int]
    region_type: str
    change_magnitude: float


@dataclass
class PageChange:
    change_type: str
    old_index: int = -1
    new_index: int = -1
    similarity: float = 1.0


def diff_text(text_old: str, text_new: str) -> List[TextChange]:
    tokens_old = text_old.split()
    tokens_new = text_new.split()
    matcher = difflib.SequenceMatcher(a=tokens_old, b=tokens_new)

    changes: List[TextChange] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        old_val = " ".join(tokens_old[i1:i2])
        new_val = " ".join(tokens_new[j1:j2])
        token_type = classify_token(new_val or old_val)
        if tag == "insert":
            changes.append(TextChange("added", token_type, "", new_val))
        elif tag == "delete":
            changes.append(TextChange("removed", token_type, old_val, ""))
        elif tag == "replace":
            changes.append(TextChange("changed", token_type, old_val, new_val))
    return changes


def diff_images(aligned_old: np.ndarray, aligned_new: np.ndarray,
                 diff_threshold: int = 30, min_area: int = 250,
                 signature_zone_y_frac: float = 0.75) -> List[ImageRegionChange]:
    gray_old = cv2.cvtColor(aligned_old, cv2.COLOR_BGR2GRAY) if len(aligned_old.shape) == 3 else aligned_old
    gray_new = cv2.cvtColor(aligned_new, cv2.COLOR_BGR2GRAY) if len(aligned_new.shape) == 3 else aligned_new

    diff = cv2.absdiff(gray_old, gray_new)
    _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    changes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(bh) if bh else 0
        in_signature_zone = (y / h) > signature_zone_y_frac
        is_sig_shape = 1.5 < aspect < 8.0 and area < 0.03 * h * w
        region_type = "signature" if (in_signature_zone and is_sig_shape) else "image"

        region_mask = mask[y:y + bh, x:x + bw]
        magnitude = float(np.count_nonzero(region_mask)) / region_mask.size

        changes.append(ImageRegionChange(bbox=(x, y, bw, bh), region_type=region_type,
                                          change_magnitude=round(magnitude, 3)))
    return changes


def diff_pages(pages_old: List[np.ndarray], pages_new: List[np.ndarray],
                similarity_threshold: float = 0.90) -> List[PageChange]:
    if imagehash is None:
        raise ImportError("imagehash is required. Install it with `pip install imagehash`.")

    def to_pil(img: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    hashes_old = [imagehash.phash(to_pil(p)) for p in pages_old]
    hashes_new = [imagehash.phash(to_pil(p)) for p in pages_new]

    max_hash_dist = 64
    matched_new = set()
    changes: List[PageChange] = []

    for i, h_old in enumerate(hashes_old):
        best_j, best_sim = -1, -1.0
        for j, h_new in enumerate(hashes_new):
            if j in matched_new:
                continue
            sim = 1.0 - (h_old - h_new) / max_hash_dist
            if sim > best_sim:
                best_sim, best_j = sim, j
        if best_sim >= similarity_threshold:
            matched_new.add(best_j)
            change_type = "unchanged" if best_sim > 0.99 else "modified"
            if best_j != i:
                change_type = "reordered"
            changes.append(PageChange(change_type, old_index=i, new_index=best_j, similarity=round(best_sim, 3)))
        else:
            changes.append(PageChange("removed", old_index=i, similarity=round(max(best_sim, 0), 3)))

    for j in range(len(hashes_new)):
        if j not in matched_new:
            changes.append(PageChange("added", new_index=j))

    return changes
