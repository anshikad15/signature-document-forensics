"""
Stamp / seal / watermark detection via HSV color thresholding, contour
shape analysis, and low-contrast structured-region detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class StampSealResult:
    stamps: List[Dict] = field(default_factory=list)
    watermarks: List[Dict] = field(default_factory=list)
    annotated_image: np.ndarray = None


_HSV_INK_RANGES = {
    "red": [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))],
    "blue": [((100, 70, 50), (130, 255, 255))],
    "purple": [((130, 40, 50), (160, 255, 255))],
}


def _color_mask(hsv: np.ndarray) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for ranges in _HSV_INK_RANGES.values():
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return mask


def detect_stamps(image: np.ndarray, min_area: int = 800, max_area_ratio: float = 0.05) -> List[Dict]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = _color_mask(hsv)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    page_area = image.shape[0] * image.shape[1]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stamps = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > page_area * max_area_ratio:
            continue
        x, y, w, h = cv2.boundingRect(c)
        perimeter = cv2.arcLength(c, True)
        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
        shape = "circular/oval" if circularity > 0.6 else "irregular/rectangular"

        region_hsv = hsv[y:y + h, x:x + w]
        color_name = _dominant_color(region_hsv)

        stamps.append({
            "bbox": (x, y, w, h),
            "area": int(area),
            "circularity": round(float(circularity), 3),
            "shape": shape,
            "color": color_name,
        })
    return stamps


def _dominant_color(region_hsv: np.ndarray) -> str:
    best_name, best_count = "unknown", 0
    for name, ranges in _HSV_INK_RANGES.items():
        count = 0
        for lo, hi in ranges:
            count += int(cv2.countNonZero(cv2.inRange(region_hsv, np.array(lo), np.array(hi))))
        if count > best_count:
            best_name, best_count = name, count
    return best_name


def detect_watermarks(image: np.ndarray, block_size: int = 64, contrast_thresh: float = 8.0) -> List[Dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape
    watermarks = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y:y + block_size, x:x + block_size]
            std = float(block.std())
            edges = cv2.Canny(block, 30, 90)
            edge_density = float(np.count_nonzero(edges)) / edges.size
            if 2.0 < std < contrast_thresh and edge_density > 0.03:
                watermarks.append({"bbox": (x, y, block_size, block_size),
                                    "contrast_std": round(std, 2),
                                    "edge_density": round(edge_density, 4)})
    return _merge_adjacent_blocks(watermarks)


def _merge_adjacent_blocks(blocks: List[Dict]) -> List[Dict]:
    if not blocks:
        return []
    mask_points = [(b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3]) for b in blocks]
    max_x = max(b[0] + b[2] for b in mask_points)
    max_y = max(b[1] + b[3] for b in mask_points)

    canvas = np.zeros((max_y, max_x), dtype=np.uint8)
    for x, y, w, h in mask_points:
        canvas[y:y + h, x:x + w] = 255
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    merged = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= 2 * 64 * 64:
            merged.append({"bbox": (x, y, w, h)})
    return merged


def analyze(image: np.ndarray) -> StampSealResult:
    stamps = detect_stamps(image)
    watermarks = detect_watermarks(image)

    annotated = image.copy()
    for s in stamps:
        x, y, w, h = s["bbox"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 128, 255), 2)
        cv2.putText(annotated, f"{s['color']} stamp", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1, cv2.LINE_AA)
    for wm in watermarks:
        x, y, w, h = wm["bbox"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 200, 0), 1)

    return StampSealResult(stamps=stamps, watermarks=watermarks, annotated_image=annotated)
