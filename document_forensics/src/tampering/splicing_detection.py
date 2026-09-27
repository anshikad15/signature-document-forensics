"""
Splicing / recompression forgery detection: ELA + block-DCT energy analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import io

from src.utils.image_utils import to_gray


@dataclass
class SplicingResult:
    tampered: bool
    confidence: float
    suspicious_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)
    ela_map: np.ndarray = None
    annotated_image: np.ndarray = None


def error_level_analysis(image: np.ndarray, quality: int = 90) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    buf = io.BytesIO()
    pil_img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = np.array(Image.open(buf).convert("RGB"))

    diff = cv2.absdiff(rgb, resaved).astype(np.float32)
    diff_gray = diff.mean(axis=2)

    max_val = diff_gray.max()
    if max_val < 1e-6:
        return np.zeros_like(diff_gray, dtype=np.uint8)
    scale = 255.0 / max_val
    return np.clip(diff_gray * scale, 0, 255).astype(np.uint8)


def block_dct_energy_map(image: np.ndarray, block_size: int = 8) -> np.ndarray:
    gray = to_gray(image).astype(np.float32)
    h, w = gray.shape
    h_blocks, w_blocks = h // block_size, w // block_size
    energy = np.zeros((h_blocks, w_blocks), dtype=np.float32)

    for by in range(h_blocks):
        for bx in range(w_blocks):
            block = gray[by * block_size:(by + 1) * block_size,
                         bx * block_size:(bx + 1) * block_size]
            dct = cv2.dct(block - block.mean())
            hf = dct.copy()
            hf[:3, :3] = 0
            energy[by, bx] = np.sum(np.abs(hf))

    energy_img = cv2.resize(energy, (w, h), interpolation=cv2.INTER_NEAREST)
    energy_img = cv2.normalize(energy_img, None, 0, 255, cv2.NORM_MINMAX)
    return energy_img.astype(np.uint8)


class SplicingDetector:
    def __init__(
        self,
        ela_quality: int = 90,
        ela_threshold: int = 55,
        dct_zscore_threshold: float = 2.5,
        min_region_area: int = 400,
    ):
        self.ela_quality = ela_quality
        self.ela_threshold = ela_threshold
        self.dct_zscore_threshold = dct_zscore_threshold
        self.min_region_area = min_region_area

    def detect(self, image: np.ndarray) -> SplicingResult:
        ela_map = error_level_analysis(image, quality=self.ela_quality)
        dct_map = block_dct_energy_map(image)

        blurred = cv2.GaussianBlur(dct_map.astype(np.float32), (51, 51), 0)
        local_std = cv2.GaussianBlur((dct_map.astype(np.float32) - blurred) ** 2, (51, 51), 0) ** 0.5
        local_std = np.clip(local_std, 1e-3, None)
        dct_z = np.abs(dct_map.astype(np.float32) - blurred) / local_std
        dct_suspicious = (dct_z > self.dct_zscore_threshold).astype(np.uint8) * 255

        ela_suspicious = (ela_map > self.ela_threshold).astype(np.uint8) * 255

        combined = cv2.bitwise_and(ela_suspicious, dct_suspicious)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.min_region_area:
                x, y, w, h = cv2.boundingRect(c)
                regions.append((x, y, w, h))

        tampered = len(regions) > 0
        total_area = sum(w * h for _, _, w, h in regions)
        page_area = image.shape[0] * image.shape[1]
        confidence = min(1.0, (total_area / page_area) * 12) if tampered else 0.0

        annotated = image.copy()
        for (x, y, w, h) in regions:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return SplicingResult(
            tampered=tampered,
            confidence=confidence,
            suspicious_regions=regions,
            ela_map=ela_map,
            annotated_image=annotated,
        )
