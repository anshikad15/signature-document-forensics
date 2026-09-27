"""
Document consistency checking: font-size, margin/alignment, and
line-spacing outliers, computed from OCR word boxes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class ConsistencyResult:
    dominant_font_height: float = 0.0
    common_margins: List[float] = field(default_factory=list)
    font_outliers: List[Dict] = field(default_factory=list)
    margin_outliers: List[Dict] = field(default_factory=list)
    spacing_outliers: List[Dict] = field(default_factory=list)
    risk_score: float = 0.0


def _group_by_line(words: List[Dict]) -> List[List[Dict]]:
    lines: Dict[tuple, List[Dict]] = {}
    for w in words:
        key = (w.get("block_num", 0), w.get("line_num", 0))
        lines.setdefault(key, []).append(w)
    return [sorted(v, key=lambda w: w["bbox"][0]) for v in lines.values()]


def analyze(words: List[Dict], font_tolerance: float = 0.28, margin_tolerance_px: int = 12) -> ConsistencyResult:
    if not words:
        return ConsistencyResult()

    heights = np.array([w["bbox"][3] for w in words if w["bbox"][3] > 0])
    dominant_height = float(np.median(heights)) if len(heights) else 0.0

    font_outliers = []
    for w in words:
        h = w["bbox"][3]
        if dominant_height > 0 and abs(h - dominant_height) / dominant_height > font_tolerance:
            font_outliers.append({"text": w["text"], "bbox": w["bbox"], "height": h,
                                   "expected_height": dominant_height})

    lines = _group_by_line(words)
    left_edges = np.array([line[0]["bbox"][0] for line in lines if line])

    common_margins: List[float] = []
    if len(left_edges):
        sorted_edges = np.sort(left_edges)
        cluster = [sorted_edges[0]]
        for edge in sorted_edges[1:]:
            if edge - cluster[-1] <= margin_tolerance_px:
                cluster.append(edge)
            else:
                if len(cluster) >= 2:
                    common_margins.append(float(np.mean(cluster)))
                cluster = [edge]
        if len(cluster) >= 2:
            common_margins.append(float(np.mean(cluster)))

    margin_outliers = []
    for line in lines:
        if not line:
            continue
        left = line[0]["bbox"][0]
        if common_margins and min(abs(left - m) for m in common_margins) > margin_tolerance_px * 2:
            margin_outliers.append({
                "text": " ".join(w["text"] for w in line),
                "left_edge": left,
                "nearest_margin": min(common_margins, key=lambda m: abs(left - m)),
            })

    line_tops = sorted(set(line[0]["bbox"][1] for line in lines if line))
    gaps = np.diff(line_tops) if len(line_tops) > 2 else np.array([])
    spacing_outliers = []
    if len(gaps) > 2:
        median_gap = np.median(gaps)
        mad = np.median(np.abs(gaps - median_gap)) + 1e-6
        for i, g in enumerate(gaps):
            if abs(g - median_gap) / mad > 4:
                spacing_outliers.append({"after_y": int(line_tops[i]), "gap_px": int(g),
                                          "expected_gap_px": int(median_gap)})

    n_issues = len(font_outliers) + len(margin_outliers) + len(spacing_outliers)
    risk_score = min(1.0, n_issues / max(10, len(words) * 0.05))

    return ConsistencyResult(
        dominant_font_height=dominant_height,
        common_margins=common_margins,
        font_outliers=font_outliers,
        margin_outliers=margin_outliers,
        spacing_outliers=spacing_outliers,
        risk_score=risk_score,
    )
