"""
Copy-move forgery detection using SIFT self-matching + offset clustering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

from src.utils.image_utils import to_gray


@dataclass
class CopyMoveResult:
    tampered: bool
    confidence: float
    num_matches: int
    source_points: List[Tuple[int, int]] = field(default_factory=list)
    target_points: List[Tuple[int, int]] = field(default_factory=list)
    annotated_image: np.ndarray = None


class CopyMoveDetector:
    def __init__(
        self,
        ratio_thresh: float = 0.5,
        min_distance_px: int = 40,
        min_cluster_size: int = 8,
        offset_tolerance_px: float = 8.0,
    ):
        self.ratio_thresh = ratio_thresh
        self.min_distance_px = min_distance_px
        self.min_cluster_size = min_cluster_size
        self.offset_tolerance_px = offset_tolerance_px
        self.sift = cv2.SIFT_create(nfeatures=4000)

    def detect(self, image: np.ndarray) -> CopyMoveResult:
        gray = to_gray(image)
        kp, desc = self.sift.detectAndCompute(gray, None)
        if desc is None or len(kp) < 10:
            return CopyMoveResult(False, 0.0, 0)

        matcher = cv2.BFMatcher(cv2.NORM_L2)
        raw_matches = matcher.knnMatch(desc, desc, k=3)

        good_matches = []
        for m_group in raw_matches:
            if len(m_group) < 3:
                continue
            m, n = m_group[1], m_group[2]
            if m.distance < self.ratio_thresh * n.distance:
                pt1 = np.array(kp[m.queryIdx].pt)
                pt2 = np.array(kp[m.trainIdx].pt)
                if np.linalg.norm(pt1 - pt2) > self.min_distance_px:
                    good_matches.append((pt1, pt2))

        if len(good_matches) < self.min_cluster_size:
            return CopyMoveResult(False, 0.0, len(good_matches))

        cluster = self._largest_offset_cluster(good_matches)
        tampered = len(cluster) >= self.min_cluster_size
        confidence = min(1.0, len(cluster) / 40.0) if tampered else 0.0

        src_pts = [tuple(map(int, p1)) for p1, _ in cluster]
        dst_pts = [tuple(map(int, p2)) for _, p2 in cluster]

        annotated = image.copy()
        for (x1, y1), (x2, y2) in zip(src_pts, dst_pts):
            cv2.circle(annotated, (x1, y1), 4, (0, 0, 255), -1)
            cv2.circle(annotated, (x2, y2), 4, (0, 255, 255), -1)
            cv2.line(annotated, (x1, y1), (x2, y2), (0, 165, 255), 1)

        return CopyMoveResult(
            tampered=tampered,
            confidence=confidence,
            num_matches=len(cluster),
            source_points=src_pts,
            target_points=dst_pts,
            annotated_image=annotated,
        )

    def _largest_offset_cluster(self, matches):
        offsets = [tuple(p2 - p1) for p1, p2 in matches]
        offsets_arr = np.array(offsets)

        used = np.zeros(len(offsets), dtype=bool)
        best_cluster_idx = []

        for i in range(len(offsets_arr)):
            if used[i]:
                continue
            dists = np.linalg.norm(offsets_arr - offsets_arr[i], axis=1)
            cluster_idx = np.where(dists < self.offset_tolerance_px)[0]
            if len(cluster_idx) > len(best_cluster_idx):
                best_cluster_idx = cluster_idx

        return [matches[i] for i in best_cluster_idx]
