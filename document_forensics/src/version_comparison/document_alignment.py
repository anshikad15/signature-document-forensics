"""
Document alignment via ORB feature matching + RANSAC homography, so two
scans of the same document (different scale/rotation/skew) can be diffed
pixel-by-pixel after warping one onto the other's coordinate frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.utils.image_utils import to_gray


@dataclass
class AlignmentResult:
    success: bool
    aligned_image: Optional[np.ndarray] = None
    homography: Optional[np.ndarray] = None
    inlier_ratio: float = 0.0
    num_matches: int = 0


def align_images(reference: np.ndarray, moving: np.ndarray,
                  max_features: int = 5000, good_match_percent: float = 0.20) -> AlignmentResult:
    ref_gray = to_gray(reference)
    mov_gray = to_gray(moving)

    orb = cv2.ORB_create(max_features)
    kp1, desc1 = orb.detectAndCompute(ref_gray, None)
    kp2, desc2 = orb.detectAndCompute(mov_gray, None)

    if desc1 is None or desc2 is None or len(kp1) < 10 or len(kp2) < 10:
        return AlignmentResult(success=False)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda m: m.distance)

    num_good = max(4, int(len(matches) * good_match_percent))
    good_matches = matches[:num_good]

    if len(good_matches) < 4:
        return AlignmentResult(success=False, num_matches=len(good_matches))

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, inlier_mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransacReprojThreshold=5.0)
    if homography is None:
        return AlignmentResult(success=False, num_matches=len(good_matches))

    inlier_ratio = float(inlier_mask.sum()) / len(inlier_mask) if inlier_mask is not None else 0.0

    h, w = reference.shape[:2]
    aligned = cv2.warpPerspective(moving, homography, (w, h))

    return AlignmentResult(
        success=True,
        aligned_image=aligned,
        homography=homography,
        inlier_ratio=inlier_ratio,
        num_matches=len(good_matches),
    )
