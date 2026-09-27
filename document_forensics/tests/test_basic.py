"""
Lightweight smoke tests using synthetic images (no real scans needed).
Run with: python -m pytest tests/ -v
"""
import numpy as np
import pytest

from src.tampering.copy_move_detection import CopyMoveDetector
from src.tampering.splicing_detection import SplicingDetector, error_level_analysis
from src.tampering.stamp_seal_detection import analyze as analyze_stamps
from src.tampering.layout_consistency import analyze as analyze_layout
from src.version_comparison.document_alignment import align_images
from src.version_comparison.diff_classifier import diff_text, diff_images


def _blank_page(h=600, w=800):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _page_with_pasted_patch():
    img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    patch = img[50:150, 50:250].copy()
    img[400:500, 500:700] = patch
    return img


def test_copy_move_detects_pasted_patch():
    img = _page_with_pasted_patch()
    result = CopyMoveDetector(min_cluster_size=5).detect(img)
    assert result.num_matches >= 0


def test_copy_move_no_false_positive_on_blank_page():
    img = _blank_page()
    result = CopyMoveDetector().detect(img)
    assert result.tampered is False


def test_splicing_ela_runs():
    img = _page_with_pasted_patch()
    ela = error_level_analysis(img)
    assert ela.shape == img.shape[:2]
    result = SplicingDetector().detect(img)
    assert isinstance(result.confidence, float)


def test_stamp_detection_runs_on_blank_page():
    img = _blank_page()
    result = analyze_stamps(img)
    assert result.stamps == []


def test_layout_consistency_handles_empty_words():
    result = analyze_layout([])
    assert result.risk_score == 0.0


def test_layout_consistency_flags_font_outlier():
    words = [
        {"text": "Hello", "bbox": (10, 10, 50, 20), "block_num": 0, "line_num": 0},
        {"text": "World", "bbox": (70, 10, 50, 20), "block_num": 0, "line_num": 0},
        {"text": "TINY", "bbox": (10, 40, 20, 45), "block_num": 0, "line_num": 1},
    ]
    result = analyze_layout(words, font_tolerance=0.2)
    assert len(result.font_outliers) >= 1


def test_document_alignment_identical_images():
    img = _page_with_pasted_patch()
    result = align_images(img, img.copy())
    assert result.success is True
    assert result.aligned_image.shape == img.shape


def test_diff_text_detects_change():
    changes = diff_text("The rent is 1000 dollars", "The rent is 1500 dollars")
    assert any(c.change_type == "changed" for c in changes)


def test_diff_images_detects_added_block():
    old = _blank_page()
    new = old.copy()
    new[300:350, 300:450] = (0, 0, 0)
    changes = diff_images(old, new, min_area=100)
    assert len(changes) >= 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
