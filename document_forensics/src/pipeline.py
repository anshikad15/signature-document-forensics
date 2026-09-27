"""
Document Forensics pipeline: combines all tampering/consistency checks
into a single weighted risk score + evidence report, and provides
multi-version comparison producing a semantic change timeline.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List

from src.tampering.copy_move_detection import CopyMoveDetector
from src.tampering.splicing_detection import SplicingDetector
from src.tampering.stamp_seal_detection import analyze as analyze_stamps
from src.tampering.layout_consistency import analyze as analyze_layout
from src.tampering import pdf_metadata_analysis
from src.ocr.ocr_extraction import OCREngine
from src.utils.image_utils import load_pages
from src.version_comparison.document_alignment import align_images
from src.version_comparison.diff_classifier import diff_text, diff_images, diff_pages
from src.version_comparison.change_timeline import build_timeline, to_json

_WEIGHTS = {
    "copy_move": 0.25,
    "splicing": 0.25,
    "layout": 0.20,
    "pdf_metadata": 0.15,
    "stamps": 0.15,
}


@dataclass
class SingleDocReport:
    file_path: str
    risk_score: float
    verdict: str
    module_scores: Dict[str, float] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    ocr_text_preview: str = ""


def _verdict_from_score(score: float) -> str:
    if score < 0.2:
        return "LOW RISK - no significant tampering indicators found"
    if score < 0.5:
        return "MODERATE RISK - some inconsistencies found, manual review recommended"
    return "HIGH RISK - strong tampering indicators found"


def analyze_document(file_path: str, run_ocr: bool = True) -> SingleDocReport:
    pages = load_pages(file_path)
    image = pages[0]

    module_scores: Dict[str, float] = {}
    evidence: List[str] = []

    cm_result = CopyMoveDetector().detect(image)
    module_scores["copy_move"] = cm_result.confidence
    if cm_result.tampered:
        evidence.append(f"Copy-move forgery suspected: {cm_result.num_matches} matched keypoint "
                         f"pairs share a consistent offset (possible duplicated region).")

    sp_result = SplicingDetector().detect(image)
    module_scores["splicing"] = sp_result.confidence
    if sp_result.tampered:
        evidence.append(f"Possible splicing/recompression in {len(sp_result.suspicious_regions)} "
                         f"region(s) with inconsistent ELA/DCT signatures.")

    stamp_result = analyze_stamps(image)
    stamp_score = min(1.0, 0.1 * len(stamp_result.watermarks))
    module_scores["stamps"] = stamp_score
    if stamp_result.stamps:
        evidence.append(f"{len(stamp_result.stamps)} stamp/seal-like region(s) detected.")
    if stamp_result.watermarks:
        evidence.append(f"{len(stamp_result.watermarks)} possible faint watermark region(s) detected.")

    ocr_text = ""
    if run_ocr:
        try:
            ocr_doc = OCREngine().extract(image)
            ocr_text = ocr_doc.full_text
            layout_result = analyze_layout(ocr_doc.words)
            module_scores["layout"] = layout_result.risk_score
            if layout_result.font_outliers:
                evidence.append(f"{len(layout_result.font_outliers)} word(s) with inconsistent font size "
                                 f"relative to the page's dominant font.")
            if layout_result.margin_outliers:
                evidence.append(f"{len(layout_result.margin_outliers)} line(s) misaligned relative to the "
                                 f"document's common left margins.")
        except ImportError as exc:
            evidence.append(f"OCR/layout check skipped: {exc}")
            module_scores["layout"] = 0.0
    else:
        module_scores["layout"] = 0.0

    if file_path.lower().endswith(".pdf"):
        try:
            pdf_result = pdf_metadata_analysis.analyze(file_path)
            module_scores["pdf_metadata"] = pdf_result.risk_score
            evidence.extend(pdf_result.flags)
        except ImportError as exc:
            evidence.append(f"PDF metadata check skipped: {exc}")
            module_scores["pdf_metadata"] = 0.0
    else:
        module_scores["pdf_metadata"] = 0.0

    active_weights = {k: v for k, v in _WEIGHTS.items() if k in module_scores}
    weight_sum = sum(active_weights.values()) or 1.0
    overall = sum(module_scores[k] * w for k, w in active_weights.items()) / weight_sum

    return SingleDocReport(
        file_path=file_path,
        risk_score=round(overall, 3),
        verdict=_verdict_from_score(overall),
        module_scores={k: round(v, 3) for k, v in module_scores.items()},
        evidence=evidence,
        ocr_text_preview=ocr_text[:500],
    )


@dataclass
class VersionComparisonReport:
    version_files: List[str]
    timeline: List[dict] = field(default_factory=list)
    summary: str = ""


def compare_versions(version_paths: List[str], run_ocr: bool = True) -> VersionComparisonReport:
    all_events = []
    version_names = [f"V{i + 1}" for i in range(len(version_paths))]

    ocr_engine = OCREngine() if run_ocr else None
    page_images = [load_pages(p)[0] for p in version_paths]

    for i in range(len(page_images) - 1):
        old_img, new_img = page_images[i], page_images[i + 1]
        old_name, new_name = version_names[i], version_names[i + 1]

        alignment = align_images(old_img, new_img)
        aligned_new = alignment.aligned_image if alignment.success else new_img

        image_changes = diff_images(old_img, aligned_new)

        text_changes = []
        if ocr_engine is not None:
            old_text = ocr_engine.extract(old_img).full_text
            new_text = ocr_engine.extract(new_img).full_text
            text_changes = diff_text(old_text, new_text)

        events = build_timeline(old_name, new_name, text_changes=text_changes,
                                 image_changes=image_changes)
        all_events.extend(events)

    events_json = to_json(all_events)
    summary = "\n".join(f"{e['from_version']} -> {e['to_version']}: {e['description']}"
                         for e in events_json) or "No differences detected between versions."

    return VersionComparisonReport(
        version_files=version_paths,
        timeline=events_json,
        summary=summary,
    )


def save_report(report, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    with open(out_path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
