"""
Turns raw diffs (text/image/page) into a plain-language chronological
change trail, e.g. "V1 -> V2: Address changed".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .diff_classifier import ImageRegionChange, PageChange, TextChange


@dataclass
class ChangeEvent:
    from_version: str
    to_version: str
    category: str
    description: str
    bbox: tuple = None
    confidence: float = 1.0


def _describe_text_change(tc: TextChange) -> str:
    label = {"numeric": "Numeric value", "date": "Date", "text": "Text"}[tc.token_type]
    if tc.change_type == "added":
        return f'{label} added ("{tc.new_value}")'
    if tc.change_type == "removed":
        return f'{label} removed ("{tc.old_value}")'
    return f'{label} changed ("{tc.old_value}" -> "{tc.new_value}")'


def _describe_image_change(ic: ImageRegionChange) -> str:
    pct = int(ic.change_magnitude * 100)
    label = "Signature" if ic.region_type == "signature" else "Image/graphic region"
    return f"{label} modified near {ic.bbox[:2]} ({pct}% of the region changed)"


def _describe_page_change(pc: PageChange):
    if pc.change_type == "added":
        return f"Page {pc.new_index + 1} added"
    if pc.change_type == "removed":
        return f"Page {pc.old_index + 1} removed"
    if pc.change_type == "reordered":
        return f"Page moved from position {pc.old_index + 1} to {pc.new_index + 1}"
    if pc.change_type == "modified":
        return f"Page {pc.old_index + 1} content modified (similarity {pc.similarity:.0%})"
    return None


def build_timeline(
    from_version: str,
    to_version: str,
    text_changes: List[TextChange] = (),
    image_changes: List[ImageRegionChange] = (),
    page_changes: List[PageChange] = (),
) -> List[ChangeEvent]:
    events: List[ChangeEvent] = []

    for tc in text_changes:
        events.append(ChangeEvent(from_version, to_version, tc.token_type,
                                   _describe_text_change(tc)))

    for ic in image_changes:
        category = "signature" if ic.region_type == "signature" else "image"
        events.append(ChangeEvent(from_version, to_version, category,
                                   _describe_image_change(ic), bbox=ic.bbox,
                                   confidence=ic.change_magnitude))

    for pc in page_changes:
        desc = _describe_page_change(pc)
        if desc:
            events.append(ChangeEvent(from_version, to_version, "page", desc,
                                       confidence=1.0 - pc.similarity if pc.change_type == "modified" else 1.0))

    return events


def render_summary(all_events: List[ChangeEvent]) -> str:
    lines = [f"{e.from_version} -> {e.to_version}: {e.description}" for e in all_events]
    return "\n".join(lines) if lines else "No differences detected between versions."


def to_json(all_events: List[ChangeEvent]) -> List[dict]:
    return [
        {
            "from_version": e.from_version,
            "to_version": e.to_version,
            "category": e.category,
            "description": e.description,
            "bbox": e.bbox,
            "confidence": round(e.confidence, 3),
        }
        for e in all_events
    ]
