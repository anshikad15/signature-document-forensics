"""
PDF metadata & structure analysis: creation/mod date mismatches,
creator/producer tool mismatches, and incremental-save (%%EOF) counting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None


@dataclass
class PDFMetadataResult:
    metadata: Dict[str, Optional[str]] = field(default_factory=dict)
    num_pages: int = 0
    num_incremental_saves: int = 1
    flags: List[str] = field(default_factory=list)
    risk_score: float = 0.0


def _read_raw_eof_count(pdf_path: str) -> int:
    with open(pdf_path, "rb") as f:
        data = f.read()
    return max(1, data.count(b"%%EOF"))


def analyze(pdf_path: str) -> PDFMetadataResult:
    if PdfReader is None:
        raise ImportError("PyPDF2 is required. Install it with `pip install PyPDF2`.")

    reader = PdfReader(pdf_path)
    info = reader.metadata or {}

    meta = {
        "author": info.get("/Author"),
        "creator": info.get("/Creator"),
        "producer": info.get("/Producer"),
        "creation_date": info.get("/CreationDate"),
        "mod_date": info.get("/ModDate"),
        "title": info.get("/Title"),
    }

    flags: List[str] = []

    if meta["creation_date"] and meta["mod_date"] and meta["creation_date"] != meta["mod_date"]:
        flags.append(
            f"Modification date ({meta['mod_date']}) differs from creation date "
            f"({meta['creation_date']}) — document was edited after initial creation."
        )

    if meta["creator"] and meta["producer"] and meta["creator"] != meta["producer"]:
        flags.append(
            f"Creator ('{meta['creator']}') and Producer ('{meta['producer']}') tools differ — "
            "the PDF was likely opened in a different application after creation "
            "(possible editing/annotation pass)."
        )

    eof_count = _read_raw_eof_count(pdf_path)
    if eof_count > 1:
        flags.append(
            f"File contains {eof_count} '%%EOF' markers, indicating {eof_count - 1} "
            "incremental save(s) after the original — content may have been appended "
            "or overwritten post-creation."
        )

    num_pages = len(reader.pages)
    risk_score = min(1.0, 0.25 * len(flags) + 0.1 * max(0, eof_count - 1))

    return PDFMetadataResult(
        metadata=meta,
        num_pages=num_pages,
        num_incremental_saves=eof_count,
        flags=flags,
        risk_score=risk_score,
    )
