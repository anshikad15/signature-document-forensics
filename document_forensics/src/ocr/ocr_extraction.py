"""
OCR + layout analysis + key-value field extraction, wrapping pytesseract.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None


_FIELD_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z \-_/]{1,30}?)\s*[:\-]\s*(.+?)\s*$")
_NUMERIC_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_DATE_RE = re.compile(r"\b\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}\b")


@dataclass
class OCRDocument:
    full_text: str
    words: List[Dict] = field(default_factory=list)
    fields: Dict[str, str] = field(default_factory=dict)
    dates: List[str] = field(default_factory=list)


class OCREngine:
    def __init__(self, lang: str = "eng", min_confidence: int = 40):
        if pytesseract is None:
            raise ImportError(
                "pytesseract is required. Install it with `pip install pytesseract` "
                "and make sure the Tesseract binary is on your PATH."
            )
        self.lang = lang
        self.min_confidence = min_confidence

    def extract(self, image: np.ndarray) -> OCRDocument:
        data = pytesseract.image_to_data(image, lang=self.lang, output_type=pytesseract.Output.DICT)

        words: List[Dict] = []
        text_lines: Dict[tuple, List[str]] = {}

        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            conf = int(float(data["conf"][i])) if data["conf"][i] not in ("-1", -1) else -1
            if not text or conf < self.min_confidence:
                continue
            bbox = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
            block_num, line_num = data["block_num"][i], data["line_num"][i]
            words.append({
                "text": text, "bbox": bbox, "conf": conf,
                "block_num": block_num, "line_num": line_num, "word_num": data["word_num"][i],
            })
            text_lines.setdefault((block_num, line_num), []).append(text)

        full_text = "\n".join(" ".join(words_in_line) for words_in_line in text_lines.values())
        fields = self._extract_fields(text_lines)
        dates = _DATE_RE.findall(full_text)

        return OCRDocument(full_text=full_text, words=words, fields=fields, dates=dates)

    @staticmethod
    def _extract_fields(text_lines: Dict[tuple, List[str]]) -> Dict[str, str]:
        fields: Dict[str, str] = {}
        for words_in_line in text_lines.values():
            line = " ".join(words_in_line)
            m = _FIELD_LINE_RE.match(line)
            if m:
                key, value = m.group(1).strip().lower(), m.group(2).strip()
                fields[key] = value
        return fields


def classify_token(token: str) -> str:
    if _DATE_RE.fullmatch(token):
        return "date"
    if _NUMERIC_RE.fullmatch(token.replace(" ", "")):
        return "numeric"
    return "text"
