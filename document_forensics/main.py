"""
CLI entry point for the Document Forensics module.

Usage:
    python main.py analyze --doc path/to/document.pdf --out report.json
    python main.py compare --versions v1.png v2.png v3.png --out timeline.json
    python main.py analyze --doc scan.jpg --no-ocr
"""
from __future__ import annotations

import argparse
import json
import sys

from src.pipeline import analyze_document, compare_versions, save_report


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Powered Document Forensics CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Run tampering + consistency checks on one document")
    p_analyze.add_argument("--doc", required=True, help="Path to a PDF or image file")
    p_analyze.add_argument("--out", default=None, help="Optional path to save the JSON report")
    p_analyze.add_argument("--no-ocr", action="store_true", help="Skip OCR/layout consistency check")

    p_compare = sub.add_parser("compare", help="Compare multiple versions of a document")
    p_compare.add_argument("--versions", nargs="+", required=True,
                            help="Paths to version files in chronological order (V1 V2 V3 ...)")
    p_compare.add_argument("--out", default=None, help="Optional path to save the JSON timeline")
    p_compare.add_argument("--no-ocr", action="store_true", help="Skip text-level diffing")

    args = parser.parse_args()

    if args.command == "analyze":
        report = analyze_document(args.doc, run_ocr=not args.no_ocr)
        print(f"\n=== Forensic Report: {args.doc} ===")
        print(f"Overall risk score : {report.risk_score:.2f} / 1.00")
        print(f"Verdict            : {report.verdict}")
        print("Module scores      :", json.dumps(report.module_scores, indent=2))
        print("Evidence:")
        for line in report.evidence or ["  (none)"]:
            print(f"  - {line}")
        if args.out:
            save_report(report, args.out)
            print(f"\nFull report saved to {args.out}")

    elif args.command == "compare":
        report = compare_versions(args.versions, run_ocr=not args.no_ocr)
        print(f"\n=== Version Comparison: {' -> '.join(args.versions)} ===")
        print(report.summary)
        if args.out:
            save_report(report, args.out)
            print(f"\nFull timeline saved to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
