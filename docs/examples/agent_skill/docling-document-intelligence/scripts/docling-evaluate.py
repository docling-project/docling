#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Evaluate a Docling JSON export and suggest pipeline / option changes.

Typical flow (agent or human):

  docling input.pdf --to json --output /tmp/
  docling input.pdf --to md --output /tmp/
  python3 scripts/docling-evaluate.py /tmp/input.json --markdown /tmp/input.md

Exit codes: 0 = pass; 1 = fail or --fail-on-warn with status warn
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_document(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        from docling_core.types.doc.document import DoclingDocument

        return DoclingDocument.model_validate(data), data
    except Exception:
        return None, data


def page_numbers_from_doc(doc) -> set[int]:
    pages: set[int] = set()
    for item, _ in doc.iterate_items():
        for prov in getattr(item, "prov", None) or []:
            p = getattr(prov, "page_no", None)
            if p is not None:
                pages.add(int(p))
    return pages


def collect_text_samples(doc, limit: int = 200) -> list[str]:
    texts: list[str] = []
    for item, _ in doc.iterate_items():
        t = getattr(item, "text", None)
        if t and str(t).strip():
            texts.append(str(t).strip())
            if len(texts) >= limit:
                break
    return texts


def metrics_from_doc(doc) -> dict[str, Any]:
    n_tables = len(getattr(doc, "tables", []) or [])
    n_pictures = len(getattr(doc, "pictures", []) or [])
    n_headers = 0
    n_text_items = 0
    total_chars = 0
    for item, _ in doc.iterate_items():
        label = getattr(getattr(item, "label", None), "name", None) or ""
        if label == "SECTION_HEADER":
            n_headers += 1
        t = getattr(item, "text", None)
        if t:
            n_text_items += 1
            total_chars += len(str(t))

    pages = page_numbers_from_doc(doc)
    n_pages = len(pages) if pages else 0
    density = (total_chars / n_pages) if n_pages else total_chars

    samples = collect_text_samples(doc)
    rep = Counter(samples)
    top_rep = rep.most_common(1)[0] if rep else ("", 0)
    dup_ratio = (
        sum(c for _, c in rep.items() if c > 2) / max(len(rep), 1) if rep else 0.0
    )

    md = ""
    try:
        md = doc.export_to_markdown()
    except Exception:
        pass

    replacement = md.count("\ufffd") + sum(str(t).count("\ufffd") for t in samples)

    return {
        "page_count": n_pages,
        "section_headers": n_headers,
        "text_items": n_text_items,
        "total_text_chars": total_chars,
        "chars_per_page": round(density, 2),
        "tables": n_tables,
        "pictures": n_pictures,
        "markdown_chars": len(md),
        "replacement_chars": replacement,
        "most_repeated_text_count": int(top_rep[1]) if top_rep else 0,
        "duplicate_heavy": dup_ratio > 0.15 and len(samples) > 10,
    }


def heuristic_metrics(data: dict) -> dict[str, Any]:
    """Fallback when DoclingDocument cannot be validated (older export / drift)."""
    texts = data.get("texts") or []
    tables = data.get("tables") or []
    body = data.get("body") or {}
    children = body.get("children") if isinstance(body, dict) else None
    n_children = len(children) if isinstance(children, list) else 0
    char_sum = 0
    for t in texts:
        if isinstance(t, dict):
            char_sum += len(str(t.get("text") or ""))
    return {
        "page_count": 0,
        "section_headers": 0,
        "text_items": len(texts),
        "total_text_chars": char_sum,
        "chars_per_page": 0.0,
        "tables": len(tables),
        "pictures": len(data.get("pictures") or []),
        "markdown_chars": 0,
        "replacement_chars": 0,
        "most_repeated_text_count": 0,
        "duplicate_heavy": False,
        "heuristic_only": True,
        "body_children": n_children,
    }


def evaluate(
    m: dict[str, Any],
    *,
    expect_tables: bool,
    min_chars_per_page: float,
    min_markdown_chars: int,
) -> tuple[str, list[str], list[str]]:
    issues: list[str] = []
    actions: list[str] = []

    if m.get("heuristic_only"):
        issues.append("Could not load full DoclingDocument; metrics are partial.")
        actions.append(
            "Ensure docling-core matches export; re-export with: docling <source> --to json --output <dir>"
        )

    cpp = m.get("chars_per_page") or 0
    if m.get("page_count", 0) >= 2 and cpp < min_chars_per_page:
        issues.append(
            f"Low text density ({cpp} chars/page); likely scan, image-heavy PDF, or extraction gap."
        )
        actions.append(
            "Retry: docling <source> --ocr-engine tesserocr (or rapidocr, ocrmac)"
        )
        actions.append("Retry: docling <source> --pipeline vlm")

    if m.get("replacement_chars", 0) > 5:
        issues.append(
            "Unicode replacement characters detected; OCR may be garbling text."
        )
        actions.append("Retry: docling <source> --ocr-engine tesserocr (or rapidocr)")
        actions.append(
            "Retry: docling <source> --pipeline vlm (use force_backend_text=True via Python API for hybrid)"
        )

    if m.get("duplicate_heavy") or (m.get("most_repeated_text_count", 0) > 8):
        issues.append(
            "Repeated text blocks; possible layout/OCR loop or bad reading order."
        )
        actions.append("Retry: docling <source> --pipeline vlm")
        actions.append("If using VLM: try force_backend_text=True via Python API for text-heavy pages")

    if expect_tables and m.get("tables", 0) == 0:
        issues.append("No tables detected but tables were expected.")
        actions.append("Retry: docling <source> (tables are enabled by default; remove --no-tables if set)")
        actions.append("Retry: docling <source> --pipeline vlm (better for merged-cell or visual tables)")

    mc = m.get("markdown_chars", 0)
    if mc > 0 and mc < min_markdown_chars and m.get("page_count", 0) >= 1:
        issues.append(f"Markdown export is very short ({mc} chars) for the page count.")
        actions.append("Retry: docling <source> --pipeline vlm (or try different --ocr-engine)")

    if m.get("text_items", 0) == 0 and m.get("page_count", 0) == 0:
        issues.append(
            "No text items and no page provenance; export may be empty or invalid."
        )
        actions.append(
            "Verify source file opens correctly; retry with: docling <source> --pipeline standard"
        )

    seen = set()
    uniq_actions = []
    for a in actions:
        if a not in seen:
            seen.add(a)
            uniq_actions.append(a)

    if not issues:
        return "pass", [], []

    severe = m.get("text_items", 0) == 0 or (
        m.get("page_count", 0) >= 1 and mc < 50 and mc > 0
    )
    status = "fail" if severe or m.get("replacement_chars", 0) > 20 else "warn"
    return status, issues, uniq_actions


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Docling JSON export quality")
    p.add_argument(
        "json_path", type=Path, help="Path to DoclingDocument JSON (export_to_dict)"
    )
    p.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional markdown file to cross-check length",
    )
    p.add_argument("--expect-tables", action="store_true")
    p.add_argument("--min-chars-per-page", type=float, default=120.0)
    p.add_argument("--min-markdown-chars", type=int, default=200)
    p.add_argument("--fail-on-warn", action="store_true")
    p.add_argument(
        "--quiet", action="store_true", help="Only print JSON report to stdout"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.json_path.is_file():
        print(json.dumps({"error": f"not found: {args.json_path}"}), file=sys.stderr)
        sys.exit(1)

    doc, raw = load_document(args.json_path)
    if doc is not None:
        m = metrics_from_doc(doc)
    else:
        m = heuristic_metrics(raw)

    if args.markdown and args.markdown.is_file():
        md_len = len(args.markdown.read_text(encoding="utf-8"))
        m["markdown_file_chars"] = md_len
        if m.get("markdown_chars", 0) == 0:
            m["markdown_chars"] = md_len

    status, issues, actions = evaluate(
        m,
        expect_tables=args.expect_tables,
        min_chars_per_page=args.min_chars_per_page,
        min_markdown_chars=args.min_markdown_chars,
    )

    report = {
        "status": status,
        "metrics": m,
        "issues": issues,
        "recommended_actions": actions,
        "next_steps_for_agent": [
            "Re-run docling with flags from recommended_actions.",
            "Re-export JSON and run this script again until status is pass.",
            "Append a row to improvement-log.md (see SKILL.md).",
        ],
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not args.quiet:
        print(f"\nstatus={status}", file=sys.stderr)
        if issues:
            print("issues:", file=sys.stderr)
            for i in issues:
                print(f"  - {i}", file=sys.stderr)
        if actions:
            print("recommended_actions:", file=sys.stderr)
            for a in actions:
                print(f"  - {a}", file=sys.stderr)

    if status == "fail":
        sys.exit(1)
    if status == "warn" and args.fail_on_warn:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
