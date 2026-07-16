"""분류 워크플로우로 보낼 '문서 전체 텍스트' 생성 (#315).

입력 형태별로 3종: DoclingDocument / parse-format elements / langchain Document 리스트.
그림(PictureItem)은 제외, 표(TableItem)는 마크다운으로 포함(표 셀 PII 도 분류 대상 —
청크도 표를 마크다운으로 담으므로 quote 매칭 가능).
"""
from __future__ import annotations

from docling_core.types.doc import DoclingDocument, PictureItem, TableItem


def doc_text(document: DoclingDocument) -> str:
    """DoclingDocument → 문서 전체 텍스트. 그림 제외, 표는 마크다운으로 포함."""
    parts = []
    for it, _ in document.iterate_items():
        if isinstance(it, PictureItem):
            continue
        if isinstance(it, TableItem):
            try:
                t = it.export_to_markdown(document)
            except Exception:
                cells = getattr(getattr(it, "data", None), "table_cells", None) or []
                t = " ".join((getattr(c, "text", "") or "") for c in cells)
        else:
            t = getattr(it, "text", None)
        if isinstance(t, str) and t.strip():
            parts.append(t)
    return "\n".join(parts)


def elements_text(elements: list) -> str:
    """parse-format element 리스트(content)에서 문서 전체 텍스트 결합."""
    return "\n".join(
        str((el or {}).get("content", "") or "")
        for el in (elements or [])
        if str((el or {}).get("content", "") or "").strip()
    )


def docs_text(documents) -> str:
    """langchain Document 리스트(page_content)에서 문서 전체 텍스트 결합(fallback 경로)."""
    return "\n".join(
        d.page_content for d in documents
        if isinstance(getattr(d, "page_content", None), str) and d.page_content.strip()
    )
