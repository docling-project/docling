from __future__ import annotations

from pathlib import Path
import asyncio
import sys
import pytest


DOCX_SAMPLES = sorted(
    (Path(__file__).resolve().parents[2] / "sample_files").glob("*.docx")
)


class _DummyRequest:
    async def is_disconnected(self) -> bool:  # pragma: no cover
        return False


def _import_processor():
    try:
        from facade.basic_processor import DocumentProcessor
        return DocumentProcessor
    except ModuleNotFoundError:
        sys.path.append(str(Path(__file__).resolve().parents[3]))
        from facade.basic_processor import DocumentProcessor
        return DocumentProcessor


def _has_adjacent_duplicate(lines: list[str]) -> bool:
    prev = None
    for line in lines:
        cur = line.strip()
        if not cur:
            prev = cur
            continue
        if prev is not None and cur == prev:
            return True
        prev = cur
    return False


@pytest.mark.unit
@pytest.mark.parametrize("sample_path", DOCX_SAMPLES, ids=lambda p: p.name)
def test_no_adjacent_duplicate_lines_in_vectors_for_docx_samples(sample_path: Path):
    DocumentProcessor = _import_processor()

    if not sample_path.exists():
        pytest.skip(f"sample not found: {sample_path}")

    dp = DocumentProcessor()

    async def _run():
        return await dp(_DummyRequest(), str(sample_path))

    vectors = asyncio.run(_run())

    assert isinstance(vectors, list)
    assert len(vectors) >= 1

    for v in vectors:
        text = getattr(v, "text", None) if hasattr(v, "text") else v.get("text")
        assert isinstance(text, str)
        lines = text.splitlines()
        assert _has_adjacent_duplicate(lines) is False


