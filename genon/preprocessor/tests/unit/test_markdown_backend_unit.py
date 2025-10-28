from __future__ import annotations

from pathlib import Path
from typing import cast
import pytest


MD_SAMPLES = sorted(
    (Path(__file__).resolve().parents[2] / "sample_files").glob("*.md")
)


@pytest.mark.unit
@pytest.mark.parametrize("sample_path", MD_SAMPLES, ids=lambda p: p.name)
def test_markdown_backend_convert_on_all_md_samples(sample_path: Path):
    from docling.datamodel.document import InputDocument
    from docling.datamodel.base_models import InputFormat
    from docling.backend.md_backend import MarkdownDocumentBackend

    if not sample_path.exists():
        pytest.skip(f"sample not found: {sample_path}")

    in_doc = InputDocument(
        path_or_stream=sample_path,
        format=InputFormat.MD,
        backend=MarkdownDocumentBackend,
        filename=sample_path.name,
    )

    assert in_doc.valid is True
    assert in_doc._backend.is_valid() is True

    backend = cast(MarkdownDocumentBackend, in_doc._backend)
    doc = backend.convert()
    assert doc is not None

    # 텍스트가 최소 1개 이상 존재하는지 확인
    assert hasattr(doc, "texts")
    assert isinstance(doc.texts, list)
    assert len(doc.texts) >= 1


