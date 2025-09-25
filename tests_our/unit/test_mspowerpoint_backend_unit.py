from __future__ import annotations

from pathlib import Path
from typing import cast
import pytest


@pytest.mark.unit
def test_mspowerpoint_backend_convert_on_sample(sample_dir: Path):
    from docling.datamodel.document import InputDocument
    from docling.datamodel.base_models import InputFormat
    from docling.backend.mspowerpoint_backend import MsPowerpointDocumentBackend

    sample_path = Path("doc_parser/sample_files/sample.pptx")
    if not sample_path.exists():
        pytest.skip(f"sample not found: {sample_path}")

    in_doc = InputDocument(
        path_or_stream=sample_path,
        format=InputFormat.PPTX,
        backend=MsPowerpointDocumentBackend,
        filename=sample_path.name,
    )

    assert in_doc.valid is True
    assert in_doc._backend.is_valid() is True
    assert in_doc.page_count >= 1

    # convert 호출 및 결과 검증
    backend = cast(MsPowerpointDocumentBackend, in_doc._backend)
    doc = backend.convert()
    assert doc is not None
    # 최소한 1페이지 이상 생성되었는지 확인
    assert hasattr(doc, "pages")
    assert isinstance(doc.pages, dict)
    assert len(doc.pages) >= 1

    # 최소 하나 이상의 텍스트 아이템이 존재하는지 확인
    assert hasattr(doc, "texts")
    assert isinstance(doc.texts, list)
    assert len(doc.texts) >= 1


