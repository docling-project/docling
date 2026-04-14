"""
HWPX 백엔드 유효성 테스트 — 팀 컨벤션(test_docx_backend_unit.py)에 맞춤.
실제 샘플 파일로 InputDocument 생성 및 backend.is_valid()만 확인.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast
import pytest


HWPX_SAMPLE = Path(__file__).resolve().parents[2] / "sample_files" / "hwpx_sample.hwpx"
HWP_SAMPLE = Path(__file__).resolve().parents[2] / "sample_files" / "hwp_sample.hwp"


@pytest.mark.unit
@pytest.mark.skipif(not HWPX_SAMPLE.exists(), reason="hwpx_sample.hwpx not found")
def test_hwpx_backend_valid_and_convert():
    """HwpxDocumentBackend가 hwpx 파일을 유효하게 인식하고 convert 가능한지 확인."""
    from docling.datamodel.document import InputDocument
    from docling.datamodel.base_models import InputFormat
    from docling.backend.xml.hwpx_backend import HwpxDocumentBackend

    in_doc = InputDocument(
        path_or_stream=HWPX_SAMPLE,
        format=InputFormat.XML_HWPX,
        backend=HwpxDocumentBackend,
        filename=HWPX_SAMPLE.name,
    )

    assert in_doc.valid is True
    assert in_doc._backend.is_valid() is True

    backend = cast(HwpxDocumentBackend, in_doc._backend)
    doc = backend.convert()
    assert doc is not None
    assert hasattr(doc, "texts")
    assert isinstance(doc.texts, list)
    assert len(doc.texts) >= 1


@pytest.mark.unit
@pytest.mark.skipif(not HWP_SAMPLE.exists(), reason="hwp_sample.hwp not found")
def test_genos_hwp_backend_valid():
    """GenosHwpDocumentBackend가 hwp 파일을 유효하게 인식하는지 확인."""
    from docling.datamodel.document import InputDocument
    from docling.datamodel.base_models import InputFormat
    from docling.backend.genos_hwp_backend import GenosHwpDocumentBackend

    in_doc = InputDocument(
        path_or_stream=HWP_SAMPLE,
        format=InputFormat.HWP,
        backend=GenosHwpDocumentBackend,
        filename=HWP_SAMPLE.name,
    )

    assert in_doc.valid is True
    assert in_doc._backend.is_valid() is True
