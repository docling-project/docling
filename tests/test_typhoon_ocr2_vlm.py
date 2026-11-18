# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT
# Typhoon OCR V2 by the Typhoon team: https://huggingface.co/typhoon-ai/typhoon-ocr2-2b

"""Typhoon OCR V2 inference on a local engine.

Set ``DOCLING_TEST_TYPHOON_OCR2=1`` to run the test. It downloads the 0.8B weights
for the engine that auto-inline picks. ``tests/test_typhoon_ocr2.py`` tests the
presets and the parser without model downloads.
"""

import os
from pathlib import Path

import pytest
from docling_core.types.doc import ContentLayer, DocItemLabel, SectionHeaderItem

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.vlm_engine_options import AutoInlineVlmEngineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

pytestmark = pytest.mark.ml_vlm

_PDF = Path(__file__).parent / "data" / "pdf" / "sources" / "2305.03393v1-pg9.pdf"


@pytest.mark.skipif(
    os.getenv("DOCLING_TEST_TYPHOON_OCR2") != "1",
    reason="set DOCLING_TEST_TYPHOON_OCR2=1 to run the Typhoon OCR V2 inference test",
)
def test_e2e_typhoon_ocr2() -> None:
    """Convert a page with the 0.8B preset on the local engine that auto-inline picks."""
    options = VlmConvertOptions.from_preset(
        "typhoon_ocr2_08b", engine_options=AutoInlineVlmEngineOptions()
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=VlmPipelineOptions(vlm_options=options),
            )
        }
    )

    result = converter.convert(_PDF)

    # SUCCESS excludes PARTIAL_SUCCESS, which a cut or invalid reply gives.
    assert result.status == ConversionStatus.SUCCESS
    doc = result.document
    assert [table.data.num_cols for table in doc.tables] == [8]
    assert "5.1 Hyper Parameter Optimization" in [
        item.text for item in doc.texts if isinstance(item, SectionHeaderItem)
    ]
    assert any(
        item.label == DocItemLabel.PAGE_HEADER
        and item.content_layer == ContentLayer.FURNITURE
        for item in doc.texts
    )
