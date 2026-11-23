# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Literal

import pytest
from docling_core.types.doc import TableData, TableItem

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    OcrMode,
    OcrOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.base_ocr_model import BaseOcrModel
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

pytestmark = pytest.mark.ml_pdf_model


class _WordOcrOptions(OcrOptions):
    kind: ClassVar[Literal["test_word_ocr"]] = "test_word_ocr"
    lang: list[str] = []


class _WordOcrModel(BaseOcrModel):
    """Reads the native text layer back as OCR output, the way an OCR plugin
    with word output reports it: lines through `post_process_cells`, words
    straight into `parsed_page.word_cells`."""

    @classmethod
    def get_options_type(cls) -> type[OcrOptions]:
        return _WordOcrOptions

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        for page in page_batch:
            assert page._backend is not None
            native = page._backend.get_segmented_page()
            assert native is not None
            lines = [
                c.model_copy(update={"from_ocr": True}) for c in native.textline_cells
            ]
            words = [c.model_copy(update={"from_ocr": True}) for c in native.word_cells]
            self.post_process_cells(lines, page, conv_res)
            assert page.parsed_page is not None
            page.parsed_page.word_cells.extend(words)
            page.parsed_page.has_words = True
            yield page


class _WordOcrPipeline(StandardPdfPipeline):
    def _make_ocr_model(self, art_path: Path | None) -> _WordOcrModel:
        return _WordOcrModel(
            enabled=self.pipeline_options.do_ocr,
            artifacts_path=art_path,
            options=self.pipeline_options.ocr_options,
            accelerator_options=self.pipeline_options.accelerator_options,
        )


def _tables(
    pipeline_cls: type[StandardPdfPipeline], options: PdfPipelineOptions
) -> list[TableData]:
    options.accelerator_options.device = AcceleratorDevice.CPU
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=pipeline_cls, pipeline_options=options
            )
        }
    )
    doc = converter.convert(
        Path("tests/data/pdf/sources/2305.03393v1-pg9.pdf")
    ).document
    return [item.data for item, _ in doc.iterate_items() if isinstance(item, TableItem)]


@pytest.fixture(scope="module")
def native_tables() -> list[TableData]:
    return _tables(StandardPdfPipeline, PdfPipelineOptions(do_ocr=False))


@pytest.mark.parametrize("mode", [OcrMode.FULL_PAGE, OcrMode.DEFAULT])
def test_word_level_ocr_plugin_gives_native_tables(
    mode: OcrMode, native_tables: list[TableData]
) -> None:
    # OCR words that repeat the native words must give the same table as the
    # native words. In full-page mode, the OCR lines as tokens would merge
    # cells. In the default mode, the merge keeps the native text, and OCR
    # words on it would double every cell.
    word_ocr = _tables(
        _WordOcrPipeline,
        PdfPipelineOptions(do_ocr=True, ocr_options=_WordOcrOptions(mode=mode)),
    )

    assert native_tables
    assert word_ocr == native_tables
