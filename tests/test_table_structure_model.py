# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Literal

import pytest
from docling_core.types.doc import BoundingBox, TableItem
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import InputFormat, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    OcrMode,
    OcrOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.base_ocr_model import BaseOcrModel
from docling.models.stages.table_structure.table_structure_model import (
    _words_and_ocr_cells,
)
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


def _cell(bbox: BoundingBox, text: str, from_ocr: bool) -> TextCell:
    return TextCell(
        index=0,
        rect=BoundingRectangle.from_bounding_box(bbox),
        text=text,
        orig=text,
        from_ocr=from_ocr,
    )


HEADER_BOX = BoundingBox(l=0, t=0, r=50, b=10)
VALUE_BOX = BoundingBox(l=60, t=20, r=90, b=30)


def test_table_tokens_keep_ocr_lines_next_to_native_words() -> None:
    # A scanned form with one filled-in native value: the OCR header must still
    # reach the table, and a native word under OCR text must not be doubled.
    header = _cell(HEADER_BOX, "Week Begin", from_ocr=True)
    native_line = _cell(VALUE_BOX, "5000", from_ocr=False)
    filled_value = _cell(VALUE_BOX, "5000", from_ocr=False)
    hidden_word = _cell(BoundingBox(l=5, t=2, r=20, b=8), "Week", from_ocr=False)

    tokens = _words_and_ocr_cells(
        word_cells=[filled_value, hidden_word],
        cluster_cells=[header, native_line],
    )

    assert [c.text for c in tokens] == ["5000", "Week Begin"]


def test_table_tokens_prefer_ocr_words_over_their_line() -> None:
    # An OCR engine with word output (e.g. a third-party plugin) also writes
    # OCR word cells. They replace their OCR line. The OCR word on the native
    # value is dropped: the OCR merge kept the native text there.
    header = _cell(HEADER_BOX, "Week Begin", from_ocr=True)
    native_line = _cell(VALUE_BOX, "5000", from_ocr=False)
    words = [
        _cell(BoundingBox(l=0, t=0, r=22, b=10), "Week", from_ocr=True),
        _cell(BoundingBox(l=26, t=0, r=50, b=10), "Begin", from_ocr=True),
        _cell(VALUE_BOX, "5000", from_ocr=True),
        _cell(VALUE_BOX, "5000", from_ocr=False),
    ]

    tokens = _words_and_ocr_cells(word_cells=words, cluster_cells=[header, native_line])

    assert [(c.text, c.from_ocr) for c in tokens] == [
        ("5000", False),
        ("Week", True),
        ("Begin", True),
    ]


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


def _tables(pipeline_cls: type[StandardPdfPipeline], options: PdfPipelineOptions):
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
    return [
        [cell.text for cell in item.data.table_cells]
        for item, _ in doc.iterate_items()
        if isinstance(item, TableItem)
    ]


@pytest.mark.parametrize("mode", [OcrMode.FULL_PAGE, OcrMode.DEFAULT])
def test_word_level_ocr_plugin_gives_native_tables(mode: OcrMode) -> None:
    # OCR words that repeat the native words must give the same table as the
    # native words. In full-page mode, the OCR lines as tokens would merge
    # cells. In the default mode, the merge keeps the native text, and OCR
    # words on it would double every cell.
    native = _tables(StandardPdfPipeline, PdfPipelineOptions(do_ocr=False))
    word_ocr = _tables(
        _WordOcrPipeline,
        PdfPipelineOptions(do_ocr=True, ocr_options=_WordOcrOptions(mode=mode)),
    )

    assert native
    assert word_ocr == native
