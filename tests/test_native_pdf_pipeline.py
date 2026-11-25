# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from threading import Barrier

import pytest
from docling_core.types.doc import CoordOrigin, DocItemLabel
from docling_core.types.doc.page import TextCellUnit

from docling.backend.docling_parse_backend import (
    ThreadedDoclingParseDocumentBackend,
    ThreadedDoclingParsePageBackend,
)
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.backend_options import ThreadedDoclingParseBackendOptions
from docling.datamodel.base_models import ConversionStatus, FailureCategory, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import NativePdfPipelineOptions
from docling.datamodel.settings import PageRange, settings
from docling.document_converter import DocumentConverter, NativePdfFormatOption
from docling.pipeline.native_pdf_pipeline import NativePdfPipeline

TEXT_PDF = Path("tests/data/pdf/sources/2305.03393v1-pg9.pdf")
PICTURE_PDF = Path("tests/data/pdf/sources/picture_classification.pdf")
MULTIPAGE_PDF = Path("tests/data/pdf/sources/normal_4pages.pdf")


def _convert(
    source: Path,
    page_range: PageRange | None = None,
    *,
    raises_on_error: bool = True,
    **options,
) -> ConversionResult:
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: NativePdfFormatOption(
                pipeline_options=NativePdfPipelineOptions(**options)
            )
        }
    )
    kwargs = {} if page_range is None else {"page_range": page_range}
    return converter.convert(source, raises_on_error=raises_on_error, **kwargs)


def test_native_pipeline_emits_one_text_item_per_line_cell():
    conv_res = _convert(TEXT_PDF)

    assert conv_res.status == ConversionStatus.SUCCESS
    doc = conv_res.document
    assert doc.name == TEXT_PDF.stem
    assert list(doc.pages) == [1]

    line_cells = [
        cell for cell in conv_res.pages[0].parsed_page.textline_cells if cell.text
    ]
    assert len(doc.texts) == len(line_cells)
    assert {text.label for text in doc.texts} == {DocItemLabel.TEXT}
    assert "Optimized Table Tokenization for Table Structure Recognition" in (
        doc.export_to_markdown()
    )


def test_native_pipeline_text_provenance_matches_the_page():
    conv_res = _convert(TEXT_PDF)

    page_size = conv_res.document.pages[1].size
    for text in conv_res.document.texts:
        assert len(text.prov) == 1
        prov = text.prov[0]
        assert prov.page_no == 1
        # Provenance boxes are expressed bottom-left, inside the page.
        assert prov.bbox.coord_origin == CoordOrigin.BOTTOMLEFT
        assert 0 <= prov.bbox.l <= prov.bbox.r <= page_size.width
        assert 0 <= prov.bbox.b <= prov.bbox.t <= page_size.height
        assert prov.charspan == (0, len(text.text))


def test_native_pipeline_word_unit_is_finer_than_line_unit():
    lines = _convert(TEXT_PDF).document
    words = _convert(TEXT_PDF, text_cell_unit=TextCellUnit.WORD).document

    assert len(words.texts) > len(lines.texts)
    assert all(" " not in text.text for text in words.texts)


def test_native_pipeline_materializes_char_cells_only_when_requested():
    conv_res = _convert(TEXT_PDF, text_cell_unit=TextCellUnit.CHAR)

    char_cells = [
        cell for cell in conv_res.pages[0].parsed_page.char_cells if cell.text
    ]
    assert char_cells
    assert len(conv_res.document.texts) == len(char_cells)

    default_format_option = NativePdfFormatOption()
    assert default_format_option.backend_options._materialize_char_cells is False

    explicit_backend_options = ThreadedDoclingParseBackendOptions()
    char_format_option = NativePdfFormatOption(
        pipeline_options=NativePdfPipelineOptions(text_cell_unit=TextCellUnit.CHAR),
        backend_options=explicit_backend_options,
    )
    assert char_format_option.backend_options._materialize_char_cells is True
    assert explicit_backend_options._materialize_char_cells is False
    assert (
        "_materialize_char_cells" not in char_format_option.backend_options.model_dump()
    )


def test_native_pipeline_extracts_native_bitmaps_as_pictures():
    doc = _convert(PICTURE_PDF).document

    assert len(doc.pictures) > 0
    assert all(picture.image is not None for picture in doc.pictures)
    assert all(len(picture.prov) == 1 for picture in doc.pictures)


def test_native_pipeline_can_skip_picture_images():
    doc = _convert(PICTURE_PDF, generate_picture_images=False).document

    # The pictures are still located on the page, they just carry no image.
    assert len(doc.pictures) > 0
    assert all(picture.image is None for picture in doc.pictures)


def test_native_pipeline_renders_page_images_at_the_requested_scale():
    doc = _convert(TEXT_PDF, images_scale=2.0).document

    page = doc.pages[1]
    assert page.image is not None
    assert page.image.dpi == 144
    assert page.image.pil_image.width == round(page.size.width * 2.0)


def test_native_pipeline_without_page_images_parses_only():
    conv_res = _convert(TEXT_PDF, generate_page_images=False)

    assert conv_res.status == ConversionStatus.SUCCESS
    assert conv_res.document.pages[1].image is None
    assert len(conv_res.document.texts) > 0


def test_native_format_option_parses_with_all_but_one_cpu_thread():
    format_option = NativePdfFormatOption()

    expected = max(1, (os.cpu_count() or 2) - 1)
    assert format_option.pipeline_options.parser_threads == expected
    assert format_option.backend_options.parser_threads == expected

    # The parser threads are independent from the (unused) inference threads.
    format_option = NativePdfFormatOption(
        pipeline_options=NativePdfPipelineOptions(
            parser_threads=3, accelerator_options=AcceleratorOptions(num_threads=7)
        )
    )
    assert format_option.backend_options.parser_threads == 3


def test_native_pipeline_restores_document_order(monkeypatch):
    """The threaded backend yields pages in completion order, not page order."""
    original_iter_pages = ThreadedDoclingParseDocumentBackend.iter_pages

    def _reversed_iter_pages(self):
        yield from reversed(list(original_iter_pages(self)))

    monkeypatch.setattr(
        ThreadedDoclingParseDocumentBackend, "iter_pages", _reversed_iter_pages
    )

    conv_res = _convert(MULTIPAGE_PDF)

    assert [page.page_no for page in conv_res.pages] == [1, 2, 3, 4]
    doc = conv_res.document
    assert list(doc.pages) == [1, 2, 3, 4]
    page_nos = [text.prov[0].page_no for text in doc.texts]
    assert page_nos == sorted(page_nos)


def test_native_pipeline_keeps_concurrent_conversion_metrics_separate(
    monkeypatch, caplog
):
    barrier = Barrier(2)
    original_assemble_document = NativePdfPipeline._assemble_document

    def synchronized_assemble_document(self, conv_res):
        barrier.wait()
        return original_assemble_document(self, conv_res)

    monkeypatch.setattr(
        NativePdfPipeline, "_assemble_document", synchronized_assemble_document
    )
    monkeypatch.setattr(settings.perf, "doc_batch_size", 2)
    monkeypatch.setattr(settings.perf, "doc_batch_concurrency", 2)
    caplog.set_level(logging.INFO, logger="docling.pipeline.native_pdf_pipeline")

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: NativePdfFormatOption(
                pipeline_options=NativePdfPipelineOptions(generate_page_images=False)
            )
        }
    )
    results = list(converter.convert_all([TEXT_PDF, MULTIPAGE_PDF]))

    logged_text_counts = {
        record.args[0]: record.args[1]
        for record in caplog.records
        if record.msg.startswith("Native assembly of")
    }
    assert logged_text_counts == {
        result.input.file.name: len(result.document.texts) for result in results
    }


def test_native_pipeline_honors_the_page_range():
    doc = _convert(PICTURE_PDF, page_range=(2, 2)).document

    assert list(doc.pages) == [2]
    assert all(text.prov[0].page_no == 2 for text in doc.texts)


@pytest.mark.parametrize(
    ("returned_pages", "status", "missing_pages"),
    [
        ((), ConversionStatus.FAILURE, [2, 3]),
        ((3,), ConversionStatus.PARTIAL_SUCCESS, [2]),
        ((2, 3), ConversionStatus.SUCCESS, []),
    ],
    ids=["no-pages", "missing-page", "complete-range"],
)
def test_native_pipeline_reports_missing_pages(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    returned_pages: tuple[int, ...],
    status: ConversionStatus,
    missing_pages: list[int],
) -> None:
    original_iter_pages = ThreadedDoclingParseDocumentBackend.iter_pages

    def iter_pages(
        backend: ThreadedDoclingParseDocumentBackend,
    ) -> Iterator[ThreadedDoclingParsePageBackend]:
        for page in original_iter_pages(backend):
            if page.page_no in returned_pages:
                yield page
            else:
                page.unload()

    monkeypatch.setattr(ThreadedDoclingParseDocumentBackend, "iter_pages", iter_pages)

    result = _convert(MULTIPAGE_PDF, page_range=(2, 3), raises_on_error=False)

    assert result.status == status
    assert [error.page_no for error in result.errors] == missing_pages
    assert all(
        error.category == FailureCategory.BACKEND_FAILURE for error in result.errors
    )
    assert list(result.document.pages) == list(returned_pages)
    assert {item.prov[0].page_no for item in result.document.texts} == set(
        returned_pages
    )
    assert ("did not return" in caplog.text) == bool(missing_pages)


def test_native_pipeline_reports_pages_skipped_by_the_timeout() -> None:
    """Each page that the document timeout left unread gets its own error."""
    result = _convert(MULTIPAGE_PDF, document_timeout=1e-9)

    converted_pages = set(result.document.pages)
    assert result.status == ConversionStatus.PARTIAL_SUCCESS
    assert len(converted_pages) == 1
    assert all(error.category == FailureCategory.TIMEOUT for error in result.errors)
    assert sorted(
        error.page_no for error in result.errors if error.page_no is not None
    ) == sorted({1, 2, 3, 4} - converted_pages)
