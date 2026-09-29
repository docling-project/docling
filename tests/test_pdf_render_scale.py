# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pytest

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.backend_options import (
    PdfBackendOptions,
    ThreadedDoclingParseBackendOptions,
)
from docling.datamodel.pipeline_options import (
    OcrAutoOptions,
    PdfPipelineOptions,
    VlmPipelineOptions,
)
from docling.document_converter import PdfFormatOption


@pytest.mark.parametrize(
    ("pipeline_options", "expected_scale"),
    [
        (PdfPipelineOptions(), 3.0),
        (PdfPipelineOptions(do_ocr=False), 2.0),
        (
            PdfPipelineOptions(
                do_ocr=False,
                do_table_structure=False,
                images_scale=5.0,
            ),
            5.0,
        ),
    ],
)
def test_threaded_backend_render_scale_covers_pipeline_consumers(
    pipeline_options: PdfPipelineOptions,
    expected_scale: float,
):
    format_option = PdfFormatOption(pipeline_options=pipeline_options)

    backend_options = format_option.backend_options_for_input("document.pdf")

    assert isinstance(backend_options, ThreadedDoclingParseBackendOptions)
    assert backend_options.render_scale == expected_scale


def test_threaded_backend_render_scale_preserves_higher_explicit_value():
    backend_options = ThreadedDoclingParseBackendOptions(
        parser_threads=2,
        render_scale=6.0,
    )
    format_option = PdfFormatOption(
        pipeline_options=PdfPipelineOptions(images_scale=5.0),
        backend_options=backend_options,
    )

    resolved = format_option.backend_options_for_input("document.pdf")

    assert resolved is backend_options
    assert resolved.render_scale == 6.0
    assert resolved.parser_threads == 2


def test_threaded_backend_render_scale_raises_lower_explicit_value():
    backend_options = ThreadedDoclingParseBackendOptions(
        parser_threads=2,
        render_scale=1.0,
    )
    format_option = PdfFormatOption(
        pipeline_options=PdfPipelineOptions(images_scale=5.0),
        backend_options=backend_options,
    )

    resolved = format_option.backend_options_for_input("document.pdf")

    assert isinstance(resolved, ThreadedDoclingParseBackendOptions)
    assert resolved is not backend_options
    assert resolved.render_scale == 5.0
    assert resolved.parser_threads == 2


def test_threaded_backend_render_scale_follows_configured_ocr_scale():
    """A page render below the OCR engine's own scale degrades recognition
    (see #4395): the backend render must never undercut it."""
    pipeline_options = PdfPipelineOptions(
        ocr_options=OcrAutoOptions(scale=4.5),
    )
    format_option = PdfFormatOption(pipeline_options=pipeline_options)

    backend_options = format_option.backend_options_for_input("document.pdf")

    assert isinstance(backend_options, ThreadedDoclingParseBackendOptions)
    assert backend_options.render_scale == 4.5


def test_threaded_backend_render_scale_ignores_non_pdf_pipeline_options():
    """`PdfFormatOption` also carries VLM pipeline options (see the VLM
    branch of the CLI). Those have no `do_ocr`/`ocr_options`/
    `do_table_structure`, so the render-scale lookup must stay gated on
    `PdfPipelineOptions` rather than reading those fields unconditionally."""
    format_option = PdfFormatOption(
        pipeline_options=VlmPipelineOptions(images_scale=2.0),
    )

    backend_options = format_option.backend_options_for_input("document.pdf")

    assert backend_options is None


def test_non_threaded_pdf_backend_options_are_unchanged():
    backend_options = PdfBackendOptions()
    format_option = PdfFormatOption(
        pipeline_options=PdfPipelineOptions(images_scale=5.0),
        backend=PyPdfiumDocumentBackend,
        backend_options=backend_options,
    )

    resolved = format_option.backend_options_for_input("document.pdf")

    assert resolved is backend_options
