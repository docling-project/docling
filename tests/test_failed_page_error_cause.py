# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Regression tests for exceptions raised during page processing."""

from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from docling.backend.image_backend import ImageDocumentBackend
from docling.datamodel.base_models import ConversionStatus, InputFormat, Page
from docling.datamodel.document import ConversionResult, InputDocument
from docling.pipeline.standard_pdf_pipeline import (
    ProcessingResult,
    StandardPdfPipeline,
)


def test_failed_image_page_preserves_decode_exception():
    image_bytes = BytesIO()
    Image.linear_gradient("L").convert("RGB").save(image_bytes, format="JPEG")
    damaged = BytesIO(image_bytes.getvalue()[: len(image_bytes.getvalue()) // 2])

    input_doc = InputDocument(
        path_or_stream=damaged,
        format=InputFormat.IMAGE,
        backend=ImageDocumentBackend,
        filename="damaged.jpg",
    )
    assert input_doc.valid

    backend = input_doc._backend
    assert backend is not None
    try:
        with pytest.raises(OSError) as exc_info:
            backend.load_page(0)
    finally:
        backend.unload()

    original_error = exc_info.value

    pipeline = StandardPdfPipeline.__new__(StandardPdfPipeline)
    pipeline.keep_images = False
    pipeline.keep_backend = False
    pipeline.pipeline_options = SimpleNamespace(generate_parsed_pages=False)

    result = ConversionResult(
        input=input_doc,
        pages=[Page(page_no=1)],
    )
    pipeline._integrate_results(
        result,
        ProcessingResult(
            failed_pages=[(1, original_error, None)],
            total_expected=1,
        ),
    )

    assert result.status == ConversionStatus.FAILURE
    assert result._page_failure_cause is original_error


def test_converter_chains_original_page_failure(monkeypatch):
    """ConversionError exposes the original page exception to callers."""
    from docling.datamodel.base_models import DocumentStream
    from docling.document_converter import DocumentConverter
    from docling.exceptions import ConversionError

    image_bytes = BytesIO()
    Image.new("RGB", (16, 16)).save(image_bytes, format="PNG")
    payload = image_bytes.getvalue()

    input_doc = InputDocument(
        path_or_stream=BytesIO(payload),
        format=InputFormat.IMAGE,
        backend=ImageDocumentBackend,
        filename="photo.png",
    )
    original_error = OSError("image file is truncated")
    result = ConversionResult(
        input=input_doc,
        status=ConversionStatus.FAILURE,
    )
    result._page_failure_cause = original_error

    converter = DocumentConverter(allowed_formats=[InputFormat.IMAGE])
    monkeypatch.setattr(
        converter,
        "_convert",
        lambda *args, **kwargs: iter([result]),
    )

    try:
        with pytest.raises(ConversionError) as exc_info:
            converter.convert(
                DocumentStream(
                    name="photo.png",
                    stream=BytesIO(payload),
                )
            )

        assert exc_info.value.__cause__ is original_error
        assert "_page_failure_cause" not in result.model_dump()
    finally:
        if input_doc._backend is not None:
            input_doc._backend.unload()


def test_damaged_image_cause_reaches_public_converter(monkeypatch):
    """Exercise image decoding and converter error chaining without ML models."""
    from docling.datamodel.base_models import DocumentStream
    from docling.document_converter import DocumentConverter
    from docling.exceptions import ConversionError

    image_bytes = BytesIO()
    Image.linear_gradient("L").convert("RGB").save(image_bytes, format="JPEG")
    damaged_bytes = image_bytes.getvalue()[: len(image_bytes.getvalue()) // 2]

    class LightweightPipeline:
        def execute(self, in_doc, raises_on_error, progress=None):
            backend = in_doc._backend
            assert backend is not None

            try:
                backend.load_page(0)
            except OSError as error:
                failure = error
            else:
                pytest.fail("Expected the damaged JPEG to fail decoding")
            finally:
                backend.unload()

            result = ConversionResult(
                input=in_doc,
                pages=[Page(page_no=1)],
            )
            pipeline = StandardPdfPipeline.__new__(StandardPdfPipeline)
            pipeline.keep_images = False
            pipeline.keep_backend = False
            pipeline.pipeline_options = SimpleNamespace(generate_parsed_pages=False)
            pipeline._integrate_results(
                result,
                ProcessingResult(
                    failed_pages=[(1, failure, None)],
                    total_expected=1,
                ),
            )
            return result

    converter = DocumentConverter(allowed_formats=[InputFormat.IMAGE])
    monkeypatch.setattr(
        converter,
        "_get_pipeline",
        lambda doc_format: LightweightPipeline(),
    )

    source = DocumentStream(
        name="damaged.jpg",
        stream=BytesIO(damaged_bytes),
    )
    with pytest.raises(ConversionError) as exc_info:
        converter.convert(source)

    cause = exc_info.value.__cause__
    assert isinstance(cause, OSError)
    assert "image file is truncated" in str(cause)


@pytest.mark.parametrize("reverse_order", [False, True])
def test_multiple_failed_pages_choose_lowest_page_exception(reverse_order):
    """Choose a deterministic cause regardless of failure arrival order."""
    image_bytes = BytesIO()
    frames = [
        Image.new("RGB", (16, 16)),
        Image.new("RGB", (16, 16)),
    ]
    frames[0].save(
        image_bytes,
        format="TIFF",
        save_all=True,
        append_images=frames[1:],
    )

    input_doc = InputDocument(
        path_or_stream=BytesIO(image_bytes.getvalue()),
        format=InputFormat.IMAGE,
        backend=ImageDocumentBackend,
        filename="two-pages.tiff",
    )

    assert input_doc.valid
    assert input_doc.page_count == 2

    pipeline = StandardPdfPipeline.__new__(StandardPdfPipeline)
    pipeline.keep_images = False
    pipeline.keep_backend = False
    pipeline.pipeline_options = SimpleNamespace(generate_parsed_pages=False)

    first_error = OSError("page 1 corrupted")
    second_error = RuntimeError("page 2 processing failed")
    failed_pages = [
        (1, first_error, None),
        (2, second_error, None),
    ]
    if reverse_order:
        failed_pages.reverse()

    result = ConversionResult(
        input=input_doc,
        pages=[Page(page_no=1), Page(page_no=2)],
    )

    try:
        pipeline._integrate_results(
            result,
            ProcessingResult(
                failed_pages=failed_pages,
                total_expected=2,
            ),
        )

        assert result.status == ConversionStatus.FAILURE
        assert result._page_failure_cause is first_error
        assert len(result.errors) == 2
        errors_by_page = {error.page_no: error.error_message for error in result.errors}
        assert "page 1 corrupted" in errors_by_page[1]
        assert "page 2 processing failed" in errors_by_page[2]
    finally:
        if input_doc._backend is not None:
            input_doc._backend.unload()
