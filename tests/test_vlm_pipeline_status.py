# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for VlmPipeline._determine_status with VLM stop reasons.

Verifies that VlmPipeline correctly reports PARTIAL_SUCCESS when
individual pages have problematic VLM stop reasons (INFERENCE_ERROR, LENGTH,
CONTENT_FILTERED) or missing predictions.

Related: https://github.com/docling-project/docling/issues/2583
"""

from unittest.mock import MagicMock

import pytest
from docling_core.types.doc.base import Size

from docling.datamodel.base_models import (
    ConversionStatus,
    FailureCategory,
    Page,
    PagePredictions,
    VlmPrediction,
    VlmStopReason,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ResponseFormat
from docling.pipeline.vlm_pipeline import VlmPipeline

pytestmark = pytest.mark.ml_vlm


def _make_page(page_no: int, stop_reason: VlmStopReason) -> Page:
    """Create a Page with a VLM prediction using the given stop reason."""
    page = Page(page_no=page_no)
    page.predictions = PagePredictions(
        vlm_response=VlmPrediction(
            text="some output",
            stop_reason=stop_reason,
        )
    )
    # Provide a valid mock backend so the parent check doesn't flag it
    backend = MagicMock()
    backend.is_valid.return_value = True
    page._backend = backend
    return page


def _make_page_no_vlm(page_no: int) -> Page:
    """Create a Page with no VLM prediction."""
    page = Page(page_no=page_no)
    page.predictions = PagePredictions(vlm_response=None)
    backend = MagicMock()
    backend.is_valid.return_value = True
    page._backend = backend
    return page


def _make_conv_res(pages: list) -> ConversionResult:
    """Create a minimal ConversionResult with the given pages."""
    conv_res = MagicMock(spec=ConversionResult)
    conv_res.pages = pages
    conv_res.errors = []
    conv_res.status = ConversionStatus.STARTED
    # Provide input with a mock backend for parent _determine_status
    conv_res.input = MagicMock()
    conv_res.input._backend = None
    return conv_res


@pytest.fixture
def pipeline() -> VlmPipeline:
    """Create a VlmPipeline instance with minimal options."""
    return VlmPipeline.__new__(VlmPipeline)


def test_all_pages_success(pipeline: VlmPipeline) -> None:
    """All pages with END_OF_SEQUENCE should yield SUCCESS."""
    pages = [
        _make_page(1, VlmStopReason.END_OF_SEQUENCE),
        _make_page(2, VlmStopReason.END_OF_SEQUENCE),
    ]
    conv_res = _make_conv_res(pages)
    status = pipeline._determine_status(conv_res)
    assert status == ConversionStatus.SUCCESS
    assert len(conv_res.errors) == 0


def test_page_truncated_length(pipeline: VlmPipeline) -> None:
    """A page with LENGTH stop reason should yield PARTIAL_SUCCESS."""
    pages = [
        _make_page(1, VlmStopReason.END_OF_SEQUENCE),
        _make_page(2, VlmStopReason.LENGTH),
    ]
    conv_res = _make_conv_res(pages)
    status = pipeline._determine_status(conv_res)
    assert status == ConversionStatus.PARTIAL_SUCCESS
    assert len(conv_res.errors) == 1
    assert conv_res.errors[0].category == FailureCategory.INFERENCE_FAILURE
    assert conv_res.errors[0].page_no == 2


def test_page_content_filtered(pipeline: VlmPipeline) -> None:
    """A page with CONTENT_FILTERED stop reason should yield PARTIAL_SUCCESS."""
    pages = [
        _make_page(1, VlmStopReason.CONTENT_FILTERED),
    ]
    conv_res = _make_conv_res(pages)
    status = pipeline._determine_status(conv_res)
    assert status == ConversionStatus.PARTIAL_SUCCESS
    assert len(conv_res.errors) == 1
    assert conv_res.errors[0].category == FailureCategory.INFERENCE_FAILURE
    assert conv_res.errors[0].page_no == 1


def test_page_no_vlm_response(pipeline: VlmPipeline) -> None:
    """A page with no VLM prediction should yield PARTIAL_SUCCESS."""
    pages = [
        _make_page(1, VlmStopReason.END_OF_SEQUENCE),
        _make_page_no_vlm(2),
    ]
    conv_res = _make_conv_res(pages)
    status = pipeline._determine_status(conv_res)
    assert status == ConversionStatus.PARTIAL_SUCCESS
    assert len(conv_res.errors) == 1
    assert conv_res.errors[0].category == FailureCategory.INFERENCE_FAILURE
    assert conv_res.errors[0].page_no == 2


def test_stop_sequence_is_success(pipeline: VlmPipeline) -> None:
    """STOP_SEQUENCE is a normal completion and should yield SUCCESS."""
    pages = [
        _make_page(1, VlmStopReason.STOP_SEQUENCE),
    ]
    conv_res = _make_conv_res(pages)
    status = pipeline._determine_status(conv_res)
    assert status == ConversionStatus.SUCCESS
    assert len(conv_res.errors) == 0


def test_unspecified_is_success(pipeline: VlmPipeline) -> None:
    """UNSPECIFIED (the default stop reason) should yield SUCCESS."""
    pages = [
        _make_page(1, VlmStopReason.UNSPECIFIED),
    ]
    conv_res = _make_conv_res(pages)
    status = pipeline._determine_status(conv_res)
    assert status == ConversionStatus.SUCCESS
    assert len(conv_res.errors) == 0


def test_multiple_failures_accumulate_errors(pipeline: VlmPipeline) -> None:
    """Multiple problematic pages should each record an error."""
    pages = [
        _make_page(1, VlmStopReason.LENGTH),
        _make_page(2, VlmStopReason.CONTENT_FILTERED),
        _make_page(3, VlmStopReason.END_OF_SEQUENCE),
    ]
    conv_res = _make_conv_res(pages)
    status = pipeline._determine_status(conv_res)
    assert status == ConversionStatus.PARTIAL_SUCCESS
    assert len(conv_res.errors) == 2
    assert conv_res.errors[0].category == FailureCategory.INFERENCE_FAILURE
    assert conv_res.errors[0].page_no == 1
    assert conv_res.errors[1].category == FailureCategory.INFERENCE_FAILURE
    assert conv_res.errors[1].page_no == 2


def test_page_inference_error_is_partial_success_with_the_reason(
    pipeline: VlmPipeline,
) -> None:
    """A failed remote API call must not pass as an empty page: the status is
    PARTIAL_SUCCESS and the ErrorItem carries the provider's reason (#4009)."""
    page = _make_page(2, VlmStopReason.INFERENCE_ERROR)
    page.predictions.vlm_response = VlmPrediction(
        text="",
        stop_reason=VlmStopReason.INFERENCE_ERROR,
        error_message="HTTP 400: Unsupported parameter: temperature",
    )
    conv_res = _make_conv_res([_make_page(1, VlmStopReason.END_OF_SEQUENCE), page])
    status = pipeline._determine_status(conv_res)
    assert status == ConversionStatus.PARTIAL_SUCCESS
    assert len(conv_res.errors) == 1
    assert conv_res.errors[0].category == FailureCategory.INFERENCE_FAILURE
    assert conv_res.errors[0].page_no == 2
    assert "HTTP 400: Unsupported parameter: temperature" in (
        conv_res.errors[0].error_message
    )


def test_one_failed_page_among_good_pages_is_partial_success(
    pipeline: VlmPipeline,
) -> None:
    pages = [
        _make_page(1, VlmStopReason.END_OF_SEQUENCE),
        _make_page(2, VlmStopReason.INFERENCE_ERROR),
        _make_page(3, VlmStopReason.END_OF_SEQUENCE),
    ]
    conv_res = _make_conv_res(pages)
    status = pipeline._determine_status(conv_res)
    assert status == ConversionStatus.PARTIAL_SUCCESS
    assert [error.page_no for error in conv_res.errors] == [2]


def test_all_pages_failed_is_a_failure(pipeline: VlmPipeline) -> None:
    """When no page produced any output the conversion is not a partial result:
    FAILURE, so that `raises_on_error=True` fires (#4009)."""
    pages = [
        _make_page(1, VlmStopReason.INFERENCE_ERROR),
        _make_page(2, VlmStopReason.INFERENCE_ERROR),
    ]
    conv_res = _make_conv_res(pages)
    status = pipeline._determine_status(conv_res)
    assert status == ConversionStatus.FAILURE
    assert [error.page_no for error in conv_res.errors] == [1, 2]
    assert all(
        error.category == FailureCategory.INFERENCE_FAILURE for error in conv_res.errors
    )


def test_failed_page_is_not_parsed_as_empty_output(
    pipeline: VlmPipeline, monkeypatch
) -> None:
    """A page whose inference failed has no output to parse. The response-format
    parser must not add a second error on top of the one _determine_status
    reports, unlike a completion that legitimately came back empty."""
    monkeypatch.setattr(pipeline, "_response_format", lambda: ResponseFormat.DOCLANG)
    monkeypatch.setattr(pipeline, "_finalize_page_output", lambda document, page: None)

    failed = _make_page(1, VlmStopReason.INFERENCE_ERROR)
    failed.predictions.vlm_response = VlmPrediction(
        text="", stop_reason=VlmStopReason.INFERENCE_ERROR, error_message="ReadTimeout"
    )
    conv_res = _make_conv_res([failed])
    document = pipeline._finalize_page_document(conv_res, failed)
    assert conv_res.errors == []
    assert document.name == "page_1"

    empty = _make_page(2, VlmStopReason.UNSPECIFIED)
    empty.predictions.vlm_response = VlmPrediction(
        text="", stop_reason=VlmStopReason.UNSPECIFIED
    )
    empty.size = Size(width=1, height=1)
    conv_res = _make_conv_res([empty])
    pipeline._finalize_page_document(conv_res, empty)
    assert [error.error_message for error in conv_res.errors] == [
        "No <doclang> XML fragment found in VLM response."
    ]
