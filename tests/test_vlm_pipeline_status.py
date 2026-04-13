"""Tests for VlmPipeline._determine_status with VLM stop reasons.

Verifies that VlmPipeline correctly reports PARTIAL_SUCCESS when
individual pages have problematic VLM stop reasons (LENGTH,
CONTENT_FILTERED) or missing predictions.

Related: https://github.com/docling-project/docling/issues/2583
"""

import os
from types import SimpleNamespace

import pytest

from docling.datamodel.base_models import (
    ConversionStatus,
    Page,
    PagePredictions,
    VlmPrediction,
    VlmStopReason,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.pipeline.vlm_pipeline import VlmPipeline

pytestmark = pytest.mark.skipif(
    os.getenv("CI"),
    reason="Skipping VLM unit tests in CI",
)


class _AlwaysValidBackend:
    def is_valid(self) -> bool:
        return True


def _make_page(page_no: int, stop_reason: VlmStopReason) -> Page:
    """Create a Page with a VLM prediction using the given stop reason."""
    page = Page(page_no=page_no)
    page.predictions = PagePredictions(
        vlm_response=VlmPrediction(
            text="some output",
            stop_reason=stop_reason,
        )
    )
    page._backend = _AlwaysValidBackend()
    return page


def _make_page_no_vlm(page_no: int) -> Page:
    """Create a Page with no VLM prediction."""
    page = Page(page_no=page_no)
    page.predictions = PagePredictions(vlm_response=None)
    page._backend = _AlwaysValidBackend()
    return page


def _make_conv_res(pages: list) -> ConversionResult:
    """Create a minimal ConversionResult with the given pages."""
    return SimpleNamespace(
        pages=pages,
        errors=[],
        status=ConversionStatus.STARTED,
        input=SimpleNamespace(_backend=None),
    )


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
    assert "Page 2" in conv_res.errors[0].error_message


def test_page_content_filtered(pipeline: VlmPipeline) -> None:
    """A page with CONTENT_FILTERED stop reason should yield PARTIAL_SUCCESS."""
    pages = [
        _make_page(1, VlmStopReason.CONTENT_FILTERED),
    ]
    conv_res = _make_conv_res(pages)
    status = pipeline._determine_status(conv_res)
    assert status == ConversionStatus.PARTIAL_SUCCESS
    assert len(conv_res.errors) == 1
    assert "content_filter" in conv_res.errors[0].error_message


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
    assert "no VLM prediction" in conv_res.errors[0].error_message


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
    assert "Page 1" in conv_res.errors[0].error_message
    assert "Page 2" in conv_res.errors[1].error_message
