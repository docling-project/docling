# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""DCLX extraction: structure + page images (dim 1), IMAGE and IMAGE_AND_TEXT (dim 2)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from docling_core.types.doc import BoundingBox, DoclingDocument, ProvenanceItem
from docling_core.types.doc.document import ImageRef, PageItem, Size
from docling_core.types.doc.labels import DocItemLabel
from PIL import Image as PILImage

from docling.backend.xml.doclang_archive_backend import DocLangArchiveBackend
from docling.datamodel.base_models import (
    ConversionStatus,
    InputFormat,
    VlmPrediction,
    VlmStopReason,
)
from docling.datamodel.document import InputDocument
from docling.datamodel.extraction import (
    ExtractionResult,
    ImageContentItem,
    TextContentItem,
)
from docling.datamodel.extraction_options import ChannelSelection
from docling.datamodel.settings import DEFAULT_PAGE_RANGE
from docling.datamodel.vlm_model_specs import NU_EXTRACT_2B_TRANSFORMERS
from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline


@pytest.fixture
def dclx_two_pages(tmp_path: Path) -> Path:
    """A two-page DCLX carrying page images and per-page text with provenance."""
    doc = DoclingDocument(name="dclx_fixture")
    for page_no, color, text in (
        (1, (10, 120, 10), "Invoice total 42"),
        (2, (10, 10, 120), "Due date 2026-01-01"),
    ):
        img = PILImage.new("RGB", (64, 48), color)
        doc.pages[page_no] = PageItem(
            page_no=page_no,
            size=Size(width=64, height=48),
            image=ImageRef.from_pil(img, dpi=72),
        )
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=text,
            prov=ProvenanceItem(
                page_no=page_no,
                bbox=BoundingBox(l=0, t=0, r=64, b=48),
                charspan=(0, len(text)),
            ),
        )
    out = tmp_path / "fixture.dclx"
    doc.save_as_doclang_archive(out)
    return out


def _pipeline_shell(channel: ChannelSelection) -> ExtractionVlmPipeline:
    pipeline = ExtractionVlmPipeline.__new__(ExtractionVlmPipeline)
    pipeline.pipeline_options = SimpleNamespace(  # type: ignore[assignment]
        vlm_options=NU_EXTRACT_2B_TRANSFORMERS,
        input_channels=channel,
        markdown_params=None,
        document_timeout=None,
    )
    return pipeline


def _input(path: Path) -> InputDocument:
    return InputDocument(
        path_or_stream=path, format=InputFormat.DCLX, backend=DocLangArchiveBackend
    )


class _StubModel:
    """Records requests and returns one prediction per request."""

    def __init__(self) -> None:
        self.image_calls: list = []
        self.content_requests: list = []

    def process_images(self, images, prompt):
        self.image_calls.append(list(images))
        return [VlmPrediction(text="{}", stop_reason=VlmStopReason.END_OF_SEQUENCE)]

    def process(self, requests, template):
        reqs = [list(r) for r in requests]
        self.content_requests.extend(reqs)
        return [
            VlmPrediction(text="{}", stop_reason=VlmStopReason.END_OF_SEQUENCE)
            for _ in reqs
        ]


def test_dclx_auto_is_image(dclx_two_pages: Path) -> None:
    pipeline = _pipeline_shell(ChannelSelection.AUTO)
    assert pipeline._resolve_channel(_input(dclx_two_pages)) == ChannelSelection.IMAGE


def test_dclx_accepts_image_and_text(dclx_two_pages: Path) -> None:
    pipeline = _pipeline_shell(ChannelSelection.IMAGE_AND_TEXT)
    assert (
        pipeline._resolve_channel(_input(dclx_two_pages))
        == ChannelSelection.IMAGE_AND_TEXT
    )


def test_dclx_image_extraction_yields_page_per_image(dclx_two_pages: Path) -> None:
    pipeline = _pipeline_shell(ChannelSelection.IMAGE)
    model = _StubModel()
    pipeline.vlm_model = model  # type: ignore[assignment]
    in_doc = _input(dclx_two_pages)
    ext_res = ExtractionResult(input=in_doc)

    pipeline._extract_per_page(ext_res, prompt="{}", include_text=False)

    assert [p.page_no for p in ext_res.pages] == [1, 2]
    assert len(model.image_calls) == 2  # one request per page, no batching
    assert not model.content_requests


def test_dclx_image_and_text_builds_content_array(dclx_two_pages: Path) -> None:
    pipeline = _pipeline_shell(ChannelSelection.IMAGE_AND_TEXT)
    model = _StubModel()
    pipeline.vlm_model = model  # type: ignore[assignment]
    in_doc = _input(dclx_two_pages)
    ext_res = ExtractionResult(input=in_doc)

    pipeline._extract_per_page(ext_res, prompt="{}", include_text=True)

    assert [p.page_no for p in ext_res.pages] == [1, 2]
    assert len(model.content_requests) == 2  # one request per page
    first = model.content_requests[0]
    # image-then-text ordering, one page's payload per request
    assert isinstance(first[0], ImageContentItem)
    assert isinstance(first[1], TextContentItem)
    assert first[1].text == "Invoice total 42"


def test_dclx_status_success(dclx_two_pages: Path) -> None:
    pipeline = _pipeline_shell(ChannelSelection.IMAGE)
    pipeline.vlm_model = _StubModel()  # type: ignore[assignment]
    in_doc = _input(dclx_two_pages)
    ext_res = ExtractionResult(input=in_doc)
    pipeline._extract_per_page(ext_res, prompt="{}", include_text=False)
    assert pipeline._determine_status(ext_res) == ConversionStatus.SUCCESS
