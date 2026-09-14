# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast

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
from docling.models.base_model import BaseVlmModel
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
    pipeline.vlm_model = cast(BaseVlmModel, model)
    in_doc = _input(dclx_two_pages)
    ext_res = ExtractionResult(input=in_doc)

    pipeline._extract_per_page(ext_res, prompt="{}", include_text=False)

    assert [p.page_no for p in ext_res.pages] == [1, 2]
    assert len(model.image_calls) == 2
    assert not model.content_requests


def test_dclx_image_and_text_builds_content_array(dclx_two_pages: Path) -> None:
    pipeline = _pipeline_shell(ChannelSelection.IMAGE_AND_TEXT)
    model = _StubModel()
    pipeline.vlm_model = cast(BaseVlmModel, model)
    in_doc = _input(dclx_two_pages)
    ext_res = ExtractionResult(input=in_doc)

    pipeline._extract_per_page(ext_res, prompt="{}", include_text=True)

    assert [p.page_no for p in ext_res.pages] == [1, 2]
    assert len(model.content_requests) == 2
    first = model.content_requests[0]
    assert isinstance(first[0], ImageContentItem)
    assert isinstance(first[1], TextContentItem)
    assert first[1].text == "Invoice total 42"


def test_dclx_status_success(dclx_two_pages: Path) -> None:
    pipeline = _pipeline_shell(ChannelSelection.IMAGE)
    pipeline.vlm_model = cast(BaseVlmModel, _StubModel())
    in_doc = _input(dclx_two_pages)
    ext_res = ExtractionResult(input=in_doc)
    pipeline._extract_per_page(ext_res, prompt="{}", include_text=False)
    assert pipeline._determine_status(ext_res) == ConversionStatus.SUCCESS


def test_dclx_without_page_images_uses_text(tmp_path: Path) -> None:
    doc = DoclingDocument(name="text_only")
    doc.pages[1] = PageItem(page_no=1, size=Size(width=64, height=48))
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Invoice total 42",
        prov=ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=0, t=0, r=64, b=48),
            charspan=(0, 16),
        ),
    )
    path = tmp_path / "text-only.dclx"
    doc.save_as_doclang_archive(path)

    assert (
        _pipeline_shell(ChannelSelection.AUTO)._resolve_channel(_input(path))
        == ChannelSelection.TEXT
    )
    with pytest.raises(ValueError, match="does not offer page images"):
        _pipeline_shell(ChannelSelection.IMAGE)._resolve_channel(_input(path))


def test_pipeline_unloads_streamed_dclx_backend(dclx_two_pages: Path) -> None:
    in_doc = InputDocument(
        path_or_stream=BytesIO(dclx_two_pages.read_bytes()),
        format=InputFormat.DCLX,
        backend=DocLangArchiveBackend,
        filename="fixture.dclx",
    )
    backend = in_doc._backend
    assert isinstance(backend, DocLangArchiveBackend)
    assert backend._temp_dir is not None

    pipeline = _pipeline_shell(ChannelSelection.IMAGE)
    pipeline.vlm_model = cast(BaseVlmModel, _StubModel())
    result = pipeline.execute(in_doc, raises_on_error=False, template="{}")

    assert result.status == ConversionStatus.SUCCESS
    assert backend._temp_dir is None
