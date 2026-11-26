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
    DocumentExtractionResult,
    ImageContentItem,
    TextContentItem,
)
from docling.datamodel.extraction_options import (
    NU_EXTRACT_2B_TRANSFORMERS,
    ChannelSelection,
)
from docling.datamodel.settings import DEFAULT_PAGE_RANGE
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

    def process(self, requests, target):
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
    ext_res = DocumentExtractionResult(input=in_doc)

    pipeline._extract_data(ext_res, target="{}")

    assert [p.scope.page_no for p in ext_res.items] == [1, 2]
    assert len(model.content_requests) == 2
    assert all(isinstance(req[0], ImageContentItem) for req in model.content_requests)
    assert not model.image_calls


def test_dclx_image_and_text_builds_content_array(dclx_two_pages: Path) -> None:
    pipeline = _pipeline_shell(ChannelSelection.IMAGE_AND_TEXT)
    model = _StubModel()
    pipeline.vlm_model = cast(BaseVlmModel, model)
    in_doc = _input(dclx_two_pages)
    ext_res = DocumentExtractionResult(input=in_doc)

    pipeline._extract_data(ext_res, target="{}")

    assert [p.scope.page_no for p in ext_res.items] == [1, 2]
    assert len(model.content_requests) == 2
    first = model.content_requests[0]
    assert isinstance(first[0], ImageContentItem)
    assert isinstance(first[1], TextContentItem)
    assert first[1].text == "Invoice total 42"


def test_dclx_status_success(dclx_two_pages: Path) -> None:
    pipeline = _pipeline_shell(ChannelSelection.IMAGE)
    pipeline.vlm_model = cast(BaseVlmModel, _StubModel())
    in_doc = _input(dclx_two_pages)
    ext_res = DocumentExtractionResult(input=in_doc)
    pipeline._extract_data(ext_res, target="{}")
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
    pipeline = _pipeline_shell(ChannelSelection.IMAGE)
    pipeline.vlm_model = cast(BaseVlmModel, _StubModel())
    result = pipeline.execute(_input(path), raises_on_error=False, template="{}")
    assert result.status == ConversionStatus.FAILURE
    assert result.pages[0].page_no == 1
    assert "no restored image" in result.pages[0].errors[0]


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


@pytest.mark.parametrize("max_size", [None, 16])
def test_borrowed_dclx_images_stay_open_and_resized_copies_close(
    dclx_two_pages, max_size
):
    pipeline = _pipeline_shell(ChannelSelection.IMAGE)
    pipeline.pipeline_options.vlm_options = NU_EXTRACT_2B_TRANSFORMERS.model_copy(
        update={"max_size": max_size}
    )
    model = _StubModel()
    pipeline.vlm_model = cast(BaseVlmModel, model)
    in_doc = _input(dclx_two_pages)
    doc = in_doc._backend.convert()
    borrowed = [page.image.pil_image for page in doc.pages.values()]
    result = DocumentExtractionResult(input=in_doc)
    pipeline._extract_data(result, target="{}")
    assert pipeline._determine_status(result) == ConversionStatus.SUCCESS
    assert [image.getpixel((0, 0)) for image in borrowed] == [
        (10, 120, 10),
        (10, 10, 120),
    ]
    used = [request[0].image for request in model.content_requests]
    if max_size is None:
        assert used == borrowed
    else:
        assert all(max(image.size) <= max_size for image in used)
        for image in used:
            with pytest.raises(ValueError, match="closed image"):
                image.getpixel((0, 0))
    in_doc._backend.unload()
