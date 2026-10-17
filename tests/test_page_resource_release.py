# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the per-page resources released once a page leaves the assemble stage.

`keep_images` is a document-wide flag, so a single figure anywhere in the document
would otherwise pin every page's rendered image until `_assemble_document` runs. The
methods under test decide per page, and are exercised directly: they read only
pipeline options, so no model needs to be loaded.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel

from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
    FigureElement,
    Page,
    Table,
    TextElement,
)
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline, ThreadedItem


def _cluster(label: DocItemLabel) -> Cluster:
    return Cluster(
        id=0,
        label=label,
        bbox=BoundingBox(l=0, t=0, r=10, b=10, coord_origin=CoordOrigin.TOPLEFT),
    )


def _text_element() -> TextElement:
    return TextElement(
        label=DocItemLabel.TEXT,
        id=0,
        page_no=1,
        cluster=_cluster(DocItemLabel.TEXT),
        text="body",
    )


def _figure_element() -> FigureElement:
    return FigureElement(
        label=DocItemLabel.PICTURE,
        id=1,
        page_no=1,
        cluster=_cluster(DocItemLabel.PICTURE),
    )


def _table_element() -> Table:
    return Table(
        label=DocItemLabel.TABLE,
        id=2,
        page_no=1,
        cluster=_cluster(DocItemLabel.TABLE),
        otsl_seq=[],
        table_cells=[],
    )


def _page(elements=None, assembled: bool = True) -> Page:
    page = Page(page_no=1)
    page._image_cache = {1.0: object()}
    if assembled:
        page.assembled = AssembledUnit(elements=list(elements or []))
    return page


def _pipeline(**opts) -> SimpleNamespace:
    """A stand-in carrying just what the two methods read."""
    options = PdfPipelineOptions(**opts)
    keep_images = (
        options.generate_page_images
        or options.generate_picture_images
        or options.generate_table_images
    )
    pipeline = SimpleNamespace(
        pipeline_options=options,
        keep_images=keep_images,
        keep_backend=False,
    )
    # `_release_page_resources` calls back into the per-page decision.
    pipeline._page_image_has_consumer = lambda page: (
        StandardPdfPipeline._page_image_has_consumer(pipeline, page)
    )
    return pipeline


def _has_consumer(pipeline, page: Page) -> bool:
    return StandardPdfPipeline._page_image_has_consumer(pipeline, page)


def _release(pipeline, page: Page) -> None:
    # `conv_res` rides along the envelope but is untouched by the release step.
    item = ThreadedItem(
        payload=page, run_id=0, page_no=page.page_no, conv_res=cast(Any, None)
    )
    StandardPdfPipeline._release_page_resources(pipeline, item)


def test_page_without_a_figure_has_no_consumer():
    # The case that matters: picture crops are requested, but this page carries none.
    pipeline = _pipeline(generate_picture_images=True)
    assert _has_consumer(pipeline, _page([_text_element()])) is False


def test_page_with_a_figure_keeps_its_image():
    pipeline = _pipeline(generate_picture_images=True)
    assert _has_consumer(pipeline, _page([_text_element(), _figure_element()])) is True


@pytest.mark.filterwarnings("ignore:This field is deprecated:DeprecationWarning")
def test_table_image_request_only_keeps_pages_holding_a_table():
    pipeline = _pipeline(generate_table_images=True)
    assert _has_consumer(pipeline, _page([_table_element()])) is True
    assert _has_consumer(pipeline, _page([_text_element()])) is False


def test_a_figure_does_not_pin_a_page_that_only_holds_a_table():
    # Cross-check that the element type is matched against the option that asks for it.
    pipeline = _pipeline(generate_picture_images=True)
    assert _has_consumer(pipeline, _page([_table_element()])) is False


@pytest.mark.filterwarnings("ignore:This field is deprecated:DeprecationWarning")
def test_a_table_does_not_pin_a_page_that_only_holds_a_figure():
    pipeline = _pipeline(generate_table_images=True)
    assert _has_consumer(pipeline, _page([_figure_element()])) is False


def test_page_images_in_the_output_keep_every_page():
    pipeline = _pipeline(generate_page_images=True)
    assert _has_consumer(pipeline, _page([_text_element()])) is True


def test_unassembled_page_is_kept():
    # Nothing has been assembled yet, so what the page will hold is unknown.
    pipeline = _pipeline(generate_picture_images=True)
    assert _has_consumer(pipeline, _page(assembled=False)) is True


@pytest.mark.parametrize("elements", [[], [_text_element()]])
def test_release_drops_the_image_of_a_page_without_a_figure(elements):
    pipeline = _pipeline(generate_picture_images=True)
    page = _page(elements)

    _release(pipeline, page)

    assert page._image_cache == {}


def test_release_keeps_the_image_of_a_page_with_a_figure():
    pipeline = _pipeline(generate_picture_images=True)
    page = _page([_figure_element()])

    _release(pipeline, page)

    assert page._image_cache != {}


def test_release_drops_every_image_when_no_crops_are_requested():
    # keep_images is false outright: the per-page question never arises.
    pipeline = _pipeline()
    page = _page([_figure_element()])

    _release(pipeline, page)

    assert page._image_cache == {}
