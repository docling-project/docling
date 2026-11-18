# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""DocLang responses in VlmPipeline are mapped from the DocLang grid to the page."""

from pathlib import PurePath
from types import SimpleNamespace

import pytest
from docling_core.transforms.deserializer.doclang import DocLangDocDeserializer
from docling_core.transforms.serializer.doclang import (
    DocLangDocSerializer,
    DocLangParams,
)
from docling_core.types.doc import CoordOrigin, PictureItem, Size, TextItem
from PIL import Image as PILImage

from docling.datamodel.base_models import Page, PagePredictions, VlmPrediction
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.pipeline.vlm_pipeline import VlmPipeline

_PAGE = Size(width=612, height=792)

# A picture over the bottom-right quarter and a text line at the top-left.
_DOCLANG = (
    '<doclang version="0.7">'
    "<picture>"
    '<location value="256"/><location value="256"/>'
    '<location value="512"/><location value="512"/>'
    "</picture>"
    "<text>"
    '<location value="0"/><location value="0"/>'
    '<location value="128"/><location value="64"/>'
    "Hello</text>"
    "</doclang>"
)


def _pipeline(images_scale: float) -> VlmPipeline:
    pipeline = VlmPipeline.__new__(VlmPipeline)
    pipeline.pipeline_options = SimpleNamespace(
        generate_page_images=False,
        generate_picture_images=True,
        images_scale=images_scale,
        vlm_options=InlineVlmOptions(
            prompt="",
            repo_id="test",
            response_format=ResponseFormat.DOCLANG,
            inference_framework=InferenceFramework.TRANSFORMERS,
            scale=images_scale,
        ),
    )
    pipeline.force_backend_text = False
    return pipeline


def _page(images_scale: float) -> Page:
    page = Page(
        page_no=1,
        size=_PAGE,
        predictions=PagePredictions(vlm_response=VlmPrediction(text=_DOCLANG)),
    )
    image = PILImage.new(
        "RGB",
        (round(_PAGE.width * images_scale), round(_PAGE.height * images_scale)),
        "white",
    )
    page._image_cache = {images_scale: image}
    page._default_image_scale = images_scale
    return page


@pytest.mark.parametrize("images_scale", [1.0, 2.0])
def test_doclang_locations_are_scaled_to_the_page(images_scale: float) -> None:
    pipeline = _pipeline(images_scale)
    conv_res = SimpleNamespace(
        input=SimpleNamespace(file=PurePath("test.pdf")), errors=[]
    )
    page = _page(images_scale)
    assert page.image is not None
    page.image.paste(
        "red",
        (
            round(306 * images_scale),
            round(396 * images_scale),
            page.image.width,
            page.image.height,
        ),
    )

    doc = pipeline._finalize_page_document(conv_res, page)

    assert not conv_res.errors
    assert doc.pages[1].size == _PAGE

    picture = doc.pictures[0]
    assert isinstance(picture, PictureItem)
    bbox = picture.prov[0].bbox
    assert bbox.coord_origin == CoordOrigin.TOPLEFT
    assert (bbox.l, bbox.t, bbox.r, bbox.b) == pytest.approx((306, 396, 612, 792))

    # The crop is the bottom-right quarter of the page image.
    assert picture.image is not None
    assert picture.image.size == Size(
        width=306 * images_scale, height=396 * images_scale
    )
    crop = picture.image.pil_image
    assert crop is not None
    assert crop.getcolors() == [(crop.width * crop.height, (255, 0, 0))]

    text = next(item for item in doc.texts if isinstance(item, TextItem))
    bbox = text.prov[0].bbox
    assert (bbox.l, bbox.t, bbox.r, bbox.b) == pytest.approx((0, 0, 153, 99))


@pytest.mark.parametrize("grid", [None, Size(width=1024, height=2048)])
@pytest.mark.parametrize("rich_cell", [False, True])
@pytest.mark.parametrize("table_location", [False, True])
def test_doclang_table_cells_keep_their_page_locations(
    grid: Size | None, rich_cell: bool, table_location: bool
) -> None:
    resolution = (
        f'<default_resolution width="{int(grid.width)}" height="{int(grid.height)}"/>'
        if grid is not None
        else ""
    )
    width, height = (512, 512) if grid is None else grid.as_tuple()
    locations = "".join(
        f'<location value="{int(value)}"/>'
        for value in (width / 4, height / 4, width * 3 / 4, height * 3 / 4)
    )
    cell = f"<text>{locations}Cell</text>" if rich_cell else "Cell"
    table_head = locations if table_location else ""
    page = _page(1.0)
    assert page.predictions.vlm_response is not None
    page.predictions.vlm_response.text = (
        f'<doclang version="0.7">{resolution}'
        f"<table>{table_head}<fcel/>{locations}{cell}<ecel/><nl/></table></doclang>"
    )
    conv_res = SimpleNamespace(
        input=SimpleNamespace(file=PurePath("test.pdf")), errors=[]
    )

    doc = _pipeline(1.0)._finalize_page_document(conv_res, page)

    assert not conv_res.errors
    assert doc.pages[1].size == _PAGE
    table = doc.tables[0]
    bbox = table.data.table_cells[0].bbox
    assert bbox is not None
    assert bbox.as_tuple() == pytest.approx((153, 198, 459, 594))
    assert table.data.table_cells[1].bbox is None
    if rich_cell:
        assert doc.texts[0].prov[0].bbox == bbox

    if table_location:
        assert table.prov[0].bbox == bbox
        exported = (
            DocLangDocSerializer(
                doc=doc, params=DocLangParams(add_table_cell_location=True)
            )
            .serialize()
            .text
        )
        reloaded = DocLangDocDeserializer().deserialize_str(exported)
        table = reloaded.tables[0]
        assert table.prov[0].bbox.as_tuple() == pytest.approx((128, 128, 384, 384))
        assert table.data.table_cells[0].bbox == table.prov[0].bbox
