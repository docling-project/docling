# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Warn when rendered page text has no ink because the host has no matching font."""

from __future__ import annotations

import logging
from pathlib import Path

from docling_core.types.doc import BoundingBox, CoordOrigin
from PIL import Image, ImageDraw

from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.datamodel.settings import DocumentLimits

_FIXTURE = Path("tests/data/pdf/invisible_text_layer.pdf")
_LOGGER = "docling.backend.docling_parse_backend"
_MARKER = "DOCLING_PARSE_FALLBACK_FONT"


def _load_page():
    document = InputDocument(
        path_or_stream=_FIXTURE,
        format=InputFormat.PDF,
        backend=ThreadedDoclingParseDocumentBackend,
        limits=DocumentLimits(page_range=(1, 1)),
    )
    backend = document._backend
    page = next(iter(backend.iter_pages()))
    return backend, page


def _use_image(page, image: Image.Image) -> None:
    def get_image(*, scale=1.0, cropbox=None, canvas_size=None):
        del canvas_size
        factor = 1.0 if scale is None else float(scale)
        if cropbox is None:
            resized = image.resize(
                (
                    max(1, round(image.width * factor)),
                    max(1, round(image.height * factor)),
                )
            )
            return resized
        return Image.new(
            "RGB",
            (
                max(1, round(abs(cropbox.width) * factor)),
                max(1, round(abs(cropbox.height) * factor)),
            ),
            "white",
        )

    page._result.get_image = get_image


def _blank_page(page) -> Image.Image:
    size = page.get_size()
    return Image.new(
        "RGB", (max(1, round(size.width)), max(1, round(size.height))), "white"
    )


def _page_with_ink_in_text_cells(page) -> Image.Image:
    image = _blank_page(page)
    draw = ImageDraw.Draw(image)
    for cell in page.get_visible_text_cells() or []:
        box = cell.rect.to_bounding_box()
        draw.rectangle((box.l, box.t, box.r - 1, box.b - 1), fill="black")
    return image


def _font_warnings(caplog) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == _LOGGER and _MARKER in record.message
    ]


def test_unpainted_text_warns_once(caplog) -> None:
    backend, page = _load_page()
    try:
        _use_image(page, _blank_page(page))
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            page.get_page_image()
            page.get_page_image()
        assert len(_font_warnings(caplog)) == 1
    finally:
        backend.unload()


def test_ink_inside_text_cells_does_not_warn(caplog) -> None:
    backend, page = _load_page()
    try:
        _use_image(page, _page_with_ink_in_text_cells(page))
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            page.get_page_image()
        assert _font_warnings(caplog) == []
    finally:
        backend.unload()


def test_a_crop_does_not_warn_before_the_full_page(caplog) -> None:
    backend, page = _load_page()
    try:
        _use_image(page, _blank_page(page))
        crop = BoundingBox(l=0, t=0, r=10, b=10, coord_origin=CoordOrigin.TOPLEFT)
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            page.get_page_image(cropbox=crop)
            assert _font_warnings(caplog) == []
            page.get_page_image()
        assert len(_font_warnings(caplog)) == 1
    finally:
        backend.unload()
