# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Docling conversion tests for the OXR input format."""

from __future__ import annotations

import json
import logging
import shutil
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from docling.datamodel.backend_options import DeclarativeBackendOptions
from docling.datamodel.base_models import (
    ConversionStatus,
    DocumentStream,
    InputFormat,
)
from docling.document_converter import DocumentConverter, OxrFormatOption

_DATA_DIR = Path(__file__).parent / "data" / "json_oxr"
_FIXTURE = _DATA_DIR / "self_authored_pages.oxr.json"


def test_oxr_suffix_selects_the_oxr_format() -> None:
    result = DocumentConverter().convert(_FIXTURE)

    assert result.status == ConversionStatus.SUCCESS
    assert result.input.format == InputFormat.OXR
    assert "Adapter Fixture" in result.document.export_to_markdown()


def test_plain_json_with_oxr_markers_selects_the_oxr_format(
    tmp_path: Path,
) -> None:
    target = tmp_path / "result.json"
    shutil.copyfile(_FIXTURE, target)

    result = DocumentConverter().convert(target)

    assert result.input.format == InputFormat.OXR
    assert result.status == ConversionStatus.SUCCESS


def test_oxr_stream_name_selects_the_oxr_format() -> None:
    stream = DocumentStream(
        name="result.oxr.json", stream=BytesIO(_FIXTURE.read_bytes())
    )

    result = DocumentConverter().convert(stream)

    assert result.input.format == InputFormat.OXR
    assert result.status == ConversionStatus.SUCCESS


def test_plain_docling_json_is_still_docling_json(tmp_path: Path) -> None:
    document = DocumentConverter().convert(_FIXTURE).document
    target = tmp_path / "docling.json"
    document.save_as_json(target)

    result = DocumentConverter().convert(target)

    assert result.input.format == InputFormat.JSON_DOCLING
    assert result.status == ConversionStatus.SUCCESS


def test_invalid_oxr_json_reports_a_conversion_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    target = tmp_path / "broken.oxr.json"
    target.write_text(
        json.dumps(
            {"code": 0, "data": {"result": {"outputs": {"json": {"pages": []}}}}}
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        result = DocumentConverter().convert(target, raises_on_error=False)

    assert result.status == ConversionStatus.FAILURE
    assert any(
        "OXR JSON contains no pages" in record.message for record in caplog.records
    )


def test_oxr_format_option_enables_local_pictures(tmp_path: Path) -> None:
    target = tmp_path / "doc.oxr.json"
    shutil.copyfile(_FIXTURE, target)
    images = tmp_path / "imgs"
    images.mkdir()
    Image.new("RGB", (4, 4), color="white").save(images / "0_6.png")

    default = DocumentConverter().convert(target).document
    assert default.pictures[0].image is None

    with_local_images = (
        DocumentConverter(
            format_options={
                InputFormat.OXR: OxrFormatOption(
                    backend_options=DeclarativeBackendOptions(enable_local_fetch=True)
                )
            }
        )
        .convert(target)
        .document
    )
    assert with_local_images.pictures[0].image is not None
