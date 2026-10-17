# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for video and audio objects on OpenDocument presentation slides."""

from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument
from PIL import Image

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

pytest.importorskip("odfdo")
from odfdo import Document as OdfDocument, DrawPage, Element, Frame


def _plugin(href: str, mime_type: str) -> Element:
    """Return a media object the way LibreOffice Impress writes it."""
    return Element.from_tag(
        f'<draw:plugin xlink:href="{href}" xlink:type="simple" xlink:show="embed"'
        f' xlink:actuate="onLoad" draw:mime-type="{mime_type}">'
        '<draw:param draw:name="Loop" draw:value="false"/></draw:plugin>'
    )


def _preview(odf: OdfDocument, tmp_path: Path) -> str:
    image = tmp_path / "preview.png"
    Image.new("RGB", (32, 18), "blue").save(image)
    return odf.add_file(str(image))


def _convert(odf: OdfDocument, frame: Frame, path: Path) -> DoclingDocument:
    odf.body.clear()
    page = DrawPage("page1", name="Slide One")
    page.append(frame)
    odf.body.append(page)
    odf.save(str(path))
    converter = DocumentConverter(allowed_formats=[InputFormat.ODP])
    return converter.convert(path).document


def test_odp_embedded_video_records_its_media_path(tmp_path: Path):
    """The preview of an embedded video records where the video is."""
    odf = OdfDocument("presentation")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not decoded by the backend")
    media_path = odf.add_file(str(video))
    frame = Frame.image_frame(_preview(odf, tmp_path), size=("8cm", "4.5cm"))
    frame.insert(_plugin(media_path, "video/mp4"), position=0)
    doc = _convert(odf, frame, tmp_path / "video.odp")

    assert len(doc.pictures) == 1
    picture = doc.pictures[0]
    assert picture.image is not None
    assert picture.meta is not None
    assert picture.meta.get_custom_part() == {"docling__video": media_path}


def test_odp_generic_media_type_falls_back_to_the_extension(tmp_path: Path):
    """LibreOffice's generic media type still gives a kind, from the file name.

    A linked file is recorded by its link, and a frame without a preview still
    gets a picture, so the media keeps its place on the slide.
    """
    odf = OdfDocument("presentation")
    frame = Frame(size=("2cm", "2cm"), position=("1cm", "1cm"))
    frame.append(_plugin("../sounds/tone.mp3", "application/vnd.sun.star.media"))
    doc = _convert(odf, frame, tmp_path / "audio.odp")

    assert len(doc.pictures) == 1
    assert doc.pictures[0].image is None
    assert doc.pictures[0].meta is not None
    assert doc.pictures[0].meta.get_custom_part() == {
        "docling__audio": "../sounds/tone.mp3"
    }


def test_odp_plugin_that_is_not_media_is_ignored(tmp_path: Path):
    """A plugin that is neither video nor audio adds nothing."""
    odf = OdfDocument("presentation")
    frame = Frame.image_frame(_preview(odf, tmp_path), size=("8cm", "4.5cm"))
    frame.append(_plugin("Objects/applet.swf", "application/x-shockwave-flash"))
    doc = _convert(odf, frame, tmp_path / "plugin.odp")

    assert len(doc.pictures) == 1
    assert doc.pictures[0].meta is None
