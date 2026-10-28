# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for JATS tables that are published as an image instead of a table."""

from pathlib import Path

from PIL import Image

from docling.datamodel.backend_options import JatsBackendOptions

from .test_backend_jats import (
    convert_jats_body,
    get_converter,
    get_pictures,
    write_jats_body,
)


def test_jats_table_published_as_image_keeps_its_caption():
    # PMC often ships a table as a <graphic> instead of a <table>. The table
    # must not disappear: it becomes a picture that keeps its label and caption.
    doc = convert_jats_body(
        "<sec><title>Methods</title>"
        "<p>The items are listed in Table 1.</p>"
        '<table-wrap id="T1"><label>Table 1</label>'
        "<caption><p>Items of data extraction.</p></caption>"
        '<graphic xmlns:xlink="http://www.w3.org/1999/xlink" '
        'xlink:href="table1.jpg"/></table-wrap>'
        "<p>Text after the table.</p></sec>"
    )

    assert doc.tables == []
    pictures = get_pictures(doc)
    assert len(pictures) == 1
    assert pictures[0].image is None
    caption = pictures[0].captions[0].resolve(doc)
    assert caption.text == "Table 1 Items of data extraction."
    assert "Table 1 Items of data extraction." in doc.export_to_markdown()


def test_jats_table_published_as_image_embeds_image_when_enabled(tmp_path: Path):
    image_path = tmp_path / "table1.png"
    Image.new("RGB", (7, 5), color=(0, 0, 255)).save(image_path)
    jats_path = tmp_path / "article.nxml"
    write_jats_body(
        jats_path,
        '<table-wrap id="T1"><label>Table 1</label>'
        "<caption><p>A blue table.</p></caption>"
        '<graphic xlink:href="table1.png"/></table-wrap>',
    )

    doc = (
        get_converter(JatsBackendOptions(fetch_images=True, enable_local_fetch=True))
        .convert(jats_path)
        .document
    )

    pictures = get_pictures(doc)
    assert len(pictures) == 1
    image = pictures[0].get_image(doc)
    assert image is not None
    assert image.getpixel((0, 0)) == (0, 0, 255)
