# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import glob
from io import BytesIO
from pathlib import Path

from docling_core.types.doc import CodeItem, DocItemLabel, ImageRefMode, ListItem

from docling.backend.asciidoc_backend import AsciiDocBackend
from docling.datamodel.backend_options import AsciiDocBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export


def _get_backend(fname):
    in_doc = InputDocument(
        path_or_stream=fname,
        format=InputFormat.ASCIIDOC,
        backend=AsciiDocBackend,
    )

    doc_backend = in_doc._backend
    return doc_backend


def test_list_dedent_to_base_does_not_crash():
    # A list that starts indented and then dedents back to the base level used
    # to raise "TypeError: '<' not supported between instances of 'int' and
    # 'NoneType'": the dedent loop walked past level 0, where the base indent is
    # never set. It should keep both items instead.
    src = b"  * a\n* b\n"
    in_doc = InputDocument(
        path_or_stream=BytesIO(src),
        format=InputFormat.ASCIIDOC,
        backend=AsciiDocBackend,
        filename="dedent.asciidoc",
    )
    doc = in_doc._backend.convert()

    assert [item.text for item in doc.texts] == ["a", "b"]


def test_auto_numbered_list_keeps_items_and_following_text():
    source = b"""= Installation Guide

== Steps

. Download the archive
. Unpack it
. Run the installer

== Troubleshooting

If the installer fails, check the log file.
"""
    in_doc = InputDocument(
        path_or_stream=BytesIO(source),
        format=InputFormat.ASCIIDOC,
        backend=AsciiDocBackend,
        filename="ordered-list.adoc",
    )
    doc = in_doc._backend.convert()

    list_items = [item for item in doc.texts if isinstance(item, ListItem)]
    assert [item.text for item in list_items] == [
        "Download the archive",
        "Unpack it",
        "Run the installer",
    ]
    assert all(item.enumerated for item in list_items)
    assert "If the installer fails, check the log file." in doc.export_to_markdown()


def test_literal_block_keeps_its_content_and_following_text():
    source = b"""= Guide

== One

Before the block.

....
raw literal
second line
....

== Two

After the block.
"""
    in_doc = InputDocument(
        path_or_stream=BytesIO(source),
        format=InputFormat.ASCIIDOC,
        backend=AsciiDocBackend,
        filename="literal-block.adoc",
    )
    doc = in_doc._backend.convert()

    code_items = [item for item in doc.texts if isinstance(item, CodeItem)]
    assert [item.text for item in code_items] == ["raw literal\nsecond line"]
    assert "After the block." in doc.export_to_markdown()


def test_literal_block_flushes_pending_caption():
    source = b""".Literal example
....
raw literal
....

image::next.png[]
"""
    in_doc = InputDocument(
        path_or_stream=BytesIO(source),
        format=InputFormat.ASCIIDOC,
        backend=AsciiDocBackend,
        filename="captioned-literal-block.adoc",
    )
    doc = in_doc._backend.convert()

    assert [(item.label, item.text) for item in doc.texts[:2]] == [
        (DocItemLabel.CAPTION, "Literal example"),
        (DocItemLabel.CODE, "raw literal"),
    ]


def test_parse_picture():
    line = (
        "image::images/example1.png[Example Image, width=200, height=150, align=center]"
    )
    res = AsciiDocBackend._parse_picture(line)
    assert res
    assert res.get("width", 0) == "200"
    assert res.get("height", 0) == "150"
    assert res.get("uri", "") == "images/example1.png"

    line = "image::renamed-bookmark.png[Renamed bookmark]"
    res = AsciiDocBackend._parse_picture(line)
    assert res
    assert "width" not in res
    assert "height" not in res
    assert res.get("uri", "") == "renamed-bookmark.png"

    line = "image::images/screenshot.png[A screenshot showing a dialog box, containing text fields, buttons, and validation errors, width=604, height=422]"
    res = AsciiDocBackend._parse_picture(line)
    assert res
    assert res.get("width", 0) == "604"
    assert res.get("height", 0) == "422"
    assert res.get("uri", "") == "images/screenshot.png"
    assert (
        res.get("alt", "")
        == "A screenshot showing a dialog box, containing text fields, buttons, and validation errors"
    )


def test_table_cell_format_specifiers():
    # A header row whose cells carry alignment + style specifiers ("^.^h|")
    # must still be detected as a table line and parsed into clean cells.
    line = "^.^h|Field               ^.^h| Description"
    assert AsciiDocBackend._is_table_line(line)
    assert AsciiDocBackend._parse_table_line(line) == ["Field", "Description"]

    # A column-spanning specifier ("2+^|") is dropped from the cell text.
    assert AsciiDocBackend._parse_table_line("2+^|Spanned ^|Next") == [
        "Spanned",
        "Next",
    ]


def test_table_cell_content_preserved():
    # Single-letter cells that coincide with style operators (s, h, m, ...) and
    # words ending in one (Eth) must not be mistaken for cell specifiers.
    assert AsciiDocBackend._parse_table_line("| s | Strong") == ["s", "Strong"]
    assert AsciiDocBackend._parse_table_line("| eth | Eth | Ethernet") == [
        "eth",
        "Eth",
        "Ethernet",
    ]


def test_empty_table_does_not_crash():
    # An empty table must yield an empty grid rather than raising.
    data = AsciiDocBackend._populate_table_as_grid([])
    assert data.num_rows == 0
    assert data.num_cols == 0


def test_non_numeric_image_dimensions_do_not_crash():
    # image width/height can be non-numeric in real AsciiDoc (e.g. "50%", "auto").
    # convert() used int(item["width"]) directly, so such an image raised
    # ValueError and failed the whole document; it must fall back to the default
    # size and keep converting the rest of the content.
    from io import BytesIO

    adoc = (
        b"= Title\n\n"
        b"Intro text.\n\n"
        b"image::diagram.png[Architecture, width=50%, height=auto]\n\n"
        b"Text after the image.\n"
    )
    in_doc = InputDocument(
        path_or_stream=BytesIO(adoc),
        format=InputFormat.ASCIIDOC,
        backend=AsciiDocBackend,
        filename="dims.adoc",
    )
    doc = in_doc._backend.convert()

    md = doc.export_to_markdown()
    assert "Intro text." in md
    assert "Text after the image." in md

    assert doc.pictures[0].image is None


def test_local_images_are_embedded_and_missing_images_do_not_break_export(tmp_path):
    in_path = Path("tests/data/asciidoc/sources/asciidoc_03.asciidoc")
    options = AsciiDocBackendOptions(
        fetch_images=True,
        enable_local_fetch=True,
        source_uri=in_path,
    )
    in_doc = InputDocument(
        path_or_stream=in_path,
        format=InputFormat.ASCIIDOC,
        backend=AsciiDocBackend,
        backend_options=options,
    )
    doc = in_doc._backend.convert()

    assert doc.pictures[0].image is not None
    assert doc.pictures[0].image.uri.scheme == "data"
    assert doc.pictures[1].image is None
    assert doc.pictures[2].image is None

    output_path = tmp_path / "asciidoc_03.md"
    doc.save_as_markdown(output_path, image_mode=ImageRefMode.EMBEDDED)
    assert "data:image/png;base64," in output_path.read_text(encoding="utf-8")


def test_images_not_fetched_when_fetch_images_is_false() -> None:
    """Images are not loaded when fetch_images is False (the default).

    No image data should be loaded even when the source file is on disk and
    enable_local_fetch would otherwise allow it.
    """
    in_path = Path("tests/data/asciidoc/sources/asciidoc_03.asciidoc")
    doc = _get_backend(in_path).convert()

    assert all(pic.image is None for pic in doc.pictures)


def test_asciidocs_examples():
    fnames = sorted(glob.glob("./tests/data/asciidoc/sources/*.asciidoc"))

    for fname in fnames:
        in_path = Path(fname)
        gt_path = Path("./tests/data/asciidoc/groundtruth/") / f"{in_path.name}"

        doc_backend = _get_backend(in_path)
        doc = doc_backend.convert()

        pred_md = doc.export_to_markdown(compact_tables=True)

        # Verify markdown export
        assert verify_export(pred_md, str(gt_path) + ".md", generate=GEN_TEST_DATA)


def test_utf8_bom_does_not_hide_the_document_title(tmp_path):
    """A leading UTF-8 BOM must not survive into the first line.

    Decoding with plain utf-8 kept it, so "= Title" started with U+FEFF, was no
    longer recognized as the document title, and the BOM reached the exported
    text. Both the stream and the file path are covered, since each decodes
    separately.
    """
    adoc_bytes = "\ufeff= Document Title\n\nSome body text.\n".encode()

    in_doc = InputDocument(
        path_or_stream=BytesIO(adoc_bytes),
        format=InputFormat.ASCIIDOC,
        backend=AsciiDocBackend,
        filename="bom.adoc",
    )
    stream_doc = in_doc._backend.convert()

    adoc_file = tmp_path / "bom.adoc"
    adoc_file.write_bytes(adoc_bytes)
    file_doc = _get_backend(adoc_file).convert()

    for doc in (stream_doc, file_doc):
        assert doc.texts[0].label == "title"
        assert doc.texts[0].text == "Document Title"
