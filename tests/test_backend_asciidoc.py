import glob
from io import BytesIO
from pathlib import Path

from docling.backend.asciidoc_backend import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    AsciiDocBackend,
)
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

    picture = doc.pictures[0]
    assert picture.image.size.width == DEFAULT_IMAGE_WIDTH
    assert picture.image.size.height == DEFAULT_IMAGE_HEIGHT


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
