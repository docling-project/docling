import glob
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


def test_table_trailing_pipe_no_phantom_cell():
    # A "|" that terminates a row (trailing-pipe style) must not add a phantom
    # empty cell, which would inflate the table's column count.
    assert AsciiDocBackend._parse_table_line("|Header 1|Header 2|") == [
        "Header 1",
        "Header 2",
    ]
    assert AsciiDocBackend._parse_table_line("|Cell 10|Cell 11|Cell 12|") == [
        "Cell 10",
        "Cell 11",
        "Cell 12",
    ]
    # Leading and mid-row empty cells must still be preserved.
    assert AsciiDocBackend._parse_table_line("|Cell 1 | | Cell 3") == [
        "Cell 1",
        "",
        "Cell 3",
    ]
    assert AsciiDocBackend._parse_table_line("|| Cell 14 | Cell 15 |") == [
        "",
        "Cell 14",
        "Cell 15",
    ]


def test_empty_table_does_not_crash():
    # An empty table must yield an empty grid rather than raising.
    data = AsciiDocBackend._populate_table_as_grid([])
    assert data.num_rows == 0
    assert data.num_cols == 0


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
