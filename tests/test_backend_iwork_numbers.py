# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the Apple Numbers (``.numbers``) spreadsheet backend.

Test Data Attribution
---------------------
``numbers_2013.numbers`` and ``numbers_iwork09.numbers`` are
``testNumbers2013.numbers`` and ``testNumbers.numbers`` from the Apache Tika test
corpus, licensed under the Apache License 2.0. They are genuine Apple Numbers
output and between them cover both container generations: ``numbers_2013``
stores its content as ``Index/*.iwa``, while ``numbers_iwork09`` uses the iWork
'09 ``index.xml`` layout. Both hold the same two-sheet checking register, so the
two readers can be checked against each other.

See https://github.com/apache/tika (``tika-parser-apple-module`` test resources).

The cell buffers in :func:`test_version_5_cell_storage_is_decoded` were captured
from Numbers documents saved by releases newer than either fixture, whose cells
use a storage layout the fixtures never exercise.
"""

import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import (
    ContentLayer,
    GroupItem,
    GroupLabel,
    PictureClassificationLabel,
    PictureItem,
    TableItem,
    TextItem,
)

from docling.backend.iwork import tables
from docling.backend.iwork.numbers_iwa import render
from docling.backend.iwork_backend import IWorkNumbersDocumentBackend
from docling.datamodel.backend_options import IWorkBackendOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import InputDocument, _DocumentConversionInput
from docling.datamodel.settings import DocumentLimits
from docling.document_converter import DocumentConverter
from docling.exceptions import DocumentLoadError

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

SOURCES = Path("./tests/data/numbers/sources")
NUMBERS_2013 = SOURCES / "numbers_2013.numbers"
NUMBERS_IWORK09 = SOURCES / "numbers_iwork09.numbers"
GROUNDTRUTH = Path("./tests/data/numbers/groundtruth")

# Every fixture, each of which converts and so has a stored groundtruth.
CONVERTIBLE = [NUMBERS_2013, NUMBERS_IWORK09]

BOTH_GENERATIONS = pytest.mark.parametrize(
    "source", [NUMBERS_2013, NUMBERS_IWORK09], ids=["iwa", "iwork09"]
)


def _backend(
    path: Path,
    options: IWorkBackendOptions | None = None,
    limits: DocumentLimits | None = None,
) -> IWorkNumbersDocumentBackend:
    in_doc = InputDocument(
        path_or_stream=path,
        format=InputFormat.IWORK_NUMBERS,
        backend=IWorkNumbersDocumentBackend,
        backend_options=options,
        limits=limits,
    )
    backend = in_doc._backend
    assert isinstance(backend, IWorkNumbersDocumentBackend)
    return backend


def _tables(doc) -> list[TableItem]:
    return list(doc.tables)


def _grid(table: TableItem) -> dict[tuple[int, int], str]:
    return {
        (cell.start_row_offset_idx, cell.start_col_offset_idx): cell.text
        for cell in table.data.table_cells
    }


def test_detects_numbers_from_path_and_named_stream():
    """`.numbers` is a ZIP, so detection must not stop at ``application/zip``."""
    conv_input = _DocumentConversionInput(path_or_stream_iterator=[])

    assert conv_input._guess_format(NUMBERS_2013) == InputFormat.IWORK_NUMBERS

    stream = DocumentStream(
        name="budget.numbers", stream=BytesIO(NUMBERS_2013.read_bytes())
    )
    assert conv_input._guess_format(stream) == InputFormat.IWORK_NUMBERS


def test_extensionless_numbers_stream_is_not_claimed():
    """Without the extension a Numbers container is indistinguishable from Pages
    and Keynote, so the backend must not claim it rather than guess wrong."""
    conv_input = _DocumentConversionInput(path_or_stream_iterator=[])
    stream = DocumentStream(name="blob", stream=BytesIO(NUMBERS_2013.read_bytes()))

    assert conv_input._guess_format(stream) is None


@BOTH_GENERATIONS
def test_each_sheet_becomes_a_page_and_a_sheet_group(source: Path):
    """The other spreadsheet backends page and group by sheet; so does this one."""
    backend = _backend(source)
    assert backend.page_count() == 2

    doc = backend.convert()
    groups = [
        item
        for item, _ in doc.iterate_items(with_groups=True)
        if isinstance(item, GroupItem) and item.label == GroupLabel.SHEET
    ]
    assert [group.name for group in groups] == ["Checking", "Second sheet"]
    assert sorted(doc.pages) == [1, 2]


@BOTH_GENERATIONS
def test_tables_keep_their_names_geometry_and_headers(source: Path):
    """Numbers names its tables and sizes them itself, so a sheet needs no
    clustering to tell one table from the next."""
    doc = _backend(source).convert()
    tables = _tables(doc)

    assert [table.caption_text(doc) for table in tables] == [
        "Account Categories",
        "Transactions",
        "Table 1",
    ]

    categories, transactions, _ = tables
    assert (categories.data.num_rows, categories.data.num_cols) == (7, 2)
    assert (transactions.data.num_rows, transactions.data.num_cols) == (14, 6)

    # Transactions declares two header rows and one header column.
    header_rows = {
        cell.start_row_offset_idx
        for cell in transactions.data.table_cells
        if cell.column_header
    }
    assert header_rows == {0, 1}
    assert all(
        cell.start_col_offset_idx == 0
        for cell in transactions.data.table_cells
        if cell.row_header
    )


@BOTH_GENERATIONS
def test_tables_are_ordered_down_their_sheet(source: Path):
    """Numbers keeps a sheet's drawables in z-order, which is the order they
    were added rather than the order a reader meets them. The checking sheet
    stores its two tables the other way up, so leaving them alone would put the
    register above the summary it feeds."""
    doc = _backend(source).convert()

    by_sheet: dict[int, list[float]] = {}
    for table in _tables(doc):
        for prov in table.prov:
            by_sheet.setdefault(prov.page_no, []).append(prov.bbox.t)

    assert len(by_sheet[1]) == 2
    for tops in by_sheet.values():
        assert tops == sorted(tops)


@BOTH_GENERATIONS
def test_typed_cells_are_rendered_as_the_sheet_shows_them(source: Path):
    """A spreadsheet is mostly not text: dates, numbers and the cached results
    of formulas all have to come out of the cell storage."""
    doc = _backend(source).convert()
    categories, transactions, _ = _tables(doc)

    register = _grid(transactions)
    assert register[(1, 1)] == "Date"
    assert register[(2, 1)] == "2009-10-01 00:00:00"
    assert register[(2, 2)] == "Rent"
    assert register[(2, 4)] == "-775"
    # The balance column is a running total, so this is a cached formula result.
    assert register[(2, 5)] == "3875"
    # A whole number must not pick up a trailing ".0", and a fraction must not
    # pick up binary floating point noise.
    assert register[(3, 4)] == "-97.4"

    totals = _grid(categories)
    assert totals[(6, 0)] == "Total"
    assert totals[(6, 1)] == "-2575.9"


def test_both_generations_agree_on_the_spreadsheet():
    """The two fixtures are the same document saved by different Numbers
    releases, so the independent IWA and XML readers must agree on it.

    They part company on one column only: a pop-up menu cell stores the label in
    an iWork '09 document but only the menu index in a 2013+ one.
    """
    modern = _grid(_tables(_backend(NUMBERS_2013).convert())[1])
    legacy = _grid(_tables(_backend(NUMBERS_IWORK09).convert())[1])

    popup_column = 3
    shared = {key for key in modern if key[1] != popup_column}
    assert shared == {key for key in legacy if key[1] != popup_column}
    assert all(modern[key] == legacy[key] for key in shared)

    assert legacy[(2, popup_column)] == "Home"
    assert modern[(2, popup_column)] == "2"


def test_sparse_rows_land_in_the_columns_the_spreadsheet_shows():
    """An iWork '09 datasource stores only the cells a row uses and gives them no
    coordinates, so a sparse row has to be placed from the grid's occupancy
    counts. Packing it to the left instead would silently shift its values.

    The 2013+ fixture states each cell's column outright, which is what makes it
    the reference here.
    """
    modern = _grid(_tables(_backend(NUMBERS_2013).convert())[2])
    legacy = _grid(_tables(_backend(NUMBERS_IWORK09).convert())[2])

    assert modern == legacy
    # "=C3 + D3" over the two numbers, so the columns are the ones that matter.
    assert modern == {(1, 1): "Test", (2, 2): "0.5", (2, 3): "0.1", (3, 3): "0.6"}


def test_sheet_names_filter_selects_sheets():
    backend = _backend(NUMBERS_2013, IWorkBackendOptions(sheet_names=["Second sheet"]))
    assert backend.page_count() == 1

    doc = backend.convert()
    assert [table.caption_text(doc) for table in _tables(doc)] == ["Table 1"]


def test_page_range_selects_sheets():
    doc = _backend(NUMBERS_2013, limits=DocumentLimits(page_range=(2, 2))).convert()

    assert sorted(doc.pages) == [2]
    assert [table.caption_text(doc) for table in _tables(doc)] == ["Table 1"]


@pytest.mark.parametrize(
    "buffer, expected",
    [
        ("0503000000000000081002000e0000000500000001000000", "YYY_2_1"),
        (
            "050200000000000041300000d00700000000000000000000"
            "00004030020000000100000003000000",
            "2000",
        ),
        (
            "05050000000008006490000000000000d0337a4110000000130000000300000002000000",
            "2001-11-15 00:00:00",
        ),
        (
            "050700000000000042120100000000000075224102000000090000000400000002000000",
            "7 days, 0:00:00",
        ),
        ("05000000000000004000000002000000", None),
    ],
    ids=["text", "decimal128", "date", "duration", "empty"],
)
def test_version_5_cell_storage_is_decoded(buffer: str, expected: str | None):
    """Numbers changed its cell layout in 2017 and neither fixture predates that,
    so the newer layout is pinned against cells captured from documents that do
    use it: a string reference, an exact decimal128, a date and a duration.
    """
    values = tables.CellValues(strings={14: "YYY_2_1"})
    decoded = tables.cell(bytes.fromhex(buffer), 0, values)

    assert render(decoded) == expected


def _write_numbers(
    path: Path, members: dict[str, bytes], *, encrypted: bool = False
) -> Path:
    """Write a ``.numbers`` container, optionally flagged as encrypted.

    zipfile cannot write an encrypted archive, so the general-purpose flag is
    set afterwards in the central directory, which is where infolist() reads it.
    """
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)

    if encrypted:
        raw = bytearray(path.read_bytes())
        at = raw.find(b"PK\x01\x02")
        raw[at + 8] |= 0x01
        path.write_bytes(raw)

    return path


def _load(path: Path, options: IWorkBackendOptions | None = None) -> None:
    """Run the backend for its failure, which InputDocument would swallow."""
    IWorkNumbersDocumentBackend(
        InputDocument(
            path_or_stream=path,
            format=InputFormat.IWORK_NUMBERS,
            backend=IWorkNumbersDocumentBackend,
        ),
        path,
        options,
    )


def test_password_protected_numbers_is_rejected_cleanly(tmp_path: Path):
    """An encrypted container cannot be read, and the advice has to name the
    application the reader is being pointed back at."""
    protected = _write_numbers(
        tmp_path / "locked.numbers",
        {"Index/Document.iwa": b"\x00\x01\x00\x00\x00"},
        encrypted=True,
    )

    with pytest.raises(DocumentLoadError, match="password in Numbers"):
        _load(protected)


def test_zip_without_a_numbers_index_is_rejected(tmp_path: Path):
    other_zip = _write_numbers(
        tmp_path / "not_really.numbers", {"word/document.xml": b"<w:document/>"}
    )

    with pytest.raises(DocumentLoadError, match="does not look like a Numbers"):
        _load(other_zip)


def test_archive_limits_are_enforced():
    """The container is untrusted input, so its size is bounded before it is read."""
    with pytest.raises(DocumentLoadError, match="max_total_bytes"):
        _load(NUMBERS_2013, IWorkBackendOptions(max_total_bytes=1024))

    with pytest.raises(DocumentLoadError, match="max_member_count"):
        _load(NUMBERS_2013, IWorkBackendOptions(max_member_count=1))


def _chart_grid(picture: PictureItem) -> list[list[str]]:
    """Read a chart picture's attached data back as rows of text."""
    assert picture.meta is not None and picture.meta.tabular_chart is not None
    data = picture.meta.tabular_chart.chart_data
    grid = {
        (cell.start_row_offset_idx, cell.start_col_offset_idx): cell.text
        for cell in data.table_cells
    }
    return [
        [grid.get((row, col), "") for col in range(data.num_cols)]
        for row in range(data.num_rows)
    ]


@BOTH_GENERATIONS
def test_charts_carry_the_data_they_plot(source: Path):
    """Numbers renders no image for a chart, so what a reader can be given is
    the data it draws. Both generations cache that beside the chart, one in the
    chart archive and one in a property list of its own, and the summary table
    on the same sheet is what says whether it was read correctly."""
    doc = _backend(source).convert()
    pictures = list(doc.pictures)
    assert len(pictures) == 1

    assert _chart_grid(pictures[0]) == [
        ["", "Amount"],
        ["Home", "-872.4"],
        ["Food", "-226"],
        ["Gas", "-137.5"],
        ["Credit Card", "-1095"],
        ["Entertainment", "-245"],
    ]

    prediction = pictures[0].meta.classification.predictions[0]
    assert prediction.class_name == PictureClassificationLabel.OTHER_CHART


def test_a_chart_is_captioned_with_its_title():
    """Only the iWork '09 fixture still has a title on its chart; the 2013 one
    was saved after it was taken off, so its picture goes uncaptioned rather
    than captioned with something invented."""
    legacy = _backend(NUMBERS_IWORK09).convert()
    modern = _backend(NUMBERS_2013).convert()

    assert next(iter(legacy.pictures)).caption_text(legacy) == (
        "Expenditure by Category"
    )
    assert next(iter(modern.pictures)).caption_text(modern) == ""


@BOTH_GENERATIONS
def test_tables_and_charts_are_interleaved_down_the_sheet(source: Path):
    """A Numbers sheet is a canvas, so a chart can sit between two tables. The
    two fixtures place theirs differently, which is the point: the order has to
    come from the document rather than from the kind of thing being placed."""
    doc = _backend(source).convert()

    drawn = [
        item
        for item, _ in doc.iterate_items(with_groups=False)
        if isinstance(item, (TableItem, PictureItem)) and item.prov
    ]
    by_sheet: dict[int, list[float]] = {}
    for item in drawn:
        by_sheet.setdefault(item.prov[0].page_no, []).append(item.prov[0].bbox.t)

    assert len(by_sheet[1]) == 3
    for tops in by_sheet.values():
        assert tops == sorted(tops)


@BOTH_GENERATIONS
def test_sticky_notes_become_comments(source: Path):
    """Numbers calls a sheet-level comment a sticky note. Both fixtures carry
    the same one, so it must come out of both readers."""
    doc = _backend(source).convert()

    groups = [
        group
        for group in doc.groups
        if isinstance(group, GroupItem) and group.name.startswith("comment-")
    ]
    assert [group.name for group in groups] == ["comment-Checking-1"]

    notes = [
        item.text
        for item in doc.texts
        if isinstance(item, TextItem) and item.content_layer == ContentLayer.NOTES
    ]
    assert len(notes) == 1
    assert "drag an OFX file to the table" in notes[0]


def test_a_comment_records_who_left_it_and_when():
    """The 2013 container attributes a sticky note to an author and a moment;
    iWork '09 recorded neither, so its note is the bare text."""
    modern = [
        item.text
        for item in _backend(NUMBERS_2013).convert().texts
        if isinstance(item, TextItem) and item.content_layer == ContentLayer.NOTES
    ]
    legacy = [
        item.text
        for item in _backend(NUMBERS_IWORK09).convert().texts
        if isinstance(item, TextItem) and item.content_layer == ContentLayer.NOTES
    ]

    assert modern[0].startswith("[author: Author, time: 2016-05-04T13:08:26")
    assert legacy[0].startswith("Try adding your own account transactions")


@pytest.mark.parametrize("source", CONVERTIBLE, ids=lambda path: path.name)
def test_conversion_matches_the_groundtruth(source: Path):
    """Pin the whole conversion of every fixture, so a change in any part of the
    backend shows up as a reviewable diff rather than passing unnoticed.

    The Markdown is the reading order a caller gets by default; the serialized
    ``DoclingDocument`` is what carries the rest — the sheet grouping, the header
    rows and columns, each chart's data, and the comments that live outside the
    body layer.
    """
    doc = (
        DocumentConverter(allowed_formats=[InputFormat.IWORK_NUMBERS])
        .convert(source)
        .document
    )
    groundtruth = GROUNDTRUTH / source.name

    assert verify_export(
        doc.export_to_markdown(), str(groundtruth) + ".md", generate=GEN_TEST_DATA
    ), f"export to markdown failed on {source}"

    assert verify_document(doc, str(groundtruth) + ".json", generate=GEN_TEST_DATA), (
        f"DoclingDocument verification failed on {source}"
    )
