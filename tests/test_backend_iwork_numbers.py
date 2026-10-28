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

``numbers_iwork09_charts.numbers`` is ``testNumbersCharts.numbers`` from the same
corpus and under the same license: iWork '09 output holding three charts of
different kinds, plotted in both directions. Only its charts are pinned.

See https://github.com/apache/tika (``tika-parser-apple-module`` test resources).

No public Numbers document places a picture on a sheet. The picture tests
therefore move a photo that Keynote placed in the Tika Keynote fixtures
``keynote_2013.key`` and ``keynote_iwork09.key`` (same corpus, same license)
onto the first sheet of these documents. Numbers and Keynote share their
drawable archives, so the picture is real Apple output in either generation.

The cell buffers in :func:`test_version_5_cell_storage_is_decoded` were captured
from Numbers documents saved by releases newer than either fixture, whose cells
use a storage layout the fixtures never exercise.

``numbers_cell_pictures.numbers``, ``numbers_cell_fills.numbers``,
``numbers_cell_pictures_repeated.numbers`` and
``numbers_cell_pictures_styled.numbers`` are ``issue-43.numbers``,
``test-package.numbers``, ``issue-69.numbers`` and ``test-styles.numbers`` from
the numbers-parser test corpus, licensed under the MIT License (Copyright 2021
Jon Connell). They are Numbers output that fills table cells with images. Only
the first two have a stored groundtruth: the photos in the other two would make
it tens of megabytes.

See https://github.com/masaccio/numbers-parser (``tests/data``).
"""

import logging
import struct
import zipfile
from io import BytesIO
from pathlib import Path

import defusedxml.ElementTree as ET
import pytest
from docling_core.types.doc import (
    ContentLayer,
    GroupItem,
    GroupLabel,
    NodeItem,
    PictureClassificationLabel,
    PictureItem,
    RichTableCell,
    TableCell,
    TableItem,
    TextItem,
)
from PIL import Image, ImageDraw

import docling.backend.iwork_backend as iwork_backend
from docling.backend.docx.drawingml.utils import get_docx_to_pdf_converter
from docling.backend.iwork import cells, numbers_xml
from docling.backend.iwork.archives import (
    PACKAGE_DATAS_FIELD,
    TSD_GROUP,
    TSD_IMAGE,
    TSP_PACKAGE_METADATA,
)
from docling.backend.iwork.chart_image import PALETTE
from docling.backend.iwork.content import Chart, ChartKind, ChartSeries
from docling.backend.iwork.iwa import IWAObject, iter_objects, read_fields
from docling.backend.iwork.legacy import SF_NAMESPACE, SFA_NAMESPACE
from docling.backend.iwork.numbers_iwa import (
    SHEET_DRAWABLES_FIELD,
    SHEET_NAME_FIELD,
    TN_SHEET_ARCHIVE,
    render,
)
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
NUMBERS_IWORK09_CHARTS = SOURCES / "numbers_iwork09_charts.numbers"
NUMBERS_CELL_PICTURES = SOURCES / "numbers_cell_pictures.numbers"
NUMBERS_CELL_FILLS = SOURCES / "numbers_cell_fills.numbers"
NUMBERS_CELL_PICTURES_REPEATED = SOURCES / "numbers_cell_pictures_repeated.numbers"
NUMBERS_CELL_PICTURES_STYLED = SOURCES / "numbers_cell_pictures_styled.numbers"
GROUNDTRUTH = Path("./tests/data/numbers/groundtruth")

KEYNOTE_2013 = Path("./tests/data/keynote/sources/keynote_2013.key")
KEYNOTE_IWORK09 = Path("./tests/data/keynote/sources/keynote_iwork09.key")

# The fixtures whose whole conversion is pinned by a stored groundtruth.
CONVERTIBLE = [NUMBERS_2013, NUMBERS_IWORK09, NUMBERS_CELL_PICTURES, NUMBERS_CELL_FILLS]

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
    values = cells.CellValues(strings={14: "YYY_2_1"})
    decoded = cells.iwa_cell(bytes.fromhex(buffer), 0, values)

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
    """Numbers keeps no image of a chart, so what a reader is always given is the
    data it draws. Both generations cache that beside the chart — one in the chart
    archive, one in a property list of its own — and the summary table on the same
    sheet is what says whether it was read correctly.

    How the values are laid out is pinned separately, by
    :func:`test_both_generations_orient_the_chart_the_same_way`.
    """
    doc = _backend(source).convert()
    pictures = list(doc.pictures)
    assert len(pictures) == 1

    grid = _chart_grid(pictures[0])
    plotted = {text for row in grid for text in row if text}
    assert plotted == {
        "Amount",
        "Home",
        "Food",
        "Gas",
        "Credit Card",
        "Entertainment",
        "-872.4",
        "-226",
        "-137.5",
        "-1095",
        "-245",
    }


def test_both_generations_orient_the_chart_the_same_way():
    """Both generations record which way round a chart's data is plotted: a 2013+
    chart in its series direction, an iWork '09 one in ``sf:chart-direction``.
    The pie plots its rows as series in both, so the same chart comes out the
    same way round from either reader: a wedge per row, as Numbers draws it."""
    modern = _chart_grid(next(iter(_backend(NUMBERS_2013).convert().pictures)))
    legacy = _chart_grid(next(iter(_backend(NUMBERS_IWORK09).convert().pictures)))

    # Series run across the header, categories down the first column.
    assert modern == legacy
    assert legacy == [
        ["", "Home", "Food", "Gas", "Credit Card", "Entertainment"],
        ["Amount", "-872.4", "-226", "-137.5", "-1095", "-245"],
    ]


@BOTH_GENERATIONS
def test_a_chart_is_classified_by_its_kind(source: Path):
    """The summary chart is a pie in both fixtures, which each generation says in
    a numbering of its own: ``TSCH.ChartType`` in a 2013+ document, and the
    order of Keynote '09's scripting dictionary in an iWork '09 one."""
    picture = next(iter(_backend(source).convert().pictures))

    assert picture.meta is not None and picture.meta.classification is not None
    assert (
        picture.meta.classification.predictions[0].class_name
        == PictureClassificationLabel.PIE_CHART
    )


def _legacy_charts(path: Path) -> list[Chart]:
    """Read the charts of an iWork '09 document, sheet by sheet, top to bottom."""
    with zipfile.ZipFile(path) as archive:
        sheets = numbers_xml.read_content(
            archive, "index.xml", 300 * 1024 * 1024, 100 * 1024 * 1024, "test"
        )
    return [placed.chart for sheet in sheets for placed in sheet.charts]


def test_iwork09_chart_kinds_follow_the_keynote_09_scripting_dictionary():
    """``sf:chart-type`` is the position of the kind in Keynote '09's ``add
    chart`` command. Each of these three charts confirms it independently: the
    pie by the thumbnail Numbers saved and by its 2013 twin, the 3D area by the
    thumbnail and its ``SFC3DAreaChartScaleProperty`` style, and the 3D column by
    its ``SFC3DColumnChartScaleProperty`` style."""
    charts = _legacy_charts(NUMBERS_IWORK09_CHARTS)

    assert [(chart.title, chart.kind, chart.stacked) for chart in charts] == [
        ("Expenditure by Category", ChartKind.PIE, False),
        ("Currency Chart name", ChartKind.AREA, False),
        ("Chart 2", ChartKind.COLUMN, False),
    ]


def test_iwork09_charts_are_read_in_the_direction_they_plot():
    """``sf:chart-direction`` 0 plots each row as a series and 1 each column.
    The 3D area chart says 1 and Numbers draws it with a legend entry for each of
    its four columns; the column chart says 0, so its two regions are the series
    and the years the categories."""
    _, area, columns = _legacy_charts(NUMBERS_IWORK09_CHARTS)

    assert len(area.series) == 4
    assert len(area.categories) == 9
    assert area.categories[1:3] == ("average pay", "maximum wage")
    assert [series.values[1:3] for series in area.series] == [
        (None, None),
        (0.5, None),
        (0.1, 0.6),
        (None, None),
    ]

    assert columns.categories == ("2007", "2008", "2009", "2010")
    assert columns.series == (
        ChartSeries("Region 1", (17.0, 26.0, 53.0, 96.0)),
        ChartSeries("Region 2", (55.0, 43.0, 70.0, 58.0)),
    )


@pytest.mark.parametrize(
    "chart_type", ['sf:chart-type="18"', ""], ids=["unknown", "unsaid"]
)
def test_an_unknown_iwork09_chart_kind_is_not_guessed(chart_type: str):
    """Kinds outside the scripting dictionary, such as the mixed and two-axis
    charts, and a chart that does not say, are left unspecified."""
    info = ET.fromstring(
        f'<sf:chart-info xmlns:sf="{SF_NAMESPACE}" xmlns:sfa="{SFA_NAMESPACE}" '
        f'{chart_type}><sf:chart-row_names><sf:string sfa:string="a"/>'
        "</sf:chart-row_names></sf:chart-info>"
    )
    placed = numbers_xml.read_chart(info, {})

    assert placed is not None
    assert placed.chart.kind == ChartKind.OTHER


def test_a_chart_is_captioned_with_its_title():
    """Both fixtures title the chart, and the title is reached differently in
    each — from the archive that holds a modern chart's non-style settings, and
    from ``sf:chart-name`` in an iWork '09 one — so both are worth pinning."""
    legacy = _backend(NUMBERS_IWORK09).convert()
    modern = _backend(NUMBERS_2013).convert()

    assert next(iter(legacy.pictures)).caption_text(legacy) == (
        "Expenditure by Category"
    )
    assert next(iter(modern.pictures)).caption_text(modern) == (
        "Expenditure by Category"
    )


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


_PHOTO_IMAGE = 4210
"""The ``TSD.ImageArchive`` of a photo in ``keynote_2013.key``."""

_PHOTO_DATA = (104, 105)
"""The data files that image names: the full photo and a smaller rendition."""

_PHOTO_MEMBER = "Data/happy_girls-small-105.jpg"
"""The only one of those that Keynote stored in the container."""

_PHOTO_FRAME = (246.0, 171.0, 533.33, 330.85)
"""Where Keynote placed the photo, the same in both generations of the deck."""

_GROUP_FRAME = (40.0, 500.0, 300.0, 200.0)

_FIRST_TABLE = '<sf:tabular-info sfa:ID="SFTTableInfo-0"'
"""The first table of the first sheet of ``numbers_iwork09.numbers``."""


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        low, value = value & 0x7F, value >> 7
        out.append(low | 0x80 if value else low)
        if not value:
            return bytes(out)


def _field(number: int, value: int | bytes) -> bytes:
    """Encode one protobuf field, as a varint or as length-delimited bytes."""
    if isinstance(value, int):
        return _varint(number << 3) + _varint(value)
    return _varint(number << 3 | 2) + _varint(len(value)) + value


def _point(x: float, y: float) -> bytes:
    """Encode a ``TSP.Point`` or a ``TSP.Size``: two 32-bit float fields."""
    return b"".join(
        _varint(number << 3 | 5) + struct.pack("<f", value)
        for number, value in ((1, x), (2, y))
    )


def _iwa(objects: list[IWAObject]) -> bytes:
    """Write objects out as an ``.iwa`` member, stored as Snappy literals."""
    stream = b""
    for obj in objects:
        info = _field(1, obj.identifier) + _field(
            2, _field(1, obj.message_type) + _field(3, len(obj.payload))
        )
        stream += _varint(len(info)) + info + obj.payload

    member = b""
    for start in range(0, len(stream), 1 << 16):
        chunk = stream[start : start + (1 << 16)]
        size = len(chunk) - 1
        width = (size.bit_length() + 7) // 8
        tag = bytes([size << 2]) if size < 60 else bytes([(59 + width) << 2])
        extra = b"" if size < 60 else size.to_bytes(width, "little")
        block = _varint(len(chunk)) + tag + extra + chunk
        member += b"\x00" + len(block).to_bytes(3, "little") + block
    return member


def _objects(archive: zipfile.ZipFile) -> dict[int, IWAObject]:
    return {
        obj.identifier: obj
        for info in archive.infolist()
        if info.filename.endswith(".iwa")
        for obj in iter_objects(archive.read(info))
    }


def _with_picture(target: Path, *, stored: bool = True, grouped: bool = False) -> Path:
    """Place the photo of ``keynote_2013.key`` on a sheet of ``numbers_2013``.

    The ``TSD.ImageArchive`` of Keynote is copied as it is, under an identifier
    that the spreadsheet does not use. The sheet gets one more reference in its
    list of drawables, and the package metadata gets the two ``TSP.DataInfo``
    entries that the image names. The objects go into an extra member, which
    replaces the objects with the same identifiers.

    Args:
        target: Where to write the document.
        stored: Whether to copy the rendition of the photo that Keynote stored.
        grouped: Whether to put the image in a ``TSD.GroupArchive`` placed at
            ``_GROUP_FRAME``.

    Returns:
        The path of the document.
    """
    with zipfile.ZipFile(KEYNOTE_2013) as keynote:
        donor = _objects(keynote)
        photo = keynote.read(_PHOTO_MEMBER)
    assert donor[_PHOTO_IMAGE].message_type == TSD_IMAGE
    donor_metadata = next(
        obj for obj in donor.values() if obj.message_type == TSP_PACKAGE_METADATA
    )
    data_infos = b"".join(
        _field(PACKAGE_DATAS_FIELD, entry)
        for entry in read_fields(donor_metadata.payload)[PACKAGE_DATAS_FIELD]
        if isinstance(entry, bytes) and read_fields(entry)[1][0] in _PHOTO_DATA
    )

    with (
        zipfile.ZipFile(NUMBERS_2013) as source,
        zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as out,
    ):
        objects = _objects(source)
        sheet = next(
            obj
            for obj in objects.values()
            if obj.message_type == TN_SHEET_ARCHIVE
            and read_fields(obj.payload)[SHEET_NAME_FIELD][0] == b"Checking"
        )
        metadata = next(
            obj for obj in objects.values() if obj.message_type == TSP_PACKAGE_METADATA
        )

        image = IWAObject(max(objects) + 1, TSD_IMAGE, donor[_PHOTO_IMAGE].payload)
        placed = [image]
        if grouped:
            left, top, width, height = _GROUP_FRAME
            geometry = _field(1, _point(left, top)) + _field(2, _point(width, height))
            group = _field(1, _field(1, geometry)) + _field(
                2, _field(1, image.identifier)
            )
            placed.append(IWAObject(image.identifier + 1, TSD_GROUP, group))

        sheet = sheet._replace(
            payload=sheet.payload
            + _field(SHEET_DRAWABLES_FIELD, _field(1, placed[-1].identifier))
        )
        metadata = metadata._replace(payload=metadata.payload + data_infos)

        for info in source.infolist():
            out.writestr(info, source.read(info))
        out.writestr("Index/Picture.iwa", _iwa([sheet, metadata, *placed]))
        if stored:
            out.writestr(_PHOTO_MEMBER, photo)
    return target


def _with_legacy_picture(target: Path, *, stored: bool = True) -> Path:
    """Place the photo of ``keynote_iwork09.key`` on a sheet of ``numbers_iwork09``.

    Both apps write the same ``sf`` vocabulary, so the ``sf:media`` of Keynote
    is copied as it is into the drawables of the sheet, beside its first table.
    It names a file that Keynote did not store in the package.

    Args:
        target: Where to write the document.
        stored: Whether to store the rendition that ``keynote_2013.key`` keeps
            of the same photo under the name the element gives.

    Returns:
        The path of the document.
    """
    with zipfile.ZipFile(KEYNOTE_IWORK09) as deck:
        media = next(
            ET.fromstring(deck.read("index.apxl")).iter(f"{{{SF_NAMESPACE}}}media")
        )
    data = next(media.iter(f"{{{SF_NAMESPACE}}}data"))
    path = data.get(f"{{{SF_NAMESPACE}}}path")
    assert path == "Shared/Happy Girls.jpg"
    with zipfile.ZipFile(KEYNOTE_2013) as keynote:
        photo = keynote.read(_PHOTO_MEMBER)

    with (
        zipfile.ZipFile(NUMBERS_IWORK09) as source,
        zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as out,
    ):
        index = source.read("index.xml").decode("utf-8")
        assert index.count(_FIRST_TABLE) == 1
        index = index.replace(
            _FIRST_TABLE, ET.tostring(media, encoding="unicode") + _FIRST_TABLE
        )
        for info in source.infolist():
            if info.filename == "index.xml":
                out.writestr(info, index.encode("utf-8"))
            else:
                out.writestr(info, source.read(info))
        if stored:
            out.writestr(path, photo)
    return target


_PICTURE_BUILDERS = {NUMBERS_2013: _with_picture, NUMBERS_IWORK09: _with_legacy_picture}


def _photos(doc) -> list[PictureItem]:
    """The pictures that are not charts. A chart carries a classification."""
    return [picture for picture in doc.pictures if picture.meta is None]


@BOTH_GENERATIONS
def test_a_picture_is_read_where_it_sits_on_its_sheet(tmp_path: Path, source: Path):
    """A picture on a sheet becomes a picture item on the page of that sheet, in
    the frame Numbers gave it. It is read in the order the sheet lays it out:
    below the chart and above the register of transactions."""
    doc = _backend(_PICTURE_BUILDERS[source](tmp_path / source.name)).convert()

    photos = _photos(doc)
    assert len(photos) == 1
    photo = photos[0]
    assert photo.image is not None
    assert (photo.image.size.width, photo.image.size.height) == (256, 170)

    prov = photo.prov[0]
    assert prov.page_no == 1
    assert (prov.bbox.l, prov.bbox.t, prov.bbox.width, prov.bbox.height) == (
        pytest.approx(_PHOTO_FRAME, abs=0.01)
    )

    order = [item.self_ref for item, _ in doc.iterate_items()]
    chart = next(picture for picture in doc.pictures if picture.meta is not None)
    register = next(
        table for table in _tables(doc) if table.caption_text(doc) == "Transactions"
    )
    assert (
        order.index(chart.self_ref)
        < order.index(photo.self_ref)
        < order.index(register.self_ref)
    )


@BOTH_GENERATIONS
def test_a_picture_whose_bytes_are_not_stored_keeps_its_place(
    tmp_path: Path, source: Path
):
    """Keynote names image data that it did not store in the package. Such a
    picture is still placed on its sheet, without an image."""
    doc = _backend(
        _PICTURE_BUILDERS[source](tmp_path / source.name, stored=False)
    ).convert()

    photos = _photos(doc)
    assert len(photos) == 1
    assert photos[0].image is None
    assert photos[0].prov[0].bbox.t == pytest.approx(_PHOTO_FRAME[1], abs=0.01)


def test_a_grouped_picture_takes_the_frame_of_its_group(tmp_path: Path):
    """Pages and Keynote read the pictures in a group, and Numbers does too. The
    group is what sits on the sheet, so the picture gets the frame of the group,
    and the page of the sheet reaches down to it."""
    doc = _backend(_with_picture(tmp_path / "grouped.numbers", grouped=True)).convert()

    photos = _photos(doc)
    assert len(photos) == 1
    assert photos[0].image is not None
    bbox = photos[0].prov[0].bbox
    assert (bbox.l, bbox.t, bbox.width, bbox.height) == pytest.approx(_GROUP_FRAME)
    _, top, _, height = _GROUP_FRAME
    assert doc.pages[1].size.height == pytest.approx(top + height)


def _cells(table: TableItem) -> dict[tuple[int, int], TableCell]:
    return {
        (cell.start_row_offset_idx, cell.start_col_offset_idx): cell
        for cell in table.data.table_cells
    }


def _cell_items(doc, cell: TableCell) -> list[NodeItem]:
    """The items that a rich cell holds, in order."""
    assert isinstance(cell, RichTableCell)
    group = cell.ref.resolve(doc)
    return [child.resolve(doc) for child in group.children]


def test_a_cell_filled_with_a_picture_becomes_a_rich_cell():
    """Numbers draws the image that fills a cell behind the text of the cell.
    The cell becomes a rich cell, as a picture in a Word table cell does. Its
    group holds the text, if there is text, and then the picture. Both are on
    the page of the sheet, in the frame of the table."""
    doc = _backend(NUMBERS_CELL_PICTURES).convert()
    (table,) = _tables(doc)
    cells = _cells(table)

    (icon,) = _cell_items(doc, cells[(0, 0)])
    text, laptop = _cell_items(doc, cells[(0, 1)])
    assert isinstance(text, TextItem)
    assert text.text == cells[(0, 1)].text == "text "
    for picture, size in ((icon, (452, 512)), (laptop, (370, 244))):
        assert isinstance(picture, PictureItem)
        assert picture.image is not None
        assert (picture.image.size.width, picture.image.size.height) == size
        assert picture.prov[0].page_no == 1
        assert picture.prov[0].bbox == table.prov[0].bbox

    assert not isinstance(cells[(0, 2)], RichTableCell)
    assert cells[(0, 2)].text == "no image"


def test_a_cell_filled_with_a_colour_holds_no_picture():
    """A cell can be filled with a colour as well as an image. Only an image
    fill makes a picture."""
    doc = _backend(NUMBERS_CELL_FILLS).convert()
    (table,) = _tables(doc)
    cells = _cells(table)

    (cat,) = _cell_items(doc, cells[(0, 0)])
    assert isinstance(cat, PictureItem)
    assert cat.image is not None
    assert not isinstance(cells[(1, 1)], RichTableCell)
    assert cells[(1, 1)].text == "No Dog"
    assert len(doc.pictures) == 1


def test_a_photo_that_fills_every_cell_is_in_every_cell():
    """All 50 cells of this table are filled with the same photo, and Numbers
    draws it in each of them."""
    doc = _backend(NUMBERS_CELL_PICTURES_REPEATED).convert()
    (table,) = _tables(doc)

    assert len(table.data.table_cells) == 50
    for cell in table.data.table_cells:
        (photo,) = _cell_items(doc, cell)
        assert isinstance(photo, PictureItem)
        assert photo.image is not None
        assert (photo.image.size.width, photo.image.size.height) == (940, 940)


def test_a_picture_whose_name_is_not_ascii_is_found():
    """Numbers writes the member names of its container in UTF-8, but does not
    set the flag that says so. zipfile then reads a name as code page 437, so
    the name that the document gives must be found under that reading."""
    with zipfile.ZipFile(NUMBERS_CELL_PICTURES_REPEATED) as archive:
        (member,) = [
            info for info in archive.infolist() if info.filename.startswith("Data/")
        ]
    assert not member.flag_bits & 0x800
    assert not member.filename.isascii()

    doc = _backend(NUMBERS_CELL_PICTURES_REPEATED).convert()

    pictures = list(doc.pictures)
    assert len(pictures) == 50
    assert all(picture.image is not None for picture in pictures)


def test_a_picture_in_a_header_cell_keeps_the_header():
    """On the Headers sheet, Numbers fills cells in a header row, in a header
    column and in a footer row with a photo. A rich cell keeps the header flags
    that a plain cell in the same place gets."""
    doc = _backend(
        NUMBERS_CELL_PICTURES_STYLED, IWorkBackendOptions(sheet_names=["Headers"])
    ).convert()
    (table,) = _tables(doc)

    rich = {
        key: cell
        for key, cell in _cells(table).items()
        if isinstance(cell, RichTableCell)
    }
    assert {
        key: (cell.column_header, cell.row_header) for key, cell in rich.items()
    } == {
        (1, 3): (True, False),
        (1, 4): (True, False),
        (6, 1): (False, True),
        (6, 2): (False, True),
        (9, 3): (False, False),
        (9, 4): (False, False),
    }
    for cell in rich.values():
        text, photo = _cell_items(doc, cell)
        assert isinstance(text, TextItem)
        assert text.text == cell.text
        assert isinstance(photo, PictureItem)
        assert photo.image is not None


def _fake_converter(received: list[bytes]):
    """Stand in for LibreOffice, drawing a black box on a white page."""

    def converter(input_path: Path, output_path: Path) -> None:
        received.append(Path(input_path).read_bytes())
        page = Image.new("RGB", (300, 200), "white")
        ImageDraw.Draw(page).rectangle((50, 40, 149, 119), fill="black")
        page.save(output_path, "PDF", resolution=72)

    return converter


def _rebuilt_kinds(received: list[bytes]) -> list[str]:
    """The DrawingML chart type each rebuilt chart was written as."""
    kinds = []
    for document in received:
        with zipfile.ZipFile(BytesIO(document)) as package:
            space = ET.fromstring(package.read("word/charts/chart1.xml"))
        (plot,) = space.iter(
            "{http://schemas.openxmlformats.org/drawingml/2006/chart}plotArea"
        )
        kinds.extend(child.tag.split("}")[1] for child in plot if "Chart" in child.tag)
    return kinds


@pytest.mark.parametrize(
    "source", [NUMBERS_2013, NUMBERS_IWORK09_CHARTS], ids=["iwa", "iwork09"]
)
def test_chart_images_are_not_rendered_by_default(source: Path):
    """Rendering needs LibreOffice and enlarges the output, so it is opt-in."""
    doc = _backend(source).convert()
    assert all(picture.image is None for picture in doc.pictures)


@pytest.mark.parametrize(
    ("source", "kinds"),
    [
        (NUMBERS_2013, ["pieChart"]),
        (NUMBERS_IWORK09_CHARTS, ["pieChart", "areaChart", "barChart"]),
    ],
    ids=["iwa", "iwork09"],
)
def test_every_chart_on_a_sheet_is_drawn_when_asked(
    source: Path, kinds: list[str], monkeypatch: pytest.MonkeyPatch
):
    """The route after LibreOffice, without LibreOffice: each chart is rebuilt
    as the kind of Office chart it is, and whatever PDF comes back is cropped to
    what was drawn and attached to the chart's picture, beside its data."""
    received: list[bytes] = []
    monkeypatch.setattr(
        iwork_backend, "get_docx_to_pdf_converter", lambda: _fake_converter(received)
    )

    doc = _backend(source, IWorkBackendOptions(render_chart_images=True)).convert()

    assert _rebuilt_kinds(received) == kinds
    for picture in doc.pictures:
        image = picture.get_image(doc)
        assert image is not None
        assert image.width < 600 and image.height < 400, "the page should be cropped"
        assert picture.meta is not None and picture.meta.tabular_chart is not None


def test_a_chart_that_cannot_be_redrawn_stays_a_picture_with_its_data(
    monkeypatch: pytest.MonkeyPatch,
):
    """A chart of a kind an Office chart cannot stand in for is left undrawn
    rather than drawn as something it is not, and keeps its place, its
    classification and its data."""
    received: list[bytes] = []
    monkeypatch.setattr(
        iwork_backend, "get_docx_to_pdf_converter", lambda: _fake_converter(received)
    )
    monkeypatch.setattr(numbers_xml, "LEGACY_CHART_TYPES", {})

    doc = _backend(
        NUMBERS_IWORK09_CHARTS, IWorkBackendOptions(render_chart_images=True)
    ).convert()

    assert received == []
    pictures = list(doc.pictures)
    assert len(pictures) == 3
    for picture in pictures:
        assert picture.image is None
        assert picture.prov
        assert picture.meta is not None and picture.meta.classification is not None
        assert (
            picture.meta.classification.predictions[0].class_name
            == PictureClassificationLabel.OTHER_CHART
        )
        assert picture.meta.tabular_chart is not None


def test_rendering_without_libreoffice_keeps_the_data(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Asking for images on a machine that cannot draw them warns once and
    leaves the chart's classification and data in place."""
    monkeypatch.setattr(iwork_backend, "get_docx_to_pdf_converter", lambda: None)

    with caplog.at_level(logging.WARNING):
        doc = _backend(
            NUMBERS_IWORK09_CHARTS, IWorkBackendOptions(render_chart_images=True)
        ).convert()

    assert caplog.text.count("LibreOffice is required") == 1
    for picture in doc.pictures:
        assert picture.image is None
        assert picture.meta is not None and picture.meta.tabular_chart is not None


@pytest.mark.parametrize(
    "source", [NUMBERS_2013, NUMBERS_IWORK09], ids=["iwa", "iwork09"]
)
def test_a_chart_is_rendered_through_libreoffice(source: Path):
    """The whole route, where LibreOffice is installed, for the same pie read
    from either generation. Its output is not byte-stable across versions, so
    rather than pixels, what is checked is that there is a picture and that its
    wedges were filled in."""
    # The backend's own check, which unlike running `soffice -h` does not open a
    # help window and wait on Windows.
    if get_docx_to_pdf_converter() is None:
        pytest.skip("LibreOffice is not installed — chart rendering cannot be tested")

    doc = _backend(source, IWorkBackendOptions(render_chart_images=True)).convert()

    picture = next(iter(doc.pictures))
    image = picture.get_image(doc)
    assert image is not None, "the chart picture should carry a rendered image"
    assert image.width > 50 and image.height > 50
    colours = image.convert("RGB").getcolors(image.width * image.height) or []
    first_wedge = tuple(bytes.fromhex(PALETTE[0]))
    assert any(colour == first_wedge for _, colour in colours), (
        "the wedges should be filled in"
    )


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
