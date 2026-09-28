# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the charts of the Apple Keynote (``.key``) document backend.

Test Data Attribution
---------------------
``numbers_2013.numbers`` is ``testNumbers2013.numbers`` from the Apache Tika
test corpus, licensed under the Apache License 2.0, the same corpus the Keynote
fixtures come from. It is genuine Apple Numbers output holding one pie chart.

See https://github.com/apache/tika (``tika-parser-apple-module`` test resources).

Why a chart from Numbers
------------------------
No Keynote deck with a chart in it could be found under a licence that lets it
be committed. Pages, Numbers and Keynote draw charts with one shared engine,
though, so a chart on a Keynote slide is the same ``TSCH`` archive as a chart on
a Numbers sheet. :func:`_keynote_with_chart` therefore places the real chart of
``numbers_2013.numbers``, byte for byte, on the third slide of the real
``keynote_2013.key``, and leaves the content of every other object alone: the
slide gains the one reference that puts the chart on it, and that is all.
Keynote never wrote the result, which is why it is built here, where how it
was made can be read, rather than shipped.
"""

import logging
import struct
import zipfile
from io import BytesIO
from pathlib import Path

import defusedxml.ElementTree as ET
import pytest
from docling_core.types.doc import (
    DocItemLabel,
    PictureClassificationLabel,
    PictureItem,
    TableItem,
)
from docling_core.types.doc.items.text import TextItem
from PIL import Image, ImageDraw

import docling.backend.iwork_backend as iwork_backend
from docling.backend.docx.drawingml.utils import get_docx_to_pdf_converter
from docling.backend.iwork.archives import iwa_reference_field
from docling.backend.iwork.chart_image import (
    EMU_PER_POINT,
    PALETTE,
    chart_document,
)
from docling.backend.iwork.charts import (
    CHART_NON_STYLE_FIELD,
    DRAWABLE_CHART_FIELD,
    MAX_CHART_CELLS,
    TSCH_CHART_DRAWABLE,
    TSCH_CHART_NON_STYLE,
    iwa_chart,
    iwa_chart_grid,
)
from docling.backend.iwork.content import Chart, ChartKind, ChartSeries, Geometry
from docling.backend.iwork.iwa import IWAObject, iter_objects, read_fields
from docling.datamodel.backend_options import IWorkBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, IWorkKeynoteFormatOption

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

KEYNOTE_2013 = Path("./tests/data/keynote/sources/keynote_2013.key")
NUMBERS_2013 = Path("./tests/data/numbers/sources/numbers_2013.numbers")
GROUNDTRUTH = Path("./tests/data/keynote/groundtruth")

CHART_DECK_NAME = "keynote_2013_chart.key"

TABLE_SLIDE_MEMBER = "Index/Slide.iwa"
"""The member of ``keynote_2013.key`` holding its third slide, the one with a table."""

KN_SLIDE_ARCHIVE = 5

SLIDE_DRAWABLES_FIELD = 7

_TITLE = "Expenditure by Category"

_WEDGES = ("Home", "Food", "Gas", "Credit Card", "Entertainment")

_AMOUNTS = (-872.4, -226.0, -137.5, -1095.0, -245.0)

# Where Numbers placed the chart on its sheet, which it keeps on the slide.
_FRAME = Geometry(96.2, 78.2, 142.0, 142.0)

_C = "{http://schemas.openxmlformats.org/drawingml/2006/chart}"

_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

_WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"


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


def _iwa(objects: list[IWAObject]) -> bytes:
    """Write objects out as an ``.iwa`` member.

    Each object gets the least ``TSP.ArchiveInfo`` the reader needs — its
    identifier, message type and length — and the stream is stored in Snappy
    literals, which any Snappy decoder reads without compressing anything.
    """
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


def _objects(path: Path) -> dict[int, IWAObject]:
    with zipfile.ZipFile(path) as archive:
        return {
            obj.identifier: obj
            for info in archive.infolist()
            if info.filename.endswith(".iwa")
            for obj in iter_objects(archive.read(info))
        }


def _donor_chart() -> tuple[IWAObject, IWAObject]:
    """The chart of ``numbers_2013.numbers``, and the archive holding its title."""
    objects = _objects(NUMBERS_2013)
    chart = next(o for o in objects.values() if o.message_type == TSCH_CHART_DRAWABLE)
    archive = read_fields(chart.payload)[DRAWABLE_CHART_FIELD][0]
    assert isinstance(archive, bytes)
    title = iwa_reference_field(archive, CHART_NON_STYLE_FIELD)
    assert title is not None
    return chart, objects[title]


def _keynote_with_chart() -> bytes:
    """Place the real chart of ``numbers_2013.numbers`` on a real Keynote slide.

    The chart and the archive holding its title go into the member of the slide
    that holds the table, unchanged, and the slide gains the one reference to
    the chart that puts it among its drawables — a repeated field, so appending
    an entry leaves every other one where it was. That member is written out
    again, every object in it keeping its payload; every other member of the
    deck is copied as it is.

    The container is written stored, with fixed timestamps and host system, so
    that it comes out the same byte for byte wherever the test runs.
    """
    chart, title = _donor_chart()
    keynote = _objects(KEYNOTE_2013)
    assert not {chart.identifier, title.identifier} & set(keynote), (
        "the chart's identifiers collide with the deck's own"
    )

    out = BytesIO()
    with (
        zipfile.ZipFile(KEYNOTE_2013) as source,
        zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as target,
    ):
        for info in source.infolist():
            data = source.read(info)
            if info.filename == TABLE_SLIDE_MEMBER:
                slide = [
                    obj._replace(
                        payload=obj.payload
                        + _field(SLIDE_DRAWABLES_FIELD, _field(1, chart.identifier))
                    )
                    if obj.message_type == KN_SLIDE_ARCHIVE
                    else obj
                    for obj in iter_objects(data)
                ]
                data = _iwa([*slide, chart, title])
            member = zipfile.ZipInfo(info.filename, date_time=(1980, 1, 1, 0, 0, 0))
            member.create_system = 3
            target.writestr(member, data)
    return out.getvalue()


@pytest.fixture(scope="module")
def chart_deck(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("keynote") / CHART_DECK_NAME
    path.write_bytes(_keynote_with_chart())
    return path


def _convert(path: Path, options: IWorkBackendOptions | None = None):
    format_options = (
        {InputFormat.IWORK_KEYNOTE: IWorkKeynoteFormatOption(backend_options=options)}
        if options is not None
        else None
    )
    return (
        DocumentConverter(
            allowed_formats=[InputFormat.IWORK_KEYNOTE], format_options=format_options
        )
        .convert(path)
        .document
    )


def _grid(table_cells) -> dict[tuple[int, int], str]:
    return {
        (cell.start_row_offset_idx, cell.start_col_offset_idx): cell.text
        for cell in table_cells
    }


def test_the_chart_reader_reads_a_real_apple_chart():
    """The shared reader against the chart as Apple wrote it: a pie whose grid
    names five rows and one column, and says the rows are its series."""
    chart_drawable, _ = _donor_chart()
    chart = iwa_chart(chart_drawable, _objects(NUMBERS_2013))

    assert chart is not None
    assert chart.kind == ChartKind.PIE
    assert not chart.stacked and not chart.interactive
    assert chart.title == _TITLE
    assert chart.categories == ("Amount",)
    assert [series.name for series in chart.series] == list(_WEDGES)
    assert [series.values for series in chart.series] == [
        (amount,) for amount in _AMOUNTS
    ]


def test_the_built_deck_is_the_transplant_it_claims_to_be(chart_deck: Path):
    """Keynote never wrote the chart deck, so pin what it is: the Keynote
    fixture with the Numbers chart on one slide, and no other object changed."""
    chart, title = _donor_chart()
    built = _objects(chart_deck)

    assert built[chart.identifier].payload == chart.payload
    assert built[title.identifier].payload == title.payload

    with zipfile.ZipFile(KEYNOTE_2013) as original, zipfile.ZipFile(chart_deck) as deck:
        assert original.namelist() == deck.namelist()
        changed = [
            name
            for name in original.namelist()
            if original.read(name) != deck.read(name)
        ]
    assert changed == [TABLE_SLIDE_MEMBER]

    before = _objects(KEYNOTE_2013)
    assert set(built) == set(before) | {chart.identifier, title.identifier}
    grown = [i for i, obj in before.items() if built[i].payload != obj.payload]
    assert len(grown) == 1
    assert built[grown[0]].message_type == KN_SLIDE_ARCHIVE
    assert built[grown[0]].payload == before[grown[0]].payload + _field(
        SLIDE_DRAWABLES_FIELD, _field(1, chart.identifier)
    )


def test_a_chart_becomes_a_classified_picture_with_its_data(chart_deck: Path):
    """The shape the PowerPoint backend gives a chart: a picture classified by
    the chart's kind, with its data as a table and its title as its caption."""
    doc = _convert(chart_deck)

    assert len(doc.pictures) == 1
    picture = doc.pictures[0]
    assert picture.caption_text(doc) == _TITLE
    assert picture.image is None

    assert picture.meta is not None
    assert picture.meta.classification is not None
    assert (
        picture.meta.classification.predictions[0].class_name
        == PictureClassificationLabel.PIE_CHART
    )

    assert picture.meta.tabular_chart is not None
    data = picture.meta.tabular_chart.chart_data
    assert (data.num_rows, data.num_cols) == (2, 6)
    grid = _grid(data.table_cells)
    assert [grid[(0, col)] for col in range(6)] == ["", *_WEDGES]
    assert [grid[(1, col)] for col in range(6)] == [
        "Amount",
        "-872.4",
        "-226",
        "-137.5",
        "-1095",
        "-245",
    ]


def test_the_chart_takes_its_place_on_its_slide(chart_deck: Path):
    """The chart is read in the order its frame puts it — below the title and
    above the table — and carries that frame as its provenance."""
    doc = _convert(chart_deck)

    on_slide = [
        item
        for item, _ in doc.iterate_items()
        if item.prov and item.prov[0].page_no == 3
    ]
    kinds = [
        "picture"
        if isinstance(item, PictureItem)
        else "table"
        if isinstance(item, TableItem)
        else item.label
        for item in on_slide
    ]
    assert kinds == [DocItemLabel.TITLE, DocItemLabel.CAPTION, "picture", "table"]

    picture = doc.pictures[0]
    box = picture.prov[0].bbox
    assert (box.l, box.t) == pytest.approx((_FRAME.left, _FRAME.top), abs=0.01)
    assert (box.r - box.l, box.b - box.t) == pytest.approx(
        (_FRAME.width, _FRAME.height), abs=0.01
    )

    caption = on_slide[1]
    assert isinstance(caption, TextItem)
    assert caption.prov[0].charspan == (0, len(_TITLE))


def test_a_hidden_title_is_not_a_caption():
    """A chart stores a title whether or not it shows one, so a title the chart
    hides must not surface as a caption the slide never shows."""
    chart_drawable, _ = _donor_chart()
    objects = _objects(NUMBERS_2013)
    archive = read_fields(chart_drawable.payload)[DRAWABLE_CHART_FIELD][0]
    assert isinstance(archive, bytes)
    title_id = iwa_reference_field(archive, CHART_NON_STYLE_FIELD)
    assert title_id is not None

    hidden = _field(10000, _field(21, 0) + _field(23, b"Chart 1"))
    objects[title_id] = IWAObject(title_id, TSCH_CHART_NON_STYLE, hidden)

    chart = iwa_chart(chart_drawable, objects)
    assert chart is not None
    assert chart.title is None


def _grid_archive(rows: list[list[float | None]], names: list[str]) -> bytes:
    """Encode a ``TSCH.ChartGridArchive`` of numbers, naming its rows."""

    def value(number: float | None) -> bytes:
        if number is None:
            return b""
        return _varint(1 << 3 | 1) + struct.pack("<d", number)

    return b"".join(_field(1, name.encode()) for name in names) + b"".join(
        _field(3, b"".join(_field(1, value(number)) for number in row)) for row in rows
    )


def test_the_series_direction_decides_which_way_the_grid_is_read():
    """By row, each row of the grid is a series across the columns; by column,
    each column is a series down the rows. A gap stays in its place."""
    raw = _grid_archive([[1.0, None, 3.0], [4.0, 5.0, 6.0]], ["first", "second"])

    categories, series = iwa_chart_grid(raw, by_row=True)
    assert categories == ("", "", "")
    assert series == (
        ChartSeries("first", (1.0, None, 3.0)),
        ChartSeries("second", (4.0, 5.0, 6.0)),
    )

    categories, series = iwa_chart_grid(raw, by_row=False)
    assert categories == ("first", "second")
    assert [s.values for s in series] == [(1.0, 4.0), (None, 5.0), (3.0, 6.0)]


def test_a_grid_too_large_to_read_is_dropped():
    """An empty value costs two bytes, so the grid is bounded as it is read
    rather than trusted to be the size it claims."""
    raw = _field(3, _field(1, b"") * (MAX_CHART_CELLS + 1))

    assert iwa_chart_grid(raw, by_row=True) == ((), ())


def test_chart_images_are_not_rendered_by_default(chart_deck: Path):
    """Rendering needs LibreOffice and enlarges the output, so it is opt-in."""
    assert _convert(chart_deck).pictures[0].image is None


def _rebuilt(chart: Chart, geometry: Geometry | None):
    """The chart part and the document part of a chart rebuilt for LibreOffice."""
    rebuilt = chart_document(chart, geometry)
    assert rebuilt is not None
    with zipfile.ZipFile(BytesIO(rebuilt)) as package:
        return (
            ET.fromstring(package.read("word/charts/chart1.xml")),
            ET.fromstring(package.read("word/document.xml")),
        )


def _texts(source) -> list[str]:
    """The values of a DrawingML literal, in order."""
    return [value.text or "" for value in source.iter(f"{_C}v")]


def test_a_rebuilt_pie_draws_a_wedge_per_series():
    """Keynote draws a pie's wedges from its series and an Office chart from its
    categories, so the rebuild turns one into the other."""
    chart_drawable, _ = _donor_chart()
    chart = iwa_chart(chart_drawable, _objects(NUMBERS_2013))
    assert chart is not None

    space, document = _rebuilt(chart, _FRAME)

    (pie,) = space.iter(f"{_C}pieChart")
    (series,) = pie.findall(f"{_C}ser")
    assert _texts(series.find(f"{_C}cat")) == list(_WEDGES)
    assert [float(v) for v in _texts(series.find(f"{_C}val"))] == list(_AMOUNTS)
    assert "".join(t.text or "" for t in space.iter(f"{_A}t")) == _TITLE

    (extent,) = document.iter(f"{_WP}extent")
    assert int(extent.get("cx")) == int(_FRAME.width * EMU_PER_POINT)
    assert int(extent.get("cy")) == int(_FRAME.height * EMU_PER_POINT)


def test_every_series_is_given_its_colour():
    """The page carries no theme, so a series left to take an automatic colour
    is drawn with no fill at all: LibreOffice renders bars and wedges invisible.
    Every one must be given its colour outright."""
    series = (ChartSeries("s1", (1.0, 2.0)), ChartSeries("s2", (3.0, 4.0)))
    columns, _ = _rebuilt(Chart(ChartKind.COLUMN, None, ("a", "b"), series), None)
    wedges, _ = _rebuilt(Chart(ChartKind.PIE, None, ("a",), series), None)

    def colours(space, element: str) -> list[str | None]:
        return [
            fill.get("val")
            for item in space.iter(f"{_C}{element}")
            for fill in item.findall(f"{_C}spPr/{_A}solidFill/{_A}srgbClr")
        ]

    assert colours(columns, "ser") == list(PALETTE[:2])
    assert colours(wedges, "dPt") == list(PALETTE[:2])


@pytest.mark.parametrize(
    "chart",
    [
        Chart(ChartKind.MIXED, None, ("a",), (ChartSeries("s", (1.0,)),)),
        Chart(ChartKind.BUBBLE, None, ("a",), (ChartSeries("s", (1.0,)),)),
        Chart(
            ChartKind.COLUMN,
            None,
            ("a",),
            (ChartSeries("s", (1.0,)),),
            interactive=True,
        ),
        Chart(ChartKind.COLUMN, None, (), ()),
    ],
    ids=["mixed", "bubble", "interactive", "empty"],
)
def test_a_chart_an_office_chart_cannot_draw_is_not_rebuilt(chart: Chart):
    """No picture is better than one that misrepresents the chart."""
    assert chart_document(chart, None) is None


@pytest.mark.parametrize(
    ("shared_x", "expected"),
    [
        (False, {"y1": [(1.0, 10.0), (2.0, 20.0)]}),
        (True, {"y1": [(1.0, 10.0), (2.0, 20.0)], "y2": [(1.0, 5.0)]}),
    ],
    ids=["separate-x", "shared-x"],
)
def test_a_scatter_chart_pairs_its_series_into_points(shared_x: bool, expected):
    """A scatter chart either shares its first series as x or pairs its series
    up, x before y; a point missing either value is not drawn."""
    chart = Chart(
        ChartKind.SCATTER,
        None,
        ("p", "q"),
        (
            ChartSeries("x", (1.0, 2.0)),
            ChartSeries("y1", (10.0, 20.0)),
            ChartSeries("y2", (5.0, None)),
        ),
        shared_x=shared_x,
    )
    space, _ = _rebuilt(chart, None)

    drawn = {
        _texts(series.find(f"{_C}tx"))[0]: list(
            zip(
                [float(x) for x in _texts(series.find(f"{_C}xVal"))],
                [float(y) for y in _texts(series.find(f"{_C}yVal"))],
            )
        )
        for series in space.iter(f"{_C}ser")
    }
    assert drawn == expected


def test_rendering_without_libreoffice_keeps_the_data(
    chart_deck: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Asking for images on a machine that cannot draw them warns once and
    leaves the chart's classification and data in place."""
    monkeypatch.setattr(iwork_backend, "get_docx_to_pdf_converter", lambda: None)

    with caplog.at_level(logging.WARNING):
        doc = _convert(chart_deck, IWorkBackendOptions(render_chart_images=True))

    assert "LibreOffice is required" in caplog.text
    picture = doc.pictures[0]
    assert picture.image is None
    assert picture.meta is not None
    assert picture.meta.tabular_chart is not None


def test_a_drawn_chart_is_cropped_and_attached(
    chart_deck: Path, monkeypatch: pytest.MonkeyPatch
):
    """The route after LibreOffice, without LibreOffice: whatever PDF it writes
    for the rebuilt chart is rasterized, cropped to what was drawn, and attached
    to the chart's picture."""
    received: list[bytes] = []

    def converter(input_path: Path, output_path: Path) -> None:
        received.append(Path(input_path).read_bytes())
        page = Image.new("RGB", (300, 200), "white")
        ImageDraw.Draw(page).rectangle((50, 40, 149, 119), fill="black")
        page.save(output_path, "PDF", resolution=72)

    monkeypatch.setattr(iwork_backend, "get_docx_to_pdf_converter", lambda: converter)
    doc = _convert(chart_deck, IWorkBackendOptions(render_chart_images=True))

    assert len(received) == 1
    with zipfile.ZipFile(BytesIO(received[0])) as package:
        assert "word/charts/chart1.xml" in package.namelist()
    image = doc.pictures[0].get_image(doc)
    assert image is not None
    assert image.width < 600 and image.height < 400, "the page should be cropped"
    assert image.width >= 190 and image.height >= 150


def test_a_chart_is_rendered_through_libreoffice(chart_deck: Path):
    """The whole route, where LibreOffice is installed. Its output is not
    byte-stable across versions, so rather than pixels, what is checked is that
    there is a picture and that its wedges were filled in."""
    # The backend's own check, which unlike running `soffice -h` does not open a
    # help window and wait on Windows.
    if get_docx_to_pdf_converter() is None:
        pytest.skip("LibreOffice is not installed — chart rendering cannot be tested")

    doc = _convert(chart_deck, IWorkBackendOptions(render_chart_images=True))

    picture = doc.pictures[0]
    image = picture.get_image(doc)
    assert image is not None, "the chart picture should carry a rendered image"
    assert image.width > 50 and image.height > 50
    colours = image.convert("RGB").getcolors(image.width * image.height) or []
    first_wedge = tuple(bytes.fromhex(PALETTE[0]))
    assert any(colour == first_wedge for _, colour in colours), (
        "the wedges should be filled in"
    )
    assert picture.meta is not None
    assert picture.meta.tabular_chart is not None


def test_conversion_matches_the_groundtruth(chart_deck: Path):
    """Pin the whole conversion of the chart deck, as the Keynote tests pin
    every other fixture."""
    doc = _convert(chart_deck)
    groundtruth = GROUNDTRUTH / CHART_DECK_NAME

    assert verify_export(
        doc.export_to_markdown(), str(groundtruth) + ".md", generate=GEN_TEST_DATA
    ), f"export to markdown failed on {CHART_DECK_NAME}"

    assert verify_document(doc, str(groundtruth) + ".json", generate=GEN_TEST_DATA), (
        f"DoclingDocument verification failed on {CHART_DECK_NAME}"
    )
