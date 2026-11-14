# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the chart types that Office 2016 added, such as waterfall and treemap.

PowerPoint keeps these charts in a chartex part and wraps their graphic frame in
an ``mc:AlternateContent``, which python-pptx does not read. python-pptx cannot
write such a chart, so the decks here add the part and the XML by hand.
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

import pytest
from docling_core.types.doc import (
    ContentLayer,
    DocItemLabel,
    PictureClassificationLabel,
    PictureItem,
    TextItem,
)
from lxml import etree
from PIL import Image

from docling.backend.utils.chartex import read_chartex

from .test_backend_pptx import get_converter

_CX = "http://schemas.microsoft.com/office/drawing/2014/chartex"
_NAMESPACES = (
    'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"'
)
_EMU_PER_INCH = 914400


def _level(points: list[str], extra: str = "") -> str:
    """Return a ``cx:lvl``. An empty point gets no ``cx:pt``, as for an empty cell."""
    return (
        f'<cx:lvl ptCount="{len(points)}"{extra}>'
        + "".join(
            f'<cx:pt idx="{index}">{text}</cx:pt>'
            for index, text in enumerate(points)
            if text
        )
        + "</cx:lvl>"
    )


def _categories(*levels: list[str]) -> str:
    """Return a category dimension. The levels go from the inner to the outer one."""
    return (
        '<cx:strDim type="cat">'
        + "".join(_level(lvl) for lvl in levels)
        + "</cx:strDim>"
    )


def _values(values: list[str], dim_type: str = "val") -> str:
    return (
        f'<cx:numDim type="{dim_type}">'
        + _level(values, ' formatCode="General"')
        + "</cx:numDim>"
    )


def _series(
    layout: str, name: Optional[str] = "Sales", data_id: Optional[str] = "0"
) -> str:
    tx = f"<cx:tx><cx:txData><cx:v>{name}</cx:v></cx:txData></cx:tx>" if name else ""
    data = f'<cx:dataId val="{data_id}"/>' if data_id is not None else ""
    return f'<cx:series layoutId="{layout}">{tx}{data}</cx:series>'


def _chart_space(
    series: str, dimensions: str = "", title: Optional[str] = None, axes: str = ""
) -> bytes:
    title_xml = (
        f"<cx:title><cx:tx><cx:txData><cx:v>{title}</cx:v></cx:txData></cx:tx></cx:title>"
        if title
        else ""
    )
    return (
        f'<cx:chartSpace xmlns:cx="{_CX}" {_NAMESPACES}>'
        f'<cx:chartData><cx:data id="0">{dimensions}</cx:data></cx:chartData>'
        f"<cx:chart>{title_xml}<cx:plotArea><cx:plotAreaRegion>{series}"
        f"</cx:plotAreaRegion>{axes}</cx:plotArea></cx:chart></cx:chartSpace>"
    ).encode()


_WATERFALL = _chart_space(
    _series("waterfall", name="Change"),
    _categories(["Opening", "Sales", "Costs", "Closing"])
    + _values(["100", "40.000000000000001", "-15", "125"]),
    title="Profit bridge",
    axes='<cx:axis id="0"><cx:catScaling/><cx:title><cx:tx><cx:txData>'
    "<cx:v>Step</cx:v></cx:txData></cx:tx></cx:title></cx:axis>"
    '<cx:axis id="1"><cx:valScaling/></cx:axis>',
)


def _png() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (40, 20), "red").save(stream, format="PNG")
    return stream.getvalue()


def _save_chartex_deck(
    path: Path,
    chart_space: bytes,
    *,
    fallback_picture: bool = False,
    group_xfrm: Optional[str] = None,
    hidden: bool = False,
    relationship_id: Optional[str] = None,
) -> None:
    """Save a deck with a text box, an Office 2016 chart and a second text box.

    The chart frame is at 1 in from the left and 1.5 in from the top, 6 in wide
    and 4 in high, in the ``mc:Choice`` of an ``mc:AlternateContent``, as
    PowerPoint writes it. ``mc:Fallback`` holds a 40 x 20 picture with
    ``fallback_picture``, else nothing. With ``group_xfrm`` the chart is in a
    group with that ``a:xfrm``. ``relationship_id`` replaces the chart's own.
    """
    from pptx import Presentation
    from pptx.opc.package import Part
    from pptx.opc.packuri import PackURI
    from pptx.oxml import parse_xml
    from pptx.util import Inches

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.shapes.add_textbox(
        Inches(1), Inches(0.3), Inches(6), Inches(1)
    ).text_frame.text = "Before"
    slide.shapes.add_textbox(
        Inches(1), Inches(6), Inches(6), Inches(1)
    ).text_frame.text = "After"

    part = Part(
        PackURI("/ppt/charts/chartEx1.xml"),
        "application/vnd.ms-office.chartex+xml",
        prs.part.package,
        chart_space,
    )
    rid = slide.part.relate_to(
        part, "http://schemas.microsoft.com/office/2014/relationships/chartEx"
    )
    picture = None
    if fallback_picture:
        picture = slide.shapes.add_picture(BytesIO(_png()), 0, 0)._element
        picture.getparent().remove(picture)

    hidden_attribute = ' hidden="1"' if hidden else ""
    alternate_content = parse_xml(
        f"<mc:AlternateContent {_NAMESPACES}>"
        f'<mc:Choice xmlns:cx1="http://schemas.microsoft.com/office/drawing/2015/9/8/chartex" Requires="cx1">'
        f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="20" name="Chart"{hidden_attribute}/>'
        "<p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>"
        f'<p:xfrm><a:off x="{Inches(1)}" y="{Inches(1.5)}"/><a:ext cx="{Inches(6)}" cy="{Inches(4)}"/></p:xfrm>'
        f'<a:graphic><a:graphicData uri="{_CX}">'
        f'<cx:chart xmlns:cx="{_CX}" r:id="{relationship_id or rid}"/>'
        "</a:graphicData></a:graphic></p:graphicFrame></mc:Choice>"
        "<mc:Fallback/></mc:AlternateContent>"
    )
    if picture is not None:
        alternate_content[1].append(picture)

    container = slide.shapes._spTree
    if group_xfrm is not None:
        group = parse_xml(
            f'<p:grpSp {_NAMESPACES}><p:nvGrpSpPr><p:cNvPr id="30" name="Group"/>'
            f"<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr>{group_xfrm}</p:grpSpPr></p:grpSp>"
        )
        container.append(group)
        container = group
    container.append(alternate_content)
    prs.save(path)


def _grid(picture: PictureItem) -> list[list[str]]:
    assert picture.meta is not None and picture.meta.tabular_chart is not None
    return [
        [cell.text for cell in row]
        for row in picture.meta.tabular_chart.chart_data.grid
    ]


def _labels(picture: PictureItem) -> list[str]:
    assert picture.meta is not None and picture.meta.classification is not None
    return [p.class_name for p in picture.meta.classification.predictions]


def _only_chart(path: Path) -> PictureItem:
    pictures = list(get_converter().convert(path).document.pictures)
    assert len(pictures) == 1
    return pictures[0]


def test_chart_keeps_its_kind_title_data_and_place(tmp_path: Path):
    """A waterfall chart was lost with its whole graphic frame (#5366).

    It now gives a bar-chart picture between the two text boxes, with its
    title as the caption, the category axis title in the corner and each
    cached number written the short way.
    """
    path = tmp_path / "waterfall.pptx"
    _save_chartex_deck(path, _WATERFALL)

    doc = get_converter().convert(path).document
    order = [
        item.text if isinstance(item, TextItem) else "<chart>"
        for item, _ in doc.iterate_items()
        if isinstance(item, PictureItem)
        or (isinstance(item, TextItem) and item.label == DocItemLabel.PARAGRAPH)
    ]
    assert order == ["Before", "<chart>", "After"]

    picture = doc.pictures[0]
    assert picture.caption_text(doc) == "Profit bridge"
    assert _labels(picture) == [PictureClassificationLabel.BAR_CHART]
    assert _grid(picture) == [
        ["Step", "Change"],
        ["Opening", "100"],
        ["Sales", "40"],
        ["Costs", "-15"],
        ["Closing", "125"],
    ]
    assert picture.image is None
    bbox = picture.prov[0].bbox
    assert (bbox.l, bbox.t, bbox.r, bbox.b) == (
        1 * _EMU_PER_INCH,
        1.5 * _EMU_PER_INCH,
        7 * _EMU_PER_INCH,
        5.5 * _EMU_PER_INCH,
    )


def test_fallback_picture_becomes_the_chart_image(tmp_path: Path):
    """The picture in ``mc:Fallback`` is the chart's image, not a second picture."""
    path = tmp_path / "fallback.pptx"
    _save_chartex_deck(path, _WATERFALL, fallback_picture=True)

    picture = _only_chart(path)

    assert picture.image is not None
    assert (picture.image.size.width, picture.image.size.height) == (40, 20)
    assert _labels(picture) == [PictureClassificationLabel.BAR_CHART]


def test_chart_in_a_group_is_placed_on_the_slide(tmp_path: Path):
    """A chart in a group is found, and the group's transform maps its box.

    The group draws its 8 x 6 in child space at half size from (1 in, 1 in).
    """
    path = tmp_path / "group.pptx"
    inch = _EMU_PER_INCH
    _save_chartex_deck(
        path,
        _WATERFALL,
        group_xfrm=(
            f'<a:xfrm><a:off x="{inch}" y="{inch}"/><a:ext cx="{4 * inch}" cy="{3 * inch}"/>'
            f'<a:chOff x="0" y="0"/><a:chExt cx="{8 * inch}" cy="{6 * inch}"/></a:xfrm>'
        ),
    )

    picture = _only_chart(path)

    bbox = picture.prov[0].bbox
    assert (bbox.l, bbox.t, bbox.r, bbox.b) == (
        1.5 * inch,
        1.75 * inch,
        4.5 * inch,
        3.75 * inch,
    )
    assert _grid(picture)[1] == ["Opening", "100"]


def test_hidden_chart_goes_to_invisible_layer(tmp_path: Path):
    path = tmp_path / "hidden.pptx"
    _save_chartex_deck(path, _WATERFALL, hidden=True)

    doc = get_converter().convert(path).document

    assert [p.content_layer for p in doc.pictures] == [ContentLayer.INVISIBLE]


@pytest.mark.parametrize(
    ("chart_space", "relationship_id"),
    [
        pytest.param(b"<cx:chartSpace", None, id="broken-xml"),
        pytest.param(_WATERFALL, "rId99", id="missing-part"),
    ],
)
def test_unreadable_chart_still_gives_a_picture(
    tmp_path: Path, caplog, chart_space: bytes, relationship_id: Optional[str]
):
    """A chart whose part cannot be read keeps its place, without data."""
    path = tmp_path / "unreadable.pptx"
    _save_chartex_deck(path, chart_space, relationship_id=relationship_id)

    with caplog.at_level(logging.WARNING):
        picture = _only_chart(path)

    assert _labels(picture) == [PictureClassificationLabel.OTHER_CHART]
    assert picture.meta is not None and picture.meta.tabular_chart is None
    assert "Could not read an Office 2016 chart" in caplog.text


@pytest.mark.parametrize(
    ("layouts", "expected"),
    [
        (["waterfall"], [PictureClassificationLabel.BAR_CHART]),
        (["funnel"], [PictureClassificationLabel.BAR_CHART]),
        (["clusteredColumn"], [PictureClassificationLabel.BAR_CHART]),
        (
            ["clusteredColumn", "paretoLine"],
            [
                PictureClassificationLabel.BAR_CHART,
                PictureClassificationLabel.LINE_CHART,
            ],
        ),
        (["boxWhisker"], [PictureClassificationLabel.BOX_PLOT]),
        (["sunburst"], [PictureClassificationLabel.PIE_CHART]),
        (["treemap"], [PictureClassificationLabel.OTHER_CHART]),
        (["regionMap"], [PictureClassificationLabel.GEOGRAPHICAL_MAP]),
        (["someFutureLayout"], [PictureClassificationLabel.OTHER_CHART]),
    ],
)
def test_classification_follows_the_series_layouts(
    layouts: list[str], expected: list[PictureClassificationLabel]
):
    series = "".join(_series(layout) for layout in layouts)
    content = read_chartex(
        etree.fromstring(_chart_space(series, _categories(["A"]) + _values(["1"])))
    )
    assert content.labels == expected


def test_hierarchical_categories_get_one_column_per_level():
    """Treemap and sunburst categories have levels: outer level first."""
    content = read_chartex(
        etree.fromstring(
            _chart_space(
                _series("treemap", name="Revenue"),
                _categories(["Delhi", "Punjab", "Kerala"], ["North", "North", "South"])
                + _values(["4860", "1980", "1870"], dim_type="size"),
            )
        )
    )

    assert content.table is not None
    assert [[cell.text for cell in row] for row in content.table.grid] == [
        ["", "", "Revenue"],
        ["North", "Delhi", "4860"],
        ["North", "Punjab", "1980"],
        ["South", "Kerala", "1870"],
    ]
    assert [cell.row_header for cell in content.table.grid[1]] == [True, True, False]


def test_sparse_parent_labels_are_filled_from_the_row_above():
    """A parent label written only on the first row of its group covers the group.

    Excel accepts a hierarchy in this sparse form, and an empty cell means the
    parent of the row above. The cache then has no point for the empty cell.
    An empty cell after the first label of a row stays empty: "Jan" and "Mar"
    are leaves at the month level, and "2nd" is a leaf at the quarter level.
    """
    content = read_chartex(
        etree.fromstring(
            _chart_space(
                _series("sunburst", name="Sales"),
                _categories(
                    ["", "Wk.1", "Wk.2", "", ""],
                    ["Jan", "Feb", "", "Mar", ""],
                    ["1st", "", "", "", "2nd"],
                )
                + _values(["3.5", "1.2", "0.8", "2", "5"], dim_type="size"),
            )
        )
    )

    assert content.table is not None
    assert [[cell.text for cell in row] for row in content.table.grid] == [
        ["", "", "", "Sales"],
        ["1st", "Jan", "", "3.5"],
        ["1st", "Feb", "Wk.1", "1.2"],
        ["1st", "Feb", "Wk.2", "0.8"],
        ["1st", "Mar", "", "2"],
        ["2nd", "", "", "5"],
    ]


def test_values_without_categories_get_no_category_column():
    """A histogram plots a plain list of values, so its table has one column."""
    content = read_chartex(
        etree.fromstring(
            _chart_space(
                _series("clusteredColumn", name="Usage"),
                _values(["17.399999999999999", "33", "9.0999999999999996"]),
            )
        )
    )

    assert content.table is not None
    assert [[cell.text for cell in row] for row in content.table.grid] == [
        ["Usage"],
        ["17.4"],
        ["33"],
        ["9.1"],
    ]


def test_derived_series_gets_no_column():
    """The line of a Pareto chart is worked out by Office and has no data."""
    content = read_chartex(
        etree.fromstring(
            _chart_space(
                _series("clusteredColumn", name="Complaints")
                + _series("paretoLine", name=None, data_id=None),
                _categories(["Coverage", "Billing"]) + _values(["412", "268"]),
            )
        )
    )

    assert content.table is not None
    assert [[cell.text for cell in row] for row in content.table.grid] == [
        ["", "Complaints"],
        ["Coverage", "412"],
        ["Billing", "268"],
    ]


def test_point_indexes_do_not_size_the_table():
    """A huge ``ptCount`` or a bad ``idx`` cannot inflate the table.

    Only the points that are present make rows. A point with a negative,
    non-numeric or out-of-range index is dropped, and a value that is not a
    number is kept as text.
    """
    level = (
        '<cx:lvl ptCount="1000000000">'
        '<cx:pt idx="0">5</cx:pt><cx:pt idx="1">n/a</cx:pt>'
        '<cx:pt idx="-1">7</cx:pt><cx:pt idx="x">8</cx:pt>'
        '<cx:pt idx="999999999">9</cx:pt></cx:lvl>'
    )
    content = read_chartex(
        etree.fromstring(
            _chart_space(
                _series("funnel", name="Stage"),
                f'<cx:numDim type="val">{level}</cx:numDim>',
            )
        )
    )

    assert content.table is not None
    assert [[cell.text for cell in row] for row in content.table.grid] == [
        ["Stage"],
        ["5"],
        ["n/a"],
    ]


def test_part_without_a_chart_has_no_data():
    content = read_chartex(etree.fromstring(f'<cx:chartSpace xmlns:cx="{_CX}"/>'))

    assert content.labels == [PictureClassificationLabel.OTHER_CHART]
    assert content.title is None
    assert content.table is None
