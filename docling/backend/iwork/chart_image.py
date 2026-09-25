# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Draw a picture of a chart, which an iWork document does not hold.

Keynote keeps a chart as the model it is drawn from and never as a picture, and
LibreOffice's Keynote import skips charts altogether, so there is no picture to
extract and none LibreOffice can make out of the ``.key`` itself. What
LibreOffice can draw is an Office chart, which is how the Office backends render
theirs, so the chart is rebuilt from the data read out of the document as a
DrawingML chart on a page of its own, and rendered along the route those
backends take: to PDF with LibreOffice, to pixels with pypdfium2, and cropped to
what was drawn.

The page is a Word document of five small parts, written here rather than with
python-docx or python-pptx: neither ships with the iWork extra, and the little
of either this would use is the XML below.

The rebuilt chart has the original's kind, data, title and proportions, but not
its colours or fonts, which live in style archives this does not read. A chart
an Office chart has no equivalent for gets no picture rather than a misleading
one: a mixed or two-axis chart, which draws its series in different ways; a
bubble chart, whose series no fixture shows how Keynote groups into x, y and
size; and an interactive chart, which shows one of its data sets at a time.
"""

import logging
import shutil
import zipfile
from collections.abc import Callable, Sequence
from io import BytesIO
from pathlib import Path
from tempfile import mkdtemp
from xml.sax.saxutils import escape

from PIL import Image

from docling.backend.docx.drawingml.utils import crop_whitespace
from docling.backend.iwork.content import Chart, ChartKind, ChartSeries, Geometry

_log = logging.getLogger(__name__)

try:  # pragma: no cover - import-time guard
    import pypdfium2
except ImportError:  # pragma: no cover - import-time guard
    pass

DEFAULT_CHART_WIDTH = 480.0

DEFAULT_CHART_HEIGHT = 360.0
"""The size a chart is drawn at when the document does not say, in points."""

MAX_CHART_SIDE = 1440.0
"""The longest side a chart is drawn at, in points; a larger one is scaled down.

Twenty inches, which keeps the page inside the 22 inches Word allows while
leaving room for the margin around the chart.
"""

PAGE_MARGIN = 36.0
"""The margin around the chart on its page, in points.

The chart sits in a paragraph, so a page cut exactly to its size would push it
onto a second one; the margin is cropped away with the rest of the whitespace.
"""

RENDER_SCALE = 2
"""Pixels per point of the rebuilt chart, the scale the PowerPoint backend uses."""

PALETTE = ("4472C4", "ED7D31", "A5A5A5", "FFC000", "5B9BD5", "70AD47")
"""The colours series are drawn in, in turn: Office's default accent colours.

The chart's own colours live in style archives this does not read, and the page
carries no theme for LibreOffice to take automatic colours from, which it
answers by drawing bars and wedges with no fill at all.
"""

TEXT_SIZE = 900

TITLE_SIZE = 1200
"""Font sizes of a chart's labels and of its title, in hundredths of a point.

Set rather than left to LibreOffice, whose defaults are sized for a full page
and crowd a chart drawn at the few inches a slide gives it.
"""

AXIS_COLOUR = "BFBFBF"

GRIDLINE_COLOUR = "D9D9D9"

EMU_PER_POINT = 12700

TWIPS_PER_POINT = 20

_PACKAGE = "http://schemas.openxmlformats.org/package/2006"

_OFFICE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

_DRAWINGML = "http://schemas.openxmlformats.org/drawingml/2006"

_WORD = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

_CONTENT_TYPES = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="{_PACKAGE}/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/charts/chart1.xml" ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>
</Types>"""

_PACKAGE_RELS = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{_PACKAGE}/relationships">
<Relationship Id="rId1" Type="{_OFFICE}/officeDocument" Target="word/document.xml"/>
</Relationships>"""

_DOCUMENT_RELS = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{_PACKAGE}/relationships">
<Relationship Id="rId1" Type="{_OFFICE}/chart" Target="charts/chart1.xml"/>
</Relationships>"""

_AXIS_IDS = '<c:axId val="1"/><c:axId val="2"/>'

_UNSMOOTHED = '<c:smooth val="0"/>'


def chart_document(chart: Chart, geometry: Geometry | None) -> bytes | None:
    """Rebuild a chart as a Word document holding nothing but that chart.

    Args:
        chart: The chart to rebuild.
        geometry: Where the chart sits in the document, whose size it is drawn
            at, or None to use a default size.

    Returns:
        The ``.docx`` file, or None when an Office chart has no equivalent for
        the chart or the chart holds nothing to plot.
    """
    plot = _plot(chart)
    if plot is None:
        return None

    width, height = _drawn_size(geometry)
    # A pie's legend names its wedges; any other chart's names its series, which
    # is only worth drawing when there is more than one to tell apart.
    with_legend = plot.count("<c:ser>") > 1 or chart.kind in (
        ChartKind.PIE,
        ChartKind.DONUT,
    )
    space = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<c:chartSpace xmlns:c="{_DRAWINGML}/chart" xmlns:a="{_DRAWINGML}/main">'
        f'<c:roundedCorners val="0"/><c:chart>{_title(chart.title)}'
        f"<c:plotArea><c:layout/>{plot}</c:plotArea>"
        + (
            '<c:legend><c:legendPos val="r"/><c:overlay val="0"/></c:legend>'
            if with_legend
            else ""
        )
        + '<c:plotVisOnly val="1"/><c:dispBlanksAs val="gap"/></c:chart>'
        f'<c:txPr><a:bodyPr/><a:lstStyle/><a:p><a:pPr><a:defRPr sz="{TEXT_SIZE}"/>'
        '</a:pPr><a:endParaRPr lang="en-US"/></a:p></c:txPr></c:chartSpace>'
    )
    margin = int(PAGE_MARGIN * TWIPS_PER_POINT)
    document = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_WORD}" xmlns:r="{_OFFICE}"'
        f' xmlns:wp="{_DRAWINGML}/wordprocessingDrawing"'
        f' xmlns:a="{_DRAWINGML}/main" xmlns:c="{_DRAWINGML}/chart">'
        "<w:body><w:p><w:r><w:drawing>"
        '<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{int(width * EMU_PER_POINT)}"'
        f' cy="{int(height * EMU_PER_POINT)}"/>'
        '<wp:docPr id="1" name="Chart 1"/>'
        f'<a:graphic><a:graphicData uri="{_DRAWINGML}/chart">'
        '<c:chart r:id="rId1"/></a:graphicData></a:graphic>'
        "</wp:inline></w:drawing></w:r></w:p>"
        f'<w:sectPr><w:pgSz w:w="{int((width + 2 * PAGE_MARGIN) * TWIPS_PER_POINT)}"'
        f' w:h="{int((height + 2 * PAGE_MARGIN) * TWIPS_PER_POINT)}"/>'
        f'<w:pgMar w:top="{margin}" w:right="{margin}" w:bottom="{margin}"'
        f' w:left="{margin}" w:header="0" w:footer="0" w:gutter="0"/>'
        "</w:sectPr></w:body></w:document>"
    )

    out = BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", _CONTENT_TYPES)
        package.writestr("_rels/.rels", _PACKAGE_RELS)
        package.writestr("word/_rels/document.xml.rels", _DOCUMENT_RELS)
        package.writestr("word/document.xml", document)
        package.writestr("word/charts/chart1.xml", space)
    return out.getvalue()


def render_chart(
    chart: Chart,
    geometry: Geometry | None,
    converter: Callable[[Path, Path], None],
) -> Image.Image | None:
    """Draw a chart by rebuilding it for LibreOffice, and crop what it drew.

    Args:
        chart: The chart to draw.
        geometry: Where the chart sits in the document, or None.
        converter: Converts a Word document to PDF, as
            ``get_docx_to_pdf_converter`` returns one.

    Returns:
        The picture, or None when the chart cannot be rebuilt or the
        conversion fails.
    """
    document = chart_document(chart, geometry)
    if document is None:
        return None

    temp_dir = Path(mkdtemp())
    try:
        input_path = temp_dir / "chart.docx"
        output_path = temp_dir / "chart.pdf"
        input_path.write_bytes(document)
        converter(input_path, output_path)
        if not output_path.exists():
            _log.debug("LibreOffice produced no PDF output for an iWork chart")
            return None
        pdf = pypdfium2.PdfDocument(str(output_path))
        page = pdf[0]
        image = crop_whitespace(page.render(scale=RENDER_SCALE).to_pil())
        page.close()
        pdf.close()
        return image
    except Exception as exc:
        _log.debug("iWork chart rendering via LibreOffice failed: %s", exc)
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _plot(chart: Chart) -> str | None:
    """Write the DrawingML plot of a chart, with its axes, or None if it has none.

    Returns:
        The chart-type element and the axes it plots against, or None when an
        Office chart has no equivalent for the chart or it holds nothing to plot.
    """
    if chart.interactive:
        return None

    if chart.kind in (ChartKind.COLUMN, ChartKind.BAR):
        if not _plottable(chart):
            return None
        grouping = "stacked" if chart.stacked else "clustered"
        direction = "col" if chart.kind == ChartKind.COLUMN else "bar"
        return (
            f'<c:barChart><c:barDir val="{direction}"/><c:grouping val="{grouping}"/>'
            f'<c:varyColors val="0"/>{_category_series(chart)}'
            + ('<c:overlap val="100"/>' if chart.stacked else "")
            + f"{_AXIS_IDS}</c:barChart>"
            + _category_axes(horizontal=chart.kind == ChartKind.BAR)
        )
    if chart.kind == ChartKind.LINE:
        if not _plottable(chart):
            return None
        return (
            '<c:lineChart><c:grouping val="standard"/><c:varyColors val="0"/>'
            # LibreOffice smooths a line series that does not say otherwise.
            f"{_category_series(chart, lines=True, tail=_UNSMOOTHED)}"
            '<c:marker val="1"/>'
            f"{_AXIS_IDS}</c:lineChart>{_category_axes()}"
        )
    if chart.kind == ChartKind.AREA:
        if not _plottable(chart):
            return None
        grouping = "stacked" if chart.stacked else "standard"
        return (
            f'<c:areaChart><c:grouping val="{grouping}"/><c:varyColors val="0"/>'
            f"{_category_series(chart)}{_AXIS_IDS}</c:areaChart>{_category_axes()}"
        )
    if chart.kind == ChartKind.RADAR:
        if not _plottable(chart):
            return None
        return (
            '<c:radarChart><c:radarStyle val="marker"/><c:varyColors val="0"/>'
            f"{_category_series(chart, lines=True)}{_AXIS_IDS}</c:radarChart>"
            f"{_category_axes()}"
        )
    if chart.kind in (ChartKind.PIE, ChartKind.DONUT):
        wedges = _wedges(chart)
        if wedges is None:
            return None
        if chart.kind == ChartKind.PIE:
            return (
                f'<c:pieChart><c:varyColors val="1"/>{wedges}'
                '<c:firstSliceAng val="0"/></c:pieChart>'
            )
        return (
            f'<c:doughnutChart><c:varyColors val="1"/>{wedges}'
            '<c:firstSliceAng val="0"/><c:holeSize val="50"/></c:doughnutChart>'
        )
    if chart.kind == ChartKind.SCATTER:
        points = _points(chart)
        if points is None:
            return None
        return (
            '<c:scatterChart><c:scatterStyle val="lineMarker"/>'
            f'<c:varyColors val="0"/>{points}{_AXIS_IDS}</c:scatterChart>'
            + _value_axis(1, 2, "b")
            + _value_axis(2, 1, "l")
        )
    return None


def _plottable(chart: Chart) -> bool:
    """Report whether a chart has categories and a value in any of them."""
    return bool(chart.categories) and any(
        value is not None for series in chart.series for value in series.values
    )


def _category_series(chart: Chart, *, lines: bool = False, tail: str = "") -> str:
    """Write every series of a chart as values across its categories.

    Args:
        chart: The chart whose series to write.
        lines: Whether the series are drawn as lines without markers, as a line
            or radar chart draws them, rather than as filled shapes.
        tail: What closes each series after its values, for the chart types
            whose series say more there.

    Returns:
        One ``c:ser`` per series.
    """
    return "".join(
        f'<c:ser><c:idx val="{index}"/><c:order val="{index}"/>'
        f"{_series_name(series.name)}"
        + (
            f'<c:spPr><a:ln w="28575" cap="rnd">{_fill(index)}<a:round/></a:ln>'
            '</c:spPr><c:marker><c:symbol val="none"/></c:marker>'
            if lines
            else f"<c:spPr>{_fill(index)}</c:spPr>"
        )
        + f"<c:cat>{_strings(chart.categories)}</c:cat>"
        f"<c:val>{_numbers(series.values)}</c:val>{tail}</c:ser>"
        for index, series in enumerate(chart.series)
    )


def _wedges(chart: Chart) -> str | None:
    """Write a pie or donut the way Keynote draws one.

    Keynote draws a wedge per series, sized by the series' first value, where an
    Office chart draws a wedge per category of a single series; so the series
    become the categories of the one series written here.
    """
    values = [series.values[0] if series.values else None for series in chart.series]
    if all(value is None for value in values):
        return None
    name = chart.categories[0] if chart.categories else ""
    points = "".join(
        f'<c:dPt><c:idx val="{index}"/><c:bubble3D val="0"/><c:spPr>{_fill(index)}'
        '<a:ln w="12700"><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill>'
        "</a:ln></c:spPr></c:dPt>"
        for index in range(len(values))
    )
    return (
        f'<c:ser><c:idx val="0"/><c:order val="0"/>{_series_name(name)}{points}'
        f"<c:cat>{_strings([series.name for series in chart.series])}</c:cat>"
        f"<c:val>{_numbers(values)}</c:val></c:ser>"
    )


def _points(chart: Chart) -> str | None:
    """Pair a scatter chart's series up into the points an Office chart plots.

    Each point takes its x from the first series when the chart shares one, and
    otherwise from the series before its own; a point missing either value is
    not drawn, as Keynote does not draw it.
    """
    series = chart.series
    if chart.shared_x:
        pairs: list[tuple[ChartSeries, ChartSeries]] = [
            (series[0], ys) for ys in series[1:]
        ]
    else:
        pairs = list(zip(series[::2], series[1::2]))

    written = []
    for xs, ys in pairs:
        points = [
            (x, y)
            for x, y in zip(xs.values, ys.values)
            if x is not None and y is not None
        ]
        if not points:
            continue
        index = len(written)
        written.append(
            f'<c:ser><c:idx val="{index}"/><c:order val="{index}"/>'
            f"{_series_name(ys.name)}"
            '<c:spPr><a:ln w="19050"><a:noFill/></a:ln></c:spPr>'
            '<c:marker><c:symbol val="circle"/><c:size val="5"/>'
            f"<c:spPr>{_fill(index)}<a:ln><a:noFill/></a:ln></c:spPr></c:marker>"
            f"<c:xVal>{_numbers([x for x, _ in points])}</c:xVal>"
            f"<c:yVal>{_numbers([y for _, y in points])}</c:yVal>"
            '<c:smooth val="0"/></c:ser>'
        )
    return "".join(written) if written else None


def _category_axes(*, horizontal: bool = False) -> str:
    """Write a category axis and the value axis crossing it.

    Args:
        horizontal: Whether the categories run down the side, as a bar chart's
            do, rather than along the bottom.
    """
    return (
        '<c:catAx><c:axId val="1"/><c:scaling><c:orientation val="minMax"/>'
        f'</c:scaling><c:delete val="0"/><c:axPos val="{"l" if horizontal else "b"}"/>'
        '<c:majorTickMark val="out"/><c:minorTickMark val="none"/>'
        f'<c:tickLblPos val="nextTo"/>{_axis_line()}<c:crossAx val="2"/>'
        '<c:crosses val="autoZero"/><c:auto val="1"/><c:lblAlgn val="ctr"/>'
        '<c:lblOffset val="100"/></c:catAx>'
    ) + _value_axis(2, 1, "b" if horizontal else "l", gridlines=True)


def _value_axis(axis: int, crosses: int, position: str, gridlines: bool = False) -> str:
    """Write one value axis, crossing the axis numbered ``crosses``."""
    return (
        f'<c:valAx><c:axId val="{axis}"/><c:scaling><c:orientation val="minMax"/>'
        f'</c:scaling><c:delete val="0"/><c:axPos val="{position}"/>'
        + (
            f'<c:majorGridlines><c:spPr><a:ln w="6350"><a:solidFill><a:srgbClr'
            f' val="{GRIDLINE_COLOUR}"/></a:solidFill></a:ln></c:spPr></c:majorGridlines>'
            if gridlines
            else ""
        )
        + '<c:numFmt formatCode="General" sourceLinked="0"/>'
        '<c:majorTickMark val="out"/><c:minorTickMark val="none"/>'
        f'<c:tickLblPos val="nextTo"/>{_axis_line()}<c:crossAx val="{crosses}"/>'
        '<c:crosses val="autoZero"/><c:crossBetween val="between"/></c:valAx>'
    )


def _title(title: str | None) -> str:
    """Write a chart's title, or say it has none so none is made up for it."""
    if not title:
        return '<c:autoTitleDeleted val="1"/>'
    return (
        f'<c:title><c:tx><c:rich><a:bodyPr/><a:p><a:pPr><a:defRPr sz="{TITLE_SIZE}"'
        f' b="1"/></a:pPr><a:r><a:rPr lang="en-US" sz="{TITLE_SIZE}" b="1"/>'
        f"<a:t>{escape(title)}</a:t></a:r></a:p></c:rich></c:tx>"
        '<c:overlay val="0"/></c:title><c:autoTitleDeleted val="0"/>'
    )


def _fill(index: int) -> str:
    """Write the solid fill of the ``index``-th series or wedge."""
    colour = PALETTE[index % len(PALETTE)]
    return f'<a:solidFill><a:srgbClr val="{colour}"/></a:solidFill>'


def _axis_line() -> str:
    return (
        f'<c:spPr><a:ln w="9525"><a:solidFill><a:srgbClr val="{AXIS_COLOUR}"/>'
        "</a:solidFill></a:ln></c:spPr>"
    )


def _series_name(name: str) -> str:
    return f"<c:tx><c:v>{escape(name)}</c:v></c:tx>"


def _strings(values: Sequence[str]) -> str:
    """Write labels as a DrawingML string literal."""
    points = "".join(
        f'<c:pt idx="{index}"><c:v>{escape(value)}</c:v></c:pt>'
        for index, value in enumerate(values)
    )
    return f'<c:strLit><c:ptCount val="{len(values)}"/>{points}</c:strLit>'


def _numbers(values: Sequence[float | None]) -> str:
    """Write values as a DrawingML number literal, leaving out the missing ones."""
    points = "".join(
        f'<c:pt idx="{index}"><c:v>{value!r}</c:v></c:pt>'
        for index, value in enumerate(values)
        if value is not None
    )
    return (
        "<c:numLit><c:formatCode>General</c:formatCode>"
        f'<c:ptCount val="{len(values)}"/>{points}</c:numLit>'
    )


def _drawn_size(geometry: Geometry | None) -> tuple[float, float]:
    """The size to draw a chart at: its own, scaled down to fit if need be."""
    if geometry is None or geometry.width < 1 or geometry.height < 1:
        return DEFAULT_CHART_WIDTH, DEFAULT_CHART_HEIGHT
    scale = min(1.0, MAX_CHART_SIDE / geometry.width, MAX_CHART_SIDE / geometry.height)
    return geometry.width * scale, geometry.height * scale
