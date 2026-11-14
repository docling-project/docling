# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Read the chart types that Office 2016 added, such as waterfall and treemap.

Office keeps these charts in a ``cx:chartSpace`` part (the "chartex" format),
not in the ``c:chartSpace`` part of the older chart types. The part caches the
data that the chart is drawn from. Office calculates the drawn values, such as
histogram bins, box-plot quartiles and Pareto percentages, each time it draws
the chart, so the part does not contain them.
"""

import math
from typing import Final

from docling_core.types.doc import PictureClassificationLabel, TableData
from lxml import etree

from docling.backend.utils.charts import (
    MAX_CHART_TABLE_CELLS,
    ChartContent,
    build_chart_grid,
)

CHARTEX_NAMESPACE: Final = "http://schemas.microsoft.com/office/drawing/2014/chartex"
"""The namespace of the chartex elements. It is also the ``uri`` of the
``a:graphicData`` element that holds a chartex chart in a drawing."""

_DRAWINGML_NAMESPACE: Final = "http://schemas.openxmlformats.org/drawingml/2006/main"

_LAYOUT_TO_CLASSIFICATION: Final[dict[str, PictureClassificationLabel]] = {
    "clusteredColumn": PictureClassificationLabel.BAR_CHART,
    "waterfall": PictureClassificationLabel.BAR_CHART,
    "funnel": PictureClassificationLabel.BAR_CHART,
    "paretoLine": PictureClassificationLabel.LINE_CHART,
    "boxWhisker": PictureClassificationLabel.BOX_PLOT,
    "sunburst": PictureClassificationLabel.PIE_CHART,
    "treemap": PictureClassificationLabel.OTHER_CHART,
    "regionMap": PictureClassificationLabel.GEOGRAPHICAL_MAP,
}
"""Maps the ``layoutId`` of a chartex series to a picture-classification label.
A histogram and the bars of a Pareto chart are ``clusteredColumn`` series. A
sunburst is a ring chart, like ``c:doughnutChart``. Layouts not listed fall back
to ``OTHER_CHART``."""


def _cx(name: str) -> str:
    return f"{{{CHARTEX_NAMESPACE}}}{name}"


def read_chartex(chart_space: etree._Element) -> ChartContent:
    """Read the kind, title and data of a chartex chart.

    The table has one column per category level, outer level first, and then
    one column per series value. A level has no header, except that the first
    one gets the title of the category axis. A series that Office derives from
    another series, such as the line of a Pareto chart, has no data of its own
    and gets no column.

    Args:
        chart_space: The root ``cx:chartSpace`` element of a chartex part.

    Returns:
        The chart's classification labels, title and data table. A part that
        is not a chartex chart gives ``[OTHER_CHART]`` with no title and no
        table.
    """
    chart = chart_space.find(_cx("chart"))
    if chart is None:
        return ChartContent(
            labels=[PictureClassificationLabel.OTHER_CHART], title=None, table=None
        )
    series = chart.findall(f"{_cx('plotArea')}/{_cx('plotAreaRegion')}/{_cx('series')}")
    return ChartContent(
        labels=_classifications(series),
        title=_text(chart.find(f"{_cx('title')}/{_cx('tx')}")),
        table=_table(chart_space, series, _category_axis_title(chart)),
    )


def _category_axis_title(chart: etree._Element) -> str:
    for axis in chart.iterfind(f"{_cx('plotArea')}/{_cx('axis')}"):
        if axis.find(_cx("catScaling")) is not None:
            return _text(axis.find(f"{_cx('title')}/{_cx('tx')}")) or ""
    return ""


def _classifications(
    series: list[etree._Element],
) -> list[PictureClassificationLabel]:
    labels: list[PictureClassificationLabel] = []
    for ser in series:
        label = _LAYOUT_TO_CLASSIFICATION.get(
            ser.get("layoutId", ""), PictureClassificationLabel.OTHER_CHART
        )
        if label not in labels:
            labels.append(label)
    return labels or [PictureClassificationLabel.OTHER_CHART]


def _text(tx: etree._Element | None) -> str | None:
    """Return the text of a ``cx:tx`` element, or None when it has none.

    The text is either a plain value (``cx:txData/cx:v``) or rich text
    (``cx:rich``) whose paragraphs are joined with line breaks.
    """
    if tx is None:
        return None
    value = tx.find(f"{_cx('txData')}/{_cx('v')}")
    if value is not None:
        text = value.text or ""
    else:
        text = "\n".join(
            "".join(
                run.text or "" for run in paragraph.iter(f"{{{_DRAWINGML_NAMESPACE}}}t")
            )
            for paragraph in tx.iter(f"{{{_DRAWINGML_NAMESPACE}}}p")
        )
    return text.strip() or None


def _table(
    chart_space: etree._Element, series: list[etree._Element], corner: str
) -> TableData | None:
    data_by_id = {
        data.get("id"): data
        for data in chart_space.iterfind(f"{_cx('chartData')}/{_cx('data')}")
    }
    category_columns: list[tuple[str, list[str]]] | None = None
    value_columns: list[tuple[str, list[str]]] = []
    for ser in series:
        data_id = ser.find(_cx("dataId"))
        if data_id is None or data_id.get("val") not in data_by_id:
            continue
        data = data_by_id[data_id.get("val")]
        if category_columns is None:
            category_columns = _category_columns(data, corner)
        name = _text(ser.find(_cx("tx"))) or ""
        dimensions = [
            dim for dim in data.iterchildren(_cx("numDim")) if dim.get("type") != "cat"
        ]
        for dim in dimensions:
            header = name if len(dimensions) == 1 else f"{name} ({dim.get('type')})"
            value_columns.append((header, _level_values(dim.find(_cx("lvl")), True)))
    return build_chart_grid(category_columns or [], value_columns)


def _category_columns(data: etree._Element, corner: str) -> list[tuple[str, list[str]]]:
    """Return the category levels of a ``cx:data`` element, outer level first."""
    for dim in data.iterchildren(_cx("strDim"), _cx("numDim")):
        if dim.get("type") == "cat":
            numeric = dim.tag == _cx("numDim")
            # The cx:lvl elements run from the inner level to the outer level.
            levels = [
                _level_values(level, numeric) for level in dim.iterchildren(_cx("lvl"))
            ][::-1]
            return [
                (corner if index == 0 else "", values)
                for index, values in enumerate(levels)
            ]
    return []


def _level_values(level: etree._Element | None, numeric: bool) -> list[str]:
    """Return the points of a ``cx:lvl`` element, one entry per index.

    The list is sized by the points that are present, not by ``ptCount``, so a
    part cannot make it larger than ``MAX_CHART_TABLE_CELLS`` entries.
    """
    if level is None:
        return []
    points: dict[int, str] = {}
    for point in level.iterchildren(_cx("pt")):
        index_text = point.get("idx", "")
        if not (index_text.isascii() and index_text.isdigit()):
            continue
        index = int(index_text)
        if index >= MAX_CHART_TABLE_CELLS:
            continue
        text = point.text or ""
        points[index] = _number_text(text) if numeric else text
    values = [""] * (max(points) + 1 if points else 0)
    for index, text in points.items():
        values[index] = text
    return values


def _number_text(text: str) -> str:
    """Write a cached number the short way, so ``118.40000000000001`` is ``118.4``.

    Whole numbers lose their ``.0``. Text that is not a finite number stays as
    it is.
    """
    try:
        value = float(text)
    except ValueError:
        return text
    if not math.isfinite(value):
        return text
    if value.is_integer():
        return str(int(value))
    return str(value)
