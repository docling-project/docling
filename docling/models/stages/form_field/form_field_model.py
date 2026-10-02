# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Native AcroForm widgets keyed to their printed captions.

The stage runs after table structure. For every page with widgets it passes
the widgets, layout clusters and detected table cells to ``keying.assign``,
which chooses the caption of each value, and turns the result into
``FieldRegionPrediction``s: one item per association, with the caption as key.
Outside detected tables, a value in a grid of like values also keeps the
caption along its other axis as context. Items are placed as the original
stage placed its values: a paragraph that inlines all of an item's widgets
hosts the item in place, otherwise the enclosing FORM region does, otherwise a
page-wide region. A value in a cell of a detected table goes into that cell
instead (``page.predictions.table_fields``), keyed only by text printed in the
same cell and without context: the table's headers already carry the
association. Caption text
that became a key leaves the body, also when it shares a text block with
text that did not, and text blocks that merely re-render a filled value are
dropped. A page whose keying fails keeps its values, without keys.
"""

import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from docling_core.types.doc import BoundingBox, DocItemLabel
from docling_core.types.doc.page import PdfWidget

from docling.datamodel.base_models import (
    Cluster,
    FieldItemPrediction,
    FieldRegionPrediction,
    FieldValuePrediction,
    Page,
    TableFieldPrediction,
)
from docling.datamodel.document import ConversionResult
from docling.models.base_layout_model import PAGE_HEADER_LABELS, TEXT_ELEM_LABELS
from docling.models.base_model import BasePageModel
from docling.models.stages.form_field.keying import (
    PUSHBUTTON_FLAG,
    WIDGET_COVERAGE,
    Assignment,
    Label,
    assign,
    is_skipped,
    paints_value,
    regions,
)
from docling.models.stages.form_field.keying.rules import THIN
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)

# Paragraphs that may host an item in place. Code, and captions or footnotes
# attached to a table or picture, take reading-order paths that ignore
# TextElement.field_item.
INLINE_HOSTS = set(TEXT_ELEM_LABELS) - {
    DocItemLabel.CODE,
    DocItemLabel.CAPTION,
    DocItemLabel.FOOTNOTE,
    *PAGE_HEADER_LABELS,
}


@dataclass
class _Unit:
    """One field item built from one association, or from one unkeyed value."""

    item: FieldItemPrediction
    positions: list[int]  # indices into Assignment.values
    consumed: Label | None  # label whose text leaves the body


@dataclass
class _Plan:
    """Everything the stage writes to a page, computed before any edit."""

    regions: list[FieldRegionPrediction]
    dropped: set[int]  # clusters whose text all became keys
    trimmed: dict[int, set[int]]  # cluster id -> its cells used as keys or values
    hosts: set[int]  # paragraphs that carry an item in place; never dropped
    values: list[FieldValuePrediction]
    table_fields: list[TableFieldPrediction] = field(default_factory=list)


def _walk(clusters: list[Cluster]) -> dict[int, Cluster]:
    """Every cluster by id, children included."""
    return {cluster.id: cluster for cluster in regions(clusters)}


def _printable(cluster: Cluster) -> set[int]:
    return {cell.index for cell in cluster.cells if cell.text.strip()}


def _assign(page: Page) -> Assignment:
    """Key the page's widgets from its layout, table structure and printed rules."""
    assert page.size is not None and page.parsed_page is not None
    assert page.predictions.layout is not None
    structure = page.predictions.tablestructure
    table_cells = (
        {}
        if structure is None
        else {
            table_id: table.table_cells
            for table_id, table in structure.table_map.items()
        }
    )
    # Printed rules only place values in the cells of detected tables. Both
    # queries are needed: stroked segments give the edges of a cell border
    # drawn as one rectangle (its own box is not thin), and thin boxes give
    # rules drawn as filled rectangles. Only ThreadedDoclingParsePageBackend,
    # the default backend, answers the second; elsewhere it returns None.
    rules: list[BoundingBox] = []
    if table_cells and page._backend is not None:
        rules = [
            *(page._backend.get_shape_lines() or []),
            *(page._backend.get_thin_shape_boxes(max_thickness=THIN) or []),
        ]
    return assign(
        page.parsed_page.widgets,
        page.predictions.layout.clusters,
        table_cells,
        page.size.height,
        rules,
    )


def _atom_sources(
    assignment: Assignment,
) -> tuple[dict[int, tuple[int, int]], set[int]]:
    """Atom -> (cluster id, cell index), plus clusters that must stay in the body.

    A cell the layout put in two clusters leaves both clusters in place
    ("unsure").
    """
    sources = {atom: min(found) for atom, found in assignment.sources.items()}
    unsure = {
        cluster_id
        for found in assignment.sources.values()
        if len({cluster_id for cluster_id, _ in found}) > 1
        for cluster_id, _ in found
    }
    return sources, unsure


def _build_units(
    assignment: Assignment, values: list[FieldValuePrediction]
) -> list[_Unit]:
    """One field item per selected association, plus one per unkeyed value."""
    units: list[_Unit] = []
    owned: set[int] = set()
    for candidate in (assignment.candidates[i] for i in assignment.selected):
        # A group question stays as ordinary text: each option keeps its own
        # caption as key.
        if candidate.kind == "choice_group":
            continue
        label = assignment.labels[candidate.label]
        units.append(
            _Unit(
                FieldItemPrediction(
                    key_text=label.text,
                    key_bbox=label.bbox,
                    values=[values[m] for m in candidate.members],
                    context_text=""
                    if candidate.context is None
                    else assignment.labels[candidate.context].text,
                ),
                list(candidate.members),
                # Table-cell labels have no atoms: the table keeps its text.
                label if label.atoms else None,
            )
        )
        owned.update(candidate.members)
    units += [
        _Unit(FieldItemPrediction(values=[value]), [m], None)
        for m, value in enumerate(values)
        if m not in owned
    ]
    return sorted(units, key=lambda u: min(u.positions))


def _cell_units(
    units: list[_Unit], assignment: Assignment
) -> tuple[list[TableFieldPrediction], list[_Unit]]:
    """Units whose values all sit in one table cell, grouped into that cell.

    The table's own row and column headers carry the association there, and
    keying.tables keys such a value only by text of its own cell, so the item
    keeps its key and drops any context. Returns the cells, then the units
    placed as before.
    """
    cells: dict[tuple[int, tuple[int, int], tuple[int, int]], list[_Unit]] = (
        defaultdict(list)
    )
    rest: list[_Unit] = []
    for unit in units:
        slots = {assignment.slots.get(m) for m in unit.positions}
        if len(slots) != 1 or None in slots:
            rest.append(unit)
            continue
        (slot,) = slots
        assert slot is not None
        unit.item = unit.item.model_copy(update={"context_text": ""})
        cells[slot.table, slot.rows, slot.columns].append(unit)
    placed = [
        TableFieldPrediction(
            table_id=table,
            start_row_offset_idx=rows[0],
            end_row_offset_idx=rows[1],
            start_col_offset_idx=columns[0],
            end_col_offset_idx=columns[1],
            items=[unit.item for unit in members],
        )
        for (table, rows, columns), members in cells.items()
    ]
    return placed, rest


def _inline_hosts(
    units: list[_Unit],
    assignment: Assignment,
    sources: dict[int, tuple[int, int]],
    page: Page,
    unsure: set[int],
) -> dict[int, Cluster]:
    """Units whose key is one whole paragraph holding all their widgets.

    Such a paragraph stays in place as a field item (see TextElement.field_item);
    page assembly attaches one item per paragraph.
    """
    assert page.predictions.layout is not None
    top = {c.id: c for c in page.predictions.layout.clusters}
    users: dict[int, set[int]] = defaultdict(set)
    for k, unit in enumerate(units):
        if unit.consumed is not None:
            for atom in unit.consumed.atoms:
                users[sources[atom][0]].add(k)
    hosts: dict[int, Cluster] = {}
    for k, unit in enumerate(units):
        if unit.consumed is None:
            continue
        clusters = {sources[atom][0] for atom in unit.consumed.atoms}
        if len(clusters) != 1:
            continue
        (cluster_id,) = clusters
        host = top.get(cluster_id)
        if (
            host is not None
            and cluster_id not in unsure
            and host.label in INLINE_HOSTS
            and users[cluster_id] == {k}
            and {sources[atom][1] for atom in unit.consumed.atoms} == _printable(host)
            and all(
                assignment.values[m].bbox.intersection_over_self(host.bbox)
                >= WIDGET_COVERAGE
                for m in unit.positions
            )
        ):
            hosts[k] = host
    return hosts


def _place(
    units: list[_Unit],
    hosts: dict[int, Cluster],
    assignment: Assignment,
    page: Page,
) -> list[FieldRegionPrediction]:
    """The three placement routes: inline paragraph, FORM region, page-wide."""
    regions: list[FieldRegionPrediction] = []
    rest: list[tuple[BoundingBox, FieldItemPrediction]] = []
    for k, unit in enumerate(units):
        host = hosts.get(k)
        if host is not None:
            regions.append(
                FieldRegionPrediction(
                    source_container_id=host.id, bbox=host.bbox, items=[unit.item]
                )
            )
            continue
        rest.append((assignment.values[min(unit.positions)].bbox, unit.item))
    return regions + _form_regions(rest, page)


def _form_regions(
    items: list[tuple[BoundingBox, FieldItemPrediction]], page: Page
) -> list[FieldRegionPrediction]:
    """Items grouped by the FORM region holding their box; the rest page-wide."""
    assert page.predictions.layout is not None
    forms = [
        c for c in page.predictions.layout.clusters if c.label == DocItemLabel.FORM
    ]
    by_form: dict[int, list[FieldItemPrediction]] = defaultdict(list)
    loose: list[FieldItemPrediction] = []
    for bbox, item in items:
        form = PdfFormFieldModel._match_form(bbox, forms)
        (loose if form is None else by_form[form.id]).append(item)
    boxes = {form.id: form.bbox for form in forms}
    regions = [
        FieldRegionPrediction(
            source_container_id=form_id, bbox=boxes[form_id], items=items
        )
        for form_id, items in by_form.items()
    ]
    if loose:
        regions.append(
            FieldRegionPrediction(
                bbox=BoundingBox.enclosing_bbox(
                    [value.bbox for item in loose for value in item.values]
                ),
                items=loose,
            )
        )
    return regions


def _table_children(page: Page) -> set[int]:
    """Text blocks inside a structured table: its TableFormer cells show the text."""
    assert page.predictions.layout is not None
    return {
        child.id
        for cluster in _walk(page.predictions.layout.clusters).values()
        if cluster.label in {DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX}
        for child in cluster.children
    }


def _consumed_clusters(
    units: list[_Unit],
    sources: dict[int, tuple[int, int]],
    assignment: Assignment,
    page: Page,
    keep: set[int],
) -> tuple[set[int], dict[int, set[int]]]:
    """Clusters whose text became key text, wholly or in part.

    Returns the clusters whose every printable cell became a key, which leave
    the body, then the clusters only partly used, with the cells that became
    keys: those cells leave the cluster and the rest stays in the body. In a
    cluster that gives a key, a cell that merely repeats a filled value (see
    keying.paints_value) counts as used too, so no paragraph is left holding
    only the values of the fields its captions keyed.
    """
    assert page.size is not None and page.predictions.layout is not None
    clusters = _walk(page.predictions.layout.clusters)
    used: dict[int, set[int]] = defaultdict(set)
    for unit in units:
        if unit.consumed is not None:
            for atom in unit.consumed.atoms:
                cluster_id, cell_index = sources[atom]
                used[cluster_id].add(cell_index)
    dropped: set[int] = set()
    trimmed: dict[int, set[int]] = {}
    for cluster_id, cells in used.items():
        if cluster_id in keep:
            continue
        cluster = clusters[cluster_id]
        cells = cells | {
            cell.index
            for cell in cluster.cells
            if cell.text.strip()
            and paints_value(
                cell.rect.to_bounding_box().to_top_left_origin(page.size.height),
                cell.text.strip(),
                assignment.values,
            )
        }
        if cells >= _printable(cluster):
            dropped.add(cluster_id)
        else:
            trimmed[cluster_id] = cells
    return dropped, trimmed


class PdfFormFieldModel(BasePageModel):
    # A rendered-text cluster counts as a widget's duplicate only when this much
    # of it sits inside the widget rect. Guards against deleting ordinary printed
    # text that merely equals a field value by coincidence.
    _DUPLICATE_CONTAINMENT_THRESHOLD = 0.6

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "".join(text.split())

    @classmethod
    def _cluster_text(cls, cluster: Cluster) -> str:
        return cls._normalize_text(
            " ".join(cell.text for cell in cluster.cells if cell.text.strip())
        )

    @classmethod
    def _normalize_widget(
        cls, widget: PdfWidget, bbox: BoundingBox
    ) -> FieldValuePrediction:
        source_value = widget.widget_text or ""
        if (
            widget.widget_field_type == "/Btn"
            and not widget.widget_field_flags & PUSHBUTTON_FLAG
        ):
            # A checkbox/radio widget carries state, not text. Encode the state
            # as a nested checkbox child (see FieldValuePrediction.checkbox) and
            # keep the value text empty -- the serializer only inlines the
            # <checkbox> token when the hosting value has no text of its own.
            source_value = widget.widget_appearance_state or source_value
            off = source_value in {"", "/Off", "Off"}
            return FieldValuePrediction(
                text="",
                orig=source_value,
                bbox=bbox,
                checkbox="unselected" if off else "selected",
            )

        return FieldValuePrediction(text=source_value, orig=source_value, bbox=bbox)

    @classmethod
    def _match_form(
        cls, widget_bbox: BoundingBox, forms: list[Cluster]
    ) -> Cluster | None:
        matches = [
            (widget_bbox.intersection_over_self(form.bbox), form)
            for form in forms
            if widget_bbox.intersection_over_self(form.bbox) > WIDGET_COVERAGE
        ]
        if not matches:
            return None
        return max(
            matches,
            key=lambda match: (
                match[0],
                -match[1].bbox.area(),
                -match[1].id,
            ),
        )[1]

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        for page in page_batch:
            if (
                not self.enabled
                or page.parsed_page is None
                or not page.parsed_page.widgets
            ):
                yield page
                continue

            with TimeRecorder(conv_res, "form_field"):
                self._key_page(page)

            yield page

    def _key_page(self, page: Page) -> None:
        """Key the page's widgets; on any failure keep the values without keys.

        A stage exception would mark the page as failed and the document as
        partially converted, so the widgets' values are worth more than the
        keys: the fallback plan places them unkeyed, and if even that fails
        the page passes through untouched.
        """
        try:
            plan = self._plan_page(page)
        except Exception:
            _log.warning(
                "Form field keying failed on page %d; its values are kept without keys",
                page.page_no,
                exc_info=True,
            )
            try:
                plan = self._keyless_plan(page)
            except Exception:
                _log.warning(
                    "Form field extraction failed on page %d; page left unchanged",
                    page.page_no,
                    exc_info=True,
                )
                return
        self._commit(page, plan)

    def _plan_page(self, page: Page) -> _Plan:
        """Compute the page's field regions and edits without touching the page."""
        assert page.size is not None and page.parsed_page is not None
        assert page.predictions.layout is not None
        assignment = _assign(page)
        if assignment.solver_status != "optimal":
            _log.warning(
                "Form field keying abstained on page %d (%s); free-form values are "
                "kept without keys",
                page.page_no,
                assignment.solver_status,
            )
        values = [self._normalize_widget(v.native, v.bbox) for v in assignment.values]
        sources, unsure = _atom_sources(assignment)
        units = _build_units(assignment, values)
        in_cells, rest = _cell_units(units, assignment)
        hosts = _inline_hosts(rest, assignment, sources, page, unsure)
        regions = _place(rest, hosts, assignment, page)
        host_ids = {host.id for host in hosts.values()}
        keep = unsure | _table_children(page) | host_ids
        dropped, trimmed = _consumed_clusters(units, sources, assignment, page, keep)
        return _Plan(regions, dropped, trimmed, host_ids, values, in_cells)

    def _keyless_plan(self, page: Page) -> _Plan:
        """Every retained widget as an unkeyed value, placed by FORM region."""
        assert page.size is not None and page.parsed_page is not None
        values: list[FieldValuePrediction] = []
        for widget in page.parsed_page.widgets:
            bbox = widget.rect.to_bounding_box().to_top_left_origin(page.size.height)
            if not is_skipped(widget, bbox):
                values.append(self._normalize_widget(widget, bbox))
        items = [(value.bbox, FieldItemPrediction(values=[value])) for value in values]
        return _Plan(_form_regions(items, page), set(), {}, set(), values)

    def _commit(self, page: Page, plan: _Plan) -> None:
        """Write the plan to the page; restore the layout if that fails."""
        assert page.predictions.layout is not None
        layout = page.predictions.layout
        clusters = list(layout.clusters)
        children = {cluster.id: cluster.children for cluster in clusters}
        trimmed = [
            (cluster, cluster.cells, cluster.bbox)
            for cluster_id, cluster in _walk(clusters).items()
            if cluster_id in plan.trimmed
        ]
        try:
            page.predictions.field_regions = plan.regions
            page.predictions.table_fields = plan.table_fields
            self._drop_clusters(page, plan.dropped)
            self._trim_clusters(page, plan.trimmed)
            self._suppress_duplicate_text(page, plan.values, keep=frozenset(plan.hosts))
        except Exception:
            _log.warning(
                "Form field extraction failed on page %d; page left unchanged",
                page.page_no,
                exc_info=True,
            )
            page.predictions.field_regions = []
            page.predictions.table_fields = []
            for cluster in clusters:
                cluster.children = children[cluster.id]
            for cluster, cells, bbox in trimmed:
                cluster.cells = cells
                cluster.bbox = bbox
            layout.clusters = clusters

    @classmethod
    def _suppress_duplicate_text(
        cls,
        page: Page,
        values: list[FieldValuePrediction],
        *,
        keep: frozenset[int] = frozenset(),
    ) -> None:
        """Drop plain text clusters that merely re-render a native field value.

        A filled widget's appearance stream is painted into the page raster, so
        the layout model also detects it as an ordinary text cluster -- producing
        a duplicate of the widget's native ``/V``. Suppress such a cluster only
        when its text equals a field value's *and* it sits inside that widget's
        rect; the containment gate keeps ordinary printed text that coincidentally
        matches a value from being deleted. A paragraph that hosts a field
        item in place (``keep``) is the item's key, not a duplicate.

        A value the layout model glued onto a neighbouring label (a substring
        of a larger line, not an equal twin) is left in place: cutting it out
        of the line risks removing printed text.
        """
        assert page.predictions.layout is not None
        by_text: dict[str, list[BoundingBox]] = {}
        for value in values:
            text = cls._normalize_text(value.text)
            if text:
                by_text.setdefault(text, []).append(value.bbox)

        if not by_text:
            return

        def is_duplicate(cluster: Cluster) -> bool:
            return cluster.label in TEXT_ELEM_LABELS and any(
                cluster.bbox.intersection_over_self(widget_bbox)
                > cls._DUPLICATE_CONTAINMENT_THRESHOLD
                for widget_bbox in by_text.get(cls._cluster_text(cluster), [])
            )

        dropped_ids = {
            cluster.id
            for cluster in page.predictions.layout.clusters
            if cluster.id not in keep and is_duplicate(cluster)
        }
        cls._drop_clusters(page, dropped_ids)

    @staticmethod
    def _trim_clusters(page: Page, trimmed: dict[int, set[int]]) -> None:
        """Remove the given cells from their clusters and refit their boxes.

        The cells left in a cluster always include printable text (a cluster
        whose every printable cell became a key is dropped instead), and the
        box encloses them as the layout postprocessor fits text clusters.
        """
        if not trimmed:
            return
        assert page.predictions.layout is not None
        for cluster_id, cluster in _walk(page.predictions.layout.clusters).items():
            used = trimmed.get(cluster_id)
            if not used:
                continue
            cluster.cells = [cell for cell in cluster.cells if cell.index not in used]
            cluster.bbox = BoundingBox.enclosing_bbox(
                [cell.rect.to_bounding_box() for cell in cluster.cells]
            )

    @staticmethod
    def _drop_clusters(page: Page, dropped_ids: set[int]) -> None:
        """Remove clusters (and any container child refs to them) from the page.

        Shared by raster-duplicate suppression and key promotion so
        reading-order assembly never references a removed cluster.
        """
        if not dropped_ids:
            return
        assert page.predictions.layout is not None
        for cluster in page.predictions.layout.clusters:
            if cluster.children:
                cluster.children = [
                    child for child in cluster.children if child.id not in dropped_ids
                ]
        page.predictions.layout.clusters = [
            cluster
            for cluster in page.predictions.layout.clusters
            if cluster.id not in dropped_ids
        ]
