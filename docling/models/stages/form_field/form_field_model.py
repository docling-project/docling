# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Native AcroForm widgets keyed to their printed captions.

The stage runs after table structure. For every page with widgets it builds
the keying snapshot (widgets, layout regions, detected table cells), lets
``keying.assign`` choose the caption of each value, and turns the result into
``FieldRegionPrediction``s: one item per association, with the caption as key
and the column header of a grid as context. Items are placed as the original
stage placed its values: a paragraph that inlines all of an item's widgets
hosts the item in place, otherwise the enclosing FORM region does, otherwise a
page-wide region. Caption text that became a key leaves the body, and text
blocks that merely re-render a filled value are dropped. A page whose keying
fails keeps its values, without keys.
"""

import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from docling_core.types.doc import BoundingBox, DocItemLabel
from docling_core.types.doc.page import PdfWidget

from docling.datamodel.base_models import (
    Cluster,
    FieldItemPrediction,
    FieldRegionPrediction,
    FieldValuePrediction,
    Page,
)
from docling.datamodel.document import ConversionResult
from docling.models.base_layout_model import PAGE_HEADER_LABELS, TEXT_ELEM_LABELS
from docling.models.base_model import BasePageModel
from docling.models.stages.form_field.keying import (
    Assignment,
    DetectedTable,
    Label,
    NativeWidget,
    Region,
    Snapshot,
    Tables,
    assign,
)
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)

# Layout regions whose cells never become labels (same list as keying.inputs).
NOT_LABELS = {
    DocItemLabel.FORM,
    DocItemLabel.KEY_VALUE_REGION,
    DocItemLabel.TABLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.PICTURE,
}
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
    dropped: set[int]  # clusters whose text became keys
    hosts: set[int]  # paragraphs that carry an item in place; never dropped
    values: list[FieldValuePrediction]


def _walk(clusters: list[Cluster]) -> dict[int, Cluster]:
    """Every cluster by id, children included."""
    found: dict[int, Cluster] = {}
    pending = list(clusters)
    while pending:
        cluster = pending.pop()
        if cluster.id not in found:
            found[cluster.id] = cluster
            pending.extend(cluster.children)
    return found


def _printable(cluster: Cluster) -> set[int]:
    return {cell.index for cell in cluster.cells if cell.text.strip()}


def _rounded(box: BoundingBox) -> tuple[float, ...]:
    return tuple(round(x, 3) for x in box.as_tuple())


def to_snapshot(page: Page) -> Snapshot:
    """The keying input of a live page, with the meaning of a frozen snapshot."""
    assert page.size is not None and page.parsed_page is not None
    assert page.predictions.layout is not None
    structure = page.predictions.tablestructure
    return Snapshot(
        page=page.page_no,
        size=page.size,
        widgets=[
            NativeWidget.model_validate(w.model_dump())
            for w in page.parsed_page.widgets
        ],
        # Copies: committing the plan later edits the live clusters.
        layout=[
            Region.model_validate(c.model_dump(mode="json"))
            for c in page.predictions.layout.clusters
        ],
        tables=Tables(
            table_map={}
            if structure is None
            else {
                table_id: DetectedTable(table_cells=table.table_cells)
                for table_id, table in structure.table_map.items()
            }
        ),
    )


def _atom_sources(
    assignment: Assignment, page: Page
) -> tuple[dict[int, tuple[int, int]], set[int]]:
    """Atom -> (cluster id, cell index), plus clusters that must stay in the body.

    Labels keep no cluster id, but keying.inputs emits a one-cell label with
    the cell's own box and text for every cell it keeps. A cell the layout put
    in two clusters leaves both clusters in place ("unsure").
    """
    assert page.size is not None and page.predictions.layout is not None
    where: dict[tuple[tuple[float, ...], str], set[tuple[int, int]]] = defaultdict(set)
    for cluster in _walk(page.predictions.layout.clusters).values():
        if cluster.label in NOT_LABELS:
            continue
        for cell in cluster.cells:
            if text := cell.text.strip():
                box = cell.rect.to_bounding_box().to_top_left_origin(page.size.height)
                where[_rounded(box), text].add((cluster.id, cell.index))
    sources: dict[int, tuple[int, int]] = {}
    unsure: set[int] = set()
    for label in assignment.labels:
        if len(label.atoms) != 1:
            continue
        (atom,) = label.atoms
        found = where[_rounded(label.bbox), label.text]
        if not found:
            raise ValueError(
                f"Page {page.page_no}: label {label.text!r} has no layout cell"
            )
        sources[atom] = min(found)
        if len({cluster_id for cluster_id, _ in found}) > 1:
            unsure |= {cluster_id for cluster_id, _ in found}
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
                >= PdfFormFieldModel._FORM_COVERAGE_THRESHOLD
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
    assert page.predictions.layout is not None
    forms = [
        c for c in page.predictions.layout.clusters if c.label == DocItemLabel.FORM
    ]
    regions: list[FieldRegionPrediction] = []
    by_form: dict[int, list[FieldItemPrediction]] = defaultdict(list)
    loose: list[FieldItemPrediction] = []
    for k, unit in enumerate(units):
        host = hosts.get(k)
        if host is not None:
            regions.append(
                FieldRegionPrediction(
                    source_container_id=host.id, bbox=host.bbox, items=[unit.item]
                )
            )
            continue
        first = assignment.values[min(unit.positions)].bbox
        form = PdfFormFieldModel._match_form(first, forms)
        (loose if form is None else by_form[form.id]).append(unit.item)
    boxes = {form.id: form.bbox for form in forms}
    regions += [
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
    page: Page,
    keep: set[int],
) -> set[int]:
    """Clusters whose every printable cell became key text.

    A cluster only partly used stays in the body, so its text shows twice.
    """
    assert page.predictions.layout is not None
    clusters = _walk(page.predictions.layout.clusters)
    used: dict[int, set[int]] = defaultdict(set)
    for unit in units:
        if unit.consumed is not None:
            for atom in unit.consumed.atoms:
                cluster_id, cell_index = sources[atom]
                used[cluster_id].add(cell_index)
    return {
        cluster_id
        for cluster_id, cells in used.items()
        if cluster_id not in keep and cells >= _printable(clusters[cluster_id])
    }


class PdfFormFieldModel(BasePageModel):
    _FORM_COVERAGE_THRESHOLD = 0.8
    _PUSHBUTTON_FLAG = 1 << 16
    # A rendered-text cluster counts as a widget's duplicate only when this much
    # of it sits inside the widget rect. Guards against deleting ordinary printed
    # text that merely equals a field value by coincidence.
    _DUPLICATE_CONTAINMENT_THRESHOLD = 0.6

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        if enabled:
            # The keying's integer program needs scipy.optimize.milp (SciPy
            # 1.9+). Fail at construction rather than page by page.
            try:
                from scipy.optimize import milp
            except ImportError as error:
                raise ImportError(
                    "extract_form_fields requires SciPy 1.9 or later "
                    "(scipy.optimize.milp)"
                ) from error

    @classmethod
    def _is_skipped(cls, widget: PdfWidget, bbox: BoundingBox) -> bool:
        """Widgets that carry no field value for the document.

        A widget of zero height or width is an artifact (Well-Tagged PDF 1.0,
        8.9.2.4.13). Push buttons trigger actions and hold no value. The same
        rule lives in keying.inputs, so both sides see the same values.
        """
        if bbox.width <= 0 or bbox.height <= 0:
            return True
        return widget.widget_field_type == "/Btn" and bool(
            widget.widget_field_flags & cls._PUSHBUTTON_FLAG
        )

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
            and not widget.widget_field_flags & cls._PUSHBUTTON_FLAG
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
            if widget_bbox.intersection_over_self(form.bbox)
            > cls._FORM_COVERAGE_THRESHOLD
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
        assignment = assign(to_snapshot(page))
        if assignment.solver_status != "optimal":
            _log.warning(
                "Form field keying abstained on page %d (%s); free-form values are "
                "kept without keys",
                page.page_no,
                assignment.solver_status,
            )
        widgets = {w.index: w for w in page.parsed_page.widgets}
        values = [
            self._normalize_widget(widgets[v.native.index], v.bbox)
            for v in assignment.values
        ]
        sources, unsure = _atom_sources(assignment, page)
        units = _build_units(assignment, values)
        hosts = _inline_hosts(units, assignment, sources, page, unsure)
        regions = _place(units, hosts, assignment, page)
        host_ids = {host.id for host in hosts.values()}
        keep = unsure | _table_children(page) | host_ids
        dropped = _consumed_clusters(units, sources, page, keep)
        return _Plan(regions, dropped, host_ids, values)

    def _keyless_plan(self, page: Page) -> _Plan:
        """Every retained widget as an unkeyed value, placed by FORM region."""
        assert page.size is not None and page.parsed_page is not None
        assert page.predictions.layout is not None
        forms = [
            c for c in page.predictions.layout.clusters if c.label == DocItemLabel.FORM
        ]
        by_form: dict[int, list[FieldItemPrediction]] = defaultdict(list)
        loose: list[FieldItemPrediction] = []
        values: list[FieldValuePrediction] = []
        for widget in page.parsed_page.widgets:
            bbox = widget.rect.to_bounding_box().to_top_left_origin(page.size.height)
            if self._is_skipped(widget, bbox):
                continue
            value = self._normalize_widget(widget, bbox)
            values.append(value)
            form = self._match_form(bbox, forms)
            item = FieldItemPrediction(values=[value])
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
        return _Plan(regions, set(), set(), values)

    def _commit(self, page: Page, plan: _Plan) -> None:
        """Write the plan to the page; restore the layout if that fails."""
        assert page.predictions.layout is not None
        layout = page.predictions.layout
        clusters = list(layout.clusters)
        children = {cluster.id: cluster.children for cluster in clusters}
        try:
            page.predictions.field_regions = plan.regions
            self._drop_clusters(page, plan.dropped)
            self._suppress_duplicate_text(page, plan.values, keep=plan.hosts)
        except Exception:
            _log.warning(
                "Form field extraction failed on page %d; page left unchanged",
                page.page_no,
                exc_info=True,
            )
            page.predictions.field_regions = []
            for cluster in clusters:
                cluster.children = children[cluster.id]
            layout.clusters = clusters

    @classmethod
    def _suppress_duplicate_text(
        cls,
        page: Page,
        values: list[FieldValuePrediction],
        *,
        keep: set[int] = frozenset(),  # type: ignore[assignment]
    ) -> None:
        """Drop plain text clusters that merely re-render a native field value.

        A filled widget's appearance stream is painted into the page raster, so
        the layout model also detects it as an ordinary text cluster -- producing
        a duplicate of the widget's native ``/V``. Suppress such a cluster only
        when its text equals a field value's *and* it sits inside that widget's
        rect; the containment gate keeps ordinary printed text that coincidentally
        matches a value from being deleted. A paragraph that hosts a field
        item in place (``keep``) is the item's key, not a duplicate.

        ponytail: leaves the ~10% of values the layout model glues onto a
        neighbouring label (value is a substring of a larger line, not an equal
        twin); excising a suffix mid-string is the risky over-editing we avoid.
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
