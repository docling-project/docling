# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import statistics
from collections.abc import Iterable

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
from docling.models.base_layout_model import TEXT_ELEM_LABELS
from docling.models.base_model import BasePageModel
from docling.utils.profiling import TimeRecorder


def _gap(w: BoundingBox, c: BoundingBox) -> float:
    """Rectangle edge-gap between a widget and a candidate label.

    Top-left origin: ``t`` is the upper edge (smaller y), ``b`` the lower. The gap
    is the direction logic for free -- a label directly left shares a horizontal
    band (``dy=0``, gap = horizontal spacing); a label above/below shares a
    vertical band (``dx=0``, gap = vertical spacing); a diagonal distractor has
    both nonzero and so scores worse. Zero when the rects overlap on both axes.
    """
    dx = max(0.0, w.l - c.r, c.l - w.r)
    dy = max(0.0, w.t - c.b, c.t - w.b)
    return dx + dy


def _precedes(c: BoundingBox, frontier: BoundingBox, row_band: float) -> bool:
    """True when ``c`` sits in a strictly higher row than ``frontier``.

    The no-crossing guard blocks a later widget from reaching back to a label in
    an earlier (higher) row -- vertical monotonicity only. It deliberately does
    *not* gate on left/right within a row: horizontal order is where multi-column
    forms interleave, and binding a right-side label (e.g. a trailing "RT")
    otherwise poisons the frontier so every left-side label in the next row is
    wrongly seen as preceding it. Per-label 1:1 consumption (``used``) handles
    the same-row case instead. ``row_band`` (a line-height multiple) is how much
    higher a center must be to count as a previous row.
    """
    cy, fy = (c.t + c.b) / 2.0, (frontier.t + frontier.b) / 2.0
    return cy < fy - row_band


def _match_labels(
    widgets: list[tuple[int, BoundingBox]],  # (widget.index, bbox), in index order
    labels: list[Cluster],  # unconsumed TEXT_ELEM_LABELS clusters
    cap: float,  # bind only if gap <= cap (text-scale bound, same units as bbox)
    row_band: float,  # same-row tolerance for the crossing guard
) -> dict[int, Cluster]:  # widget.index -> bound key cluster
    """Monotonic order-preserving binding of widgets to label clusters.

    One forward pass in ``widget.index`` order (proven effectively reading order).
    Each widget takes the nearest unconsumed label at or after the last binding in
    reading order, within ``cap``. Minimizing total gap subject to no-crossings
    *is* "minimize global ordering deviation"; a widget with no label in reach (a
    standalone tabular field) falls out as a skip. Binding is 1:1 -- a shared
    header binds one field and the rest stay keyless (a later-phase concern).
    """
    bound: dict[int, Cluster] = {}
    used: set[int] = set()
    frontier: BoundingBox | None = None  # last bound label -> no crossing past it
    for index, w in widgets:
        best: tuple[float, Cluster] | None = None
        for c in labels:
            if c.id in used or (
                frontier is not None and _precedes(c.bbox, frontier, row_band)
            ):
                continue
            g = _gap(w, c.bbox)
            if g <= cap and (best is None or g < best[0]):
                best = (g, c)
        if best is not None:
            bound[index], frontier = best[1], best[1].bbox
            used.add(best[1].id)
    return bound


class PdfFormFieldModel(BasePageModel):
    _FORM_COVERAGE_THRESHOLD = 0.8
    _PUSHBUTTON_FLAG = 1 << 16
    # A rendered-text cluster counts as a widget's duplicate only when this much
    # of it sits inside the widget rect. Guards against deleting ordinary printed
    # text that merely equals a field value by coincidence.
    _DUPLICATE_CONTAINMENT_THRESHOLD = 0.6
    # A layout CHECKBOX_* cluster is the visual twin of a /Btn widget when this
    # much of the (small) widget rect sits inside the cluster. The mark glyph the
    # layout model detects overlaps the widget square only partially and the
    # cluster also absorbs the neighbouring option label, so the gate is well
    # below full containment; on f1040s1_filled the real match measures ~0.70.
    # ponytail: overlap-only heuristic, single filled fixture -- widen the corpus
    # before tightening. Unselected boxes usually have no overlapping cluster and
    # fall through to the widget-only path (see docs handoff prereq B.5).
    _CHECKBOX_OVERLAP_THRESHOLD = 0.5
    # Bind a keyless widget to a nearby label only when their rectangle edge-gap
    # is within this many median line-heights -- a text-scale bound, not a page
    # fraction, so a field with no nearby label stays keyless. Row-band is the
    # same-row tolerance for the crossing guard. Both are the calibration knobs
    # the fixtures set; ponytail: 2.0 / 1.5 are the starting points, tune on the
    # 15-form corpus before trusting them.
    _LABEL_GAP_CAP_LINES = 2.0
    _LABEL_ROW_BAND_LINES = 1.5

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled

    @classmethod
    def _is_skipped(cls, widget: PdfWidget, bbox: BoundingBox) -> bool:
        """Widgets that carry no field value for the document.

        A widget of zero height or width is an artifact (Well-Tagged PDF 1.0,
        8.9.2.4.13). Push buttons trigger actions and hold no value.
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

    @staticmethod
    def _median_line_height(clusters: list[Cluster]) -> float:
        """Text scale for the label-gap cap: median label-cluster height.

        Most field labels are a single line, so cluster height is a good line-height
        proxy without digging into per-cell rects. Zero when there are no labels,
        which short-circuits the binding pass.
        """
        heights = [c.bbox.height for c in clusters if c.bbox.height > 0]
        return statistics.median(heights) if heights else 0.0

    @classmethod
    def _cluster_label_text(cls, cluster: Cluster) -> str:
        """Option label of a checkbox cluster (mark glyph included, if detected)."""
        return " ".join(
            cell.text.strip() for cell in cluster.cells if cell.text.strip()
        )

    @classmethod
    def _match_checkbox_cluster(
        cls, widget_bbox: BoundingBox, clusters: list[Cluster]
    ) -> Cluster | None:
        """Find the CHECKBOX_* cluster whose detected mark hosts this widget.

        Match on the widget rect sitting inside the cluster (``IoS(widget,
        cluster)``), not the reverse: the cluster is larger because it absorbs the
        neighbouring option label, so the widget is the subset.
        """
        best: tuple[float, Cluster] | None = None
        for cluster in clusters:
            ios = widget_bbox.intersection_over_self(cluster.bbox)
            if ios > cls._CHECKBOX_OVERLAP_THRESHOLD and (
                best is None or ios > best[0]
            ):
                best = (ios, cluster)
        return best[1] if best else None

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

    @classmethod
    def _match_text_container(
        cls, widget_bbox: BoundingBox, text_clusters: list[Cluster]
    ) -> Cluster | None:
        """Smallest text cluster that inlines this widget (the paragraph key).

        Precedence after FORM: a widget with no FORM host may sit inside a text
        paragraph (e.g. "check here [] and enter amount: 1221.00"). The smallest
        enclosing cluster is the guard against attaching to a big wrapping block
        -- it is the whole "is this widget really inlined in this paragraph"
        decision.
        """
        best: tuple[float, Cluster] | None = None
        for cluster in text_clusters:
            ios = widget_bbox.intersection_over_self(cluster.bbox)
            if ios >= cls._FORM_COVERAGE_THRESHOLD and (
                best is None or cluster.bbox.area() < best[1].bbox.area()
            ):
                best = (ios, cluster)
        return best[1] if best else None

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
                assert page.size is not None
                assert page.predictions.layout is not None
                forms = [
                    cluster
                    for cluster in page.predictions.layout.clusters
                    if cluster.label == DocItemLabel.FORM
                ]
                checkbox_clusters = [
                    cluster
                    for cluster in page.predictions.layout.clusters
                    if cluster.label
                    in {
                        DocItemLabel.CHECKBOX_SELECTED,
                        DocItemLabel.CHECKBOX_UNSELECTED,
                    }
                ]
                text_clusters = [
                    cluster
                    for cluster in page.predictions.layout.clusters
                    if cluster.label in TEXT_ELEM_LABELS
                ]
                matched_values: dict[int, list[FieldValuePrediction]] = {}
                matched_forms: dict[int, Cluster] = {}
                text_values: dict[int, list[FieldValuePrediction]] = {}
                text_containers: dict[int, Cluster] = {}
                unmatched_values: list[FieldValuePrediction] = []
                promoted_cluster_ids: set[int] = set()
                # (widget.index, value) for keyless widgets, in index order, fed to
                # the order-preserving label binding after the triage loop.
                keyless: list[tuple[int, FieldValuePrediction]] = []

                for widget in page.parsed_page.widgets:
                    bbox = widget.rect.to_bounding_box().to_top_left_origin(
                        page.size.height
                    )
                    if self._is_skipped(widget, bbox):
                        continue
                    value = self._normalize_widget(widget, bbox)
                    if value.checkbox is not None:
                        # Lift the visual checkbox's option label onto the value
                        # and take it out of the plain-text stream: state stays
                        # from /AS (already on value.checkbox), the label rides
                        # on the nested child. The classifier's own state guess
                        # is discarded -- /AS is authoritative.
                        cluster = self._match_checkbox_cluster(bbox, checkbox_clusters)
                        if cluster is not None:
                            value.checkbox_label = self._cluster_label_text(cluster)
                            # The layout cluster encloses both the widget square
                            # and its option label; take its bbox as the field
                            # item's prov so the box wraps the whole checkbox, not
                            # just the tiny widget rect.
                            value.bbox = cluster.bbox
                            promoted_cluster_ids.add(cluster.id)
                    # Precedence: smallest enclosing container wins. A FORM cluster
                    # usually wraps the whole page, so a widget inlined in a
                    # paragraph (IRS Sch.1 line 7: "check here [] and enter amount:
                    # 1221.00") sits inside both the FORM and a much smaller
                    # list_item -- the paragraph is the more specific host and
                    # becomes the item's key. Only a strictly-smaller text cluster
                    # beats the FORM; otherwise the widget stays a keyless FORM
                    # field, reproducing today's output.
                    form = self._match_form(bbox, forms)
                    text_cluster = self._match_text_container(bbox, text_clusters)
                    if text_cluster is not None and (
                        form is None or text_cluster.bbox.area() < form.bbox.area()
                    ):
                        text_containers[text_cluster.id] = text_cluster
                        text_values.setdefault(text_cluster.id, []).append(value)
                        # The paragraph cluster stays in the body: it materializes
                        # in place as a field_item (key = its text, values = these
                        # widgets), keeping its position in its list/container.
                        # page_assemble attaches the item onto the text element.
                        continue
                    if form is not None:
                        matched_forms[form.id] = form
                        matched_values.setdefault(form.id, []).append(value)
                    else:
                        unmatched_values.append(value)
                    # Value-only so far: a keyless FORM/unmatched widget whose label
                    # (if any) lives detached in the body. Queue it for the binding
                    # pass. A matched checkbox already carries its option label (a
                    # non-empty checkbox_label), so it is not keyless -- leave its
                    # validated behaviour untouched.
                    if not value.checkbox_label:
                        keyless.append((widget.index, value))

                # Order-preserving binding: pair each keyless widget with the
                # nearest unconsumed body label at or after the last binding in
                # reading order. text_containers are already keys, so they leave
                # the candidate pool. Bound labels become field-item keys and drop
                # from the body, reusing the overlapping-case promotion machinery.
                label_pool = [c for c in text_clusters if c.id not in text_containers]
                line_height = self._median_line_height(label_pool)
                bound_value_keys: dict[int, Cluster] = {}
                if keyless and label_pool and line_height > 0:
                    bound = _match_labels(
                        widgets=[(index, value.bbox) for index, value in keyless],
                        labels=label_pool,
                        cap=self._LABEL_GAP_CAP_LINES * line_height,
                        row_band=self._LABEL_ROW_BAND_LINES * line_height,
                    )
                    value_by_index = dict(keyless)
                    for index, cluster in bound.items():
                        # Identity map: values are unique, live objects for this
                        # page, so id() safely tags which item gets the key below.
                        bound_value_keys[id(value_by_index[index])] = cluster
                        promoted_cluster_ids.add(cluster.id)

                def _field_item(value: FieldValuePrediction) -> FieldItemPrediction:
                    cluster = bound_value_keys.get(id(value))
                    if cluster is None:
                        return FieldItemPrediction(values=[value])
                    return FieldItemPrediction(
                        key_text=self._cluster_label_text(cluster),
                        key_bbox=cluster.bbox,
                        values=[value],
                    )

                regions = [
                    FieldRegionPrediction(
                        source_container_id=form_id,
                        bbox=matched_forms[form_id].bbox,
                        items=[_field_item(value) for value in values],
                    )
                    for form_id, values in matched_values.items()
                ]
                # Widgets sharing one enclosing paragraph accumulate into a single
                # keyed item; the key text/bbox is the paragraph cluster.
                regions.extend(
                    FieldRegionPrediction(
                        source_container_id=cluster_id,
                        bbox=text_containers[cluster_id].bbox,
                        items=[
                            FieldItemPrediction(
                                key_text=self._cluster_label_text(
                                    text_containers[cluster_id]
                                ),
                                key_bbox=text_containers[cluster_id].bbox,
                                values=values,
                            )
                        ],
                    )
                    for cluster_id, values in text_values.items()
                )
                if unmatched_values:
                    regions.append(
                        FieldRegionPrediction(
                            bbox=BoundingBox.enclosing_bbox(
                                [value.bbox for value in unmatched_values]
                            ),
                            items=[_field_item(value) for value in unmatched_values],
                        )
                    )
                page.predictions.field_regions = regions

                all_values = (
                    [value for values in matched_values.values() for value in values]
                    + [value for values in text_values.values() for value in values]
                    + unmatched_values
                )
                # Promote matched checkbox clusters (now hosted inside a field
                # item) out of the body before suppressing raster duplicates.
                self._drop_clusters(page, promoted_cluster_ids)
                self._suppress_duplicate_text(page, all_values)

            yield page

    @classmethod
    def _suppress_duplicate_text(
        cls, page: Page, values: list[FieldValuePrediction]
    ) -> None:
        """Drop plain text clusters that merely re-render a native field value.

        A filled widget's appearance stream is painted into the page raster, so
        the layout model also detects it as an ordinary text cluster -- producing
        a duplicate of the widget's native ``/V``. Suppress such a cluster only
        when its text equals a field value's *and* it sits inside that widget's
        rect; the containment gate keeps ordinary printed text that coincidentally
        matches a value from being deleted.

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
            if is_duplicate(cluster)
        }
        cls._drop_clusters(page, dropped_ids)

    @staticmethod
    def _drop_clusters(page: Page, dropped_ids: set[int]) -> None:
        """Remove clusters (and any container child refs to them) from the page.

        Shared by raster-duplicate suppression and checkbox promotion so
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
