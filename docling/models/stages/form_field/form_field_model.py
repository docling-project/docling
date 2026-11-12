# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

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

_CHECKBOX_LABELS = {DocItemLabel.CHECKBOX_SELECTED, DocItemLabel.CHECKBOX_UNSELECTED}


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

    @staticmethod
    def _tooltip_keyed_item(
        widget: PdfWidget, value: FieldValuePrediction
    ) -> FieldItemPrediction:
        """Single-value item keyed by the field's ``/TU`` description, if any.

        ``/TU`` is the field's accessible description and one of the context
        sources listed in Well-Tagged PDF 1.0, 8.10.2.1; the field name ``/T``
        is explicitly not one and never becomes document text. The key is
        dictionary metadata, not page text, so it carries no bbox.
        """
        key = (widget.widget_description or "").strip()
        return FieldItemPrediction(
            key_text=key, key_source="tooltip" if key else None, values=[value]
        )

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
                    if cluster.label in _CHECKBOX_LABELS
                ]
                # A CHECKBOX_* cluster is a control's own option label, not a
                # paragraph that hosts controls: it is lifted onto the checkbox
                # child above, never used as a key container. Admitting it would
                # key the widget by its own label (duplicating it) and point the
                # region at a cluster that promotion drops before assembly.
                text_clusters = [
                    cluster
                    for cluster in page.predictions.layout.clusters
                    if cluster.label in TEXT_ELEM_LABELS
                    and cluster.label not in _CHECKBOX_LABELS
                ]
                matched_items: dict[int, list[FieldItemPrediction]] = {}
                matched_forms: dict[int, Cluster] = {}
                text_values: dict[int, list[FieldValuePrediction]] = {}
                text_containers: dict[int, Cluster] = {}
                unmatched_items: list[FieldItemPrediction] = []
                promoted_cluster_ids: set[int] = set()

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
                    # Not inlined in a paragraph: the item stands alone, keyed by
                    # the field's /TU description when the PDF carries one.
                    item = self._tooltip_keyed_item(widget, value)
                    if form is not None:
                        matched_forms[form.id] = form
                        matched_items.setdefault(form.id, []).append(item)
                        continue
                    unmatched_items.append(item)

                regions = [
                    FieldRegionPrediction(
                        source_container_id=form_id,
                        bbox=matched_forms[form_id].bbox,
                        items=items,
                    )
                    for form_id, items in matched_items.items()
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
                                key_source="layout",
                                values=values,
                            )
                        ],
                    )
                    for cluster_id, values in text_values.items()
                )
                if unmatched_items:
                    regions.append(
                        FieldRegionPrediction(
                            bbox=BoundingBox.enclosing_bbox(
                                [
                                    value.bbox
                                    for item in unmatched_items
                                    for value in item.values
                                ]
                            ),
                            items=unmatched_items,
                        )
                    )
                page.predictions.field_regions = regions

                all_values = [
                    value
                    for region in regions
                    for item in region.items
                    for value in item.values
                ]
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
