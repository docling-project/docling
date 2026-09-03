# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from collections.abc import Iterable

from docling_core.types.doc import BoundingBox, DocItemLabel
from docling_core.types.doc.page import PdfWidget

from docling.datamodel.base_models import (
    Cluster,
    FieldRegionPrediction,
    FieldValuePrediction,
    Page,
)
from docling.datamodel.document import ConversionResult
from docling.models.base_layout_model import TEXT_ELEM_LABELS
from docling.models.base_model import BasePageModel
from docling.utils.profiling import TimeRecorder


class PdfFormFieldModel(BasePageModel):
    _FORM_COVERAGE_THRESHOLD = 0.8
    _PUSHBUTTON_FLAG = 1 << 16
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
                assert page.size is not None
                assert page.predictions.layout is not None
                forms = [
                    cluster
                    for cluster in page.predictions.layout.clusters
                    if cluster.label == DocItemLabel.FORM
                ]
                matched_values: dict[int, list[FieldValuePrediction]] = {}
                matched_forms: dict[int, Cluster] = {}
                unmatched_values: list[FieldValuePrediction] = []

                for widget in page.parsed_page.widgets:
                    bbox = widget.rect.to_bounding_box().to_top_left_origin(
                        page.size.height
                    )
                    value = self._normalize_widget(widget, bbox)
                    form = self._match_form(bbox, forms)
                    if form is None:
                        unmatched_values.append(value)
                        continue
                    matched_forms[form.id] = form
                    matched_values.setdefault(form.id, []).append(value)

                regions = [
                    FieldRegionPrediction(
                        source_container_id=form_id,
                        bbox=matched_forms[form_id].bbox,
                        values=values,
                    )
                    for form_id, values in matched_values.items()
                ]
                if unmatched_values:
                    regions.append(
                        FieldRegionPrediction(
                            bbox=BoundingBox.enclosing_bbox(
                                [value.bbox for value in unmatched_values]
                            ),
                            values=unmatched_values,
                        )
                    )
                page.predictions.field_regions = regions

                all_values = [
                    value for values in matched_values.values() for value in values
                ] + unmatched_values
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
        if not dropped_ids:
            return

        # Prune both the top-level clusters and any container's child list, so
        # reading-order assembly does not reference a removed cluster.
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
