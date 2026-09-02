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
from docling.models.base_model import BasePageModel
from docling.utils.profiling import TimeRecorder


class PdfFormFieldModel(BasePageModel):
    _FORM_COVERAGE_THRESHOLD = 0.8
    _PUSHBUTTON_FLAG = 1 << 16

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled

    @classmethod
    def _normalize_widget(
        cls, widget: PdfWidget, bbox: BoundingBox
    ) -> FieldValuePrediction:
        source_value = widget.widget_text or ""
        if (
            widget.widget_field_type == "/Btn"
            and not widget.widget_field_flags & cls._PUSHBUTTON_FLAG
        ):
            source_value = widget.widget_appearance_state or source_value
            text = "unchecked" if source_value in {"", "/Off", "Off"} else "checked"
        else:
            text = source_value

        return FieldValuePrediction(text=text, orig=source_value, bbox=bbox)

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

            yield page
