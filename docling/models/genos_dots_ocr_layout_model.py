import io
import base64
import json
import math
import requests
import copy
import logging
import re
import time
import warnings
from collections.abc import Iterable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from bs4 import BeautifulSoup, Tag
from docling_core.types.doc import DocItemLabel, TableData
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image, ImageDraw, ImageFont

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import (
    BoundingBox,
    Cluster,
    EquationPrediction,
    LayoutPrediction,
    Page,
    Table,
    TableStructurePrediction,
    TextElement,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_V2, LayoutModelConfig
from docling.datamodel.pipeline_options import (
    LayoutOptions,
    PdfPipelineOptions,
    TableStructureModelType,
)
from docling.datamodel.settings import settings
from docling.models.base_model import BasePageModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device

from docling.utils.genos_dotsocr_postprocessor import LayoutPostprocessor
from docling.utils.profiling import ProfilingItem, ProfilingScope, TimeRecorder
from docling.utils.visualization import draw_clusters

_log = logging.getLogger(__name__)


def _record_stage_elapsed(conv_res, key: str, elapsed: float) -> None:
    """Append a sub-stage duration into ``conv_res.timings[key]``.

    Mirrors :class:`TimeRecorder` but works without re-indenting large code
    blocks and is safe to call from the per-page worker threads (``list.append``
    is atomic under the GIL; keys are pre-created in ``__call__``). No-op unless
    ``settings.debug.profile_pipeline_timings`` is enabled.

    Also records an approximate start timestamp (``now - elapsed``; this is
    called right after the work finishes) so the profiling summary can compute
    a document-level wall-clock by union-ing the per-page intervals. The two
    ``list.append`` calls are individually atomic; concurrent worker threads may
    interleave them, but intervals within one batch share near-identical start
    and duration, so the union estimate is unaffected in practice. Wrap both in
    a ``Lock`` if exact (start, duration) pairing is ever required.
    """
    if not settings.debug.profile_pipeline_timings:
        return
    item = conv_res.timings.setdefault(key, ProfilingItem(scope=ProfilingScope.PAGE))
    item.start_timestamps.append(datetime.utcnow() - timedelta(seconds=elapsed))
    item.times.append(elapsed)
    item.count += 1


DOTSOCR_IMAGE_FACTOR = 28
DOTSOCR_MIN_PIXELS = 3136
DOTSOCR_MAX_PIXELS = 11289600


class GenosDotsOCRLayoutModel(BasePageModel):
    GLYPH_RE = re.compile(r"GLYPH\w*")

    TEXT_ELEM_LABELS = [
        DocItemLabel.TEXT,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.CAPTION,
        DocItemLabel.CHECKBOX_UNSELECTED,
        DocItemLabel.CHECKBOX_SELECTED,
        DocItemLabel.SECTION_HEADER,
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.CODE,
        DocItemLabel.LIST_ITEM,
        DocItemLabel.FORMULA,
    ]
    PAGE_HEADER_LABELS = [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER]

    TABLE_LABELS = [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]
    FIGURE_LABEL = DocItemLabel.PICTURE
    FORMULA_LABEL = DocItemLabel.FORMULA
    CONTAINER_LABELS = [DocItemLabel.FORM, DocItemLabel.KEY_VALUE_REGION]

    def __init__(self, pipeline_options: PdfPipelineOptions) -> None:
        self.pipeline_options = pipeline_options
        self.options = pipeline_options.layout_options
        self.dotsocr_options = getattr(self.options, "dotsocr_options", None)
        if self.dotsocr_options is None:
            self.dotsocr_options = self.options.genos_layout_options
        self.dotocr_endpoint = self.dotsocr_options.endpoint
        self.api_key = self.dotsocr_options.api_key
        self.model = getattr(self.dotsocr_options, "model", "dots-mocr")
        self.max_completion_tokens = self.dotsocr_options.max_completion_tokens
        self.timeout = self.dotsocr_options.timeout
        retry_count = getattr(self.dotsocr_options, "retry_count", 2)
        try:
            retry_count = int(retry_count)
        except (TypeError, ValueError):
            _log.warning(
                "Invalid genos_layout_options.retry_count=%r. Falling back to 2.",
                retry_count,
            )
            retry_count = 2
        self.retry_count = max(0, retry_count)
        self.temperature = getattr(self.dotsocr_options, "temperature", 0.1)
        self.top_p = getattr(self.dotsocr_options, "top_p", 0.9)
        self.repetition_penalty = getattr(
            self.dotsocr_options, "repetition_penalty", 1.15
        )

    def _use_dotsocr_table_structure(self) -> bool:
        return (
            self.pipeline_options.do_table_structure
            and self.pipeline_options.table_structure_options.table_structure_model_type
            == TableStructureModelType.DOTSOCR
        )

    @classmethod
    def _check_glyph_text(cls, text: str, threshold: int = 1) -> bool:
        # Keep parity with intelligent_processor.check_glyph_text
        if not text:
            return False
        return len(cls.GLYPH_RE.findall(text)) >= threshold

    @classmethod
    def _check_glyphs(cls, texts: Iterable[str]) -> bool:
        # Keep parity with intelligent_processor.check_glyphs (>10 glyph markers)
        for text in texts:
            if not text:
                continue
            if len(cls.GLYPH_RE.findall(text)) > 10:
                return True
        return False

    @staticmethod
    def _collect_cluster_text(cluster: Cluster) -> str:
        return " ".join(
            (cell.text or "").strip()
            for cell in cluster.cells
            if (cell.text or "").strip()
        )

    @staticmethod
    def _overwrite_cells_text(cells: list[TextCell], replacement_text: str) -> bool:
        if not cells:
            return False

        target_idx = 0
        for idx, cell in enumerate(cells):
            if (cell.text or "").strip():
                target_idx = idx
                break

        updated = False
        for idx, cell in enumerate(cells):
            new_text = replacement_text if idx == target_idx else ""
            if cell.text != new_text:
                cell.text = new_text
                cell.orig = new_text
                updated = True

        return updated

    @staticmethod
    def _assign_cells_to_best_cluster(
        page_cells: list[TextCell],
        clusters: list[Cluster],
        min_overlap: float = 0.2,
    ) -> dict[int, list[TextCell]]:
        assigned_cells_by_cluster_id: dict[int, list[TextCell]] = {}

        for cell in page_cells:
            cell_bbox = cell.rect.to_bounding_box()
            if cell_bbox.area() <= 0:
                continue

            best_overlap = min_overlap
            best_cluster_id: Optional[int] = None
            for cluster in clusters:
                overlap_ratio = cell_bbox.intersection_over_self(cluster.bbox)
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_cluster_id = cluster.id

            if best_cluster_id is None:
                continue

            assigned_cells_by_cluster_id.setdefault(best_cluster_id, []).append(cell)

        return assigned_cells_by_cluster_id

    def _replace_glyph_text_with_dotsocr(
        self,
        page: Page,
        clusters: list[Cluster],
        dotsocr_text_by_cluster_id: dict[int, str],
    ) -> int:
        text_clusters = [
            cluster
            for cluster in clusters
            if cluster.label in self.TEXT_ELEM_LABELS and cluster.label != self.FORMULA_LABEL
        ]
        special_labels = set(self.TABLE_LABELS)
        special_labels.add(self.FIGURE_LABEL)
        special_labels.update(self.CONTAINER_LABELS)
        regular_clusters = [
            cluster for cluster in clusters if cluster.label not in special_labels
        ]
        assigned_cells_by_cluster_id = self._assign_cells_to_best_cluster(
            page_cells=page.cells,
            clusters=regular_clusters,
        )
        source_text_by_cluster_id = {
            cluster.id: " ".join(
                (cell.text or "").strip()
                for cell in assigned_cells_by_cluster_id.get(cluster.id, [])
                if (cell.text or "").strip()
            )
            for cluster in text_clusters
        }

        # Follow intelligent_processor.check_glyphs-style gating for glyph-based replacement.
        # Missing source text replacement is always allowed.
        has_glyph_issue = self._check_glyphs(source_text_by_cluster_id.values())

        replaced_clusters = 0
        replaced_missing_clusters = 0
        replaced_glyph_clusters = 0
        created_synthetic_cells = 0
        next_cell_index = max((cell.index for cell in page.cells), default=-1) + 1

        for cluster in text_clusters:
            source_text = source_text_by_cluster_id.get(cluster.id, "")
            source_text_missing = not source_text.strip()
            source_text_has_glyph = has_glyph_issue and self._check_glyph_text(
                source_text, threshold=1
            )
            if not source_text_missing and not source_text_has_glyph:
                continue

            dotsocr_text = (dotsocr_text_by_cluster_id.get(cluster.id) or "").strip()
            if not dotsocr_text:
                continue
            if self._check_glyph_text(dotsocr_text, threshold=1):
                continue

            target_cells = assigned_cells_by_cluster_id.get(cluster.id, [])
            if target_cells:
                updated = self._overwrite_cells_text(target_cells, dotsocr_text)
            else:
                synthetic_cell = TextCell(
                    index=next_cell_index,
                    text=dotsocr_text,
                    orig=dotsocr_text,
                    confidence=cluster.confidence,
                    from_ocr=True,
                    rect=BoundingRectangle.from_bounding_box(cluster.bbox),
                )
                page.cells.append(synthetic_cell)
                assigned_cells_by_cluster_id[cluster.id] = [synthetic_cell]
                next_cell_index += 1
                created_synthetic_cells += 1
                updated = True

            if updated:
                replaced_clusters += 1
                if source_text_missing:
                    replaced_missing_clusters += 1
                elif source_text_has_glyph:
                    replaced_glyph_clusters += 1

        if replaced_clusters:
            _log.info(
                "Replaced parser text with DotsOCR text "
                "(page=%s, clusters=%s, missing=%s, glyph=%s, synthetic_cells=%s).",
                page.page_no,
                replaced_clusters,
                replaced_missing_clusters,
                replaced_glyph_clusters,
                created_synthetic_cells,
            )
        return replaced_clusters

    def _build_tablestructure_from_dotsocr(
        self,
        page: Page,
        clusters: list[Cluster],
        table_html_by_cluster_id: dict[int, str],
    ) -> TableStructurePrediction:
        tablestructure = TableStructurePrediction()
        for cluster in clusters:
            if cluster.label not in self.TABLE_LABELS:
                continue

            table_html = table_html_by_cluster_id.get(cluster.id)
            if not table_html:
                _log.warning(
                    "DotsOCR table cluster has no HTML text (page=%s, cluster_id=%s).",
                    page.page_no,
                    cluster.id,
                )
                continue

            table_data = _parse_html_to_table_data(table_html)
            if table_data is None:
                _log.warning(
                    "Failed to parse DotsOCR table HTML (page=%s, cluster_id=%s).",
                    page.page_no,
                    cluster.id,
                )
                continue

            tablestructure.table_map[cluster.id] = Table(
                otsl_seq=[],
                table_cells=table_data.table_cells,
                num_rows=table_data.num_rows,
                num_cols=table_data.num_cols,
                text=table_html,
                id=cluster.id,
                page_no=page.page_no,
                cluster=cluster,
                label=cluster.label,
            )

        return tablestructure

    def draw_clusters_and_cells_side_by_side(
        self,
        conv_res,
        page,
        clusters,
        mode_prefix: str,
        show: bool = False,
        draw_side_by_side: Optional[bool] = None,
        cluster_order_map: Optional[dict[int, int]] = None,
    ):
        """
        Draws layout clusters for debug visualization.
        - If side-by-side mode is enabled, clusters are split into two panes:
          left excludes FORM/KEY_VALUE_REGION/PICTURE and right contains them.
        - Otherwise all clusters are drawn on a single image.
        Includes label names, confidence scores, and order index for each cluster.
        """
        scale_x = page.image.width / page.size.width
        scale_y = page.image.height / page.size.height

        # Global order map:
        # - If provided, preserve source(DotsOCR raw) cluster order.
        # - Otherwise preserve incoming cluster list order.
        if cluster_order_map is None:
            order_map = {cluster.id: idx + 1 for idx, cluster in enumerate(clusters)}
        else:
            order_map = dict(cluster_order_map)
            next_order = max(order_map.values(), default=0) + 1
            # Keep deterministic numbering for newly created clusters (e.g. orphans).
            for cluster in sorted(clusters, key=lambda c: c.id):
                if cluster.id not in order_map:
                    order_map[cluster.id] = next_order
                    next_order += 1

        side_by_side_from_options = getattr(
            self.options, "visualize_layout_side_by_side", False
        )
        render_side_by_side = (
            bool(side_by_side_from_options)
            if draw_side_by_side is None
            else bool(draw_side_by_side)
        )

        if render_side_by_side:
            # Filter clusters for left and right images
            exclude_labels = {
                DocItemLabel.FORM,
                DocItemLabel.KEY_VALUE_REGION,
                DocItemLabel.PICTURE,
            }
            left_clusters = [c for c in clusters if c.label not in exclude_labels]
            right_clusters = [c for c in clusters if c.label in exclude_labels]

            # Create a deep copy of the original image for both sides
            left_image = copy.deepcopy(page.image)
            right_image = copy.deepcopy(page.image)

            draw_clusters(left_image, left_clusters, scale_x, scale_y)
            draw_clusters(right_image, right_clusters, scale_x, scale_y)
            self._draw_cluster_order_indices(
                left_image, left_clusters, order_map, scale_x, scale_y
            )
            self._draw_cluster_order_indices(
                right_image, right_clusters, order_map, scale_x, scale_y
            )

            # Combine the images side by side
            combined_width = left_image.width * 2
            combined_height = left_image.height
            output_image = Image.new("RGB", (combined_width, combined_height))
            output_image.paste(left_image, (0, 0))
            output_image.paste(right_image, (left_image.width, 0))
        else:
            output_image = copy.deepcopy(page.image)
            draw_clusters(output_image, clusters, scale_x, scale_y)
            self._draw_cluster_order_indices(
                output_image, clusters, order_map, scale_x, scale_y
            )

        if show:
            output_image.show()
        else:
            out_path: Path = (
                Path(settings.debug.debug_output_path)
                / f"debug_{conv_res.input.file.stem}"
            )
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / f"{mode_prefix}_layout_page_{page.page_no:05}.png"
            output_image.save(str(out_file), format="png")

    def _draw_cluster_order_indices(
        self,
        image: Image.Image,
        clusters: list[Cluster],
        order_map: dict[int, int],
        scale_x: float,
        scale_y: float,
    ) -> None:
        draw = ImageDraw.Draw(image, "RGBA")
        try:
            order_font = ImageFont.truetype("arial.ttf", 24)
        except OSError:
            try:
                order_font = ImageFont.truetype("DejaVuSans.ttf", 24)
            except OSError:
                order_font = ImageFont.load_default()
        try:
            # Match draw_clusters label font size to estimate occupied label regions.
            label_font = ImageFont.truetype("arial.ttf", 12)
        except OSError:
            label_font = ImageFont.load_default()

        def _as_valid_rect(x0, y0, x1, y1):
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            return (float(x0), float(y0), float(x1), float(y1))

        def _clamp_box(x0, y0, box_w, box_h):
            x0 = float(min(max(x0, 0.0), max(float(image.width) - box_w, 0.0)))
            y0 = float(min(max(y0, 0.0), max(float(image.height) - box_h, 0.0)))
            return (x0, y0, x0 + box_w, y0 + box_h)

        def _boxes_intersect(a, b):
            return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

        def _overlap_area(a, b):
            inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            return inter_w * inter_h

        def _create_text_mask(text: str, font: ImageFont.ImageFont, min_height: int):
            probe = Image.new("L", (1, 1), 0)
            probe_draw = ImageDraw.Draw(probe)
            bbox = probe_draw.textbbox((0, 0), text, font=font)
            text_w = max(1, int(bbox[2] - bbox[0]))
            text_h = max(1, int(bbox[3] - bbox[1]))

            mask = Image.new("L", (text_w, text_h), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.text((-bbox[0], -bbox[1]), text, font=font, fill=255)

            if text_h < min_height:
                scale = float(min_height) / float(text_h)
                new_w = max(1, int(round(text_w * scale)))
                new_h = max(1, int(round(text_h * scale)))
                resampling = (
                    Image.Resampling.NEAREST
                    if hasattr(Image, "Resampling")
                    else Image.NEAREST
                )
                mask = mask.resize((new_w, new_h), resample=resampling)

            return mask

        # Pre-compute existing label boxes (already drawn by draw_clusters) as forbidden regions.
        label_boxes: list[tuple[float, float, float, float]] = []
        label_boxes_by_cluster_id: dict[int, tuple[float, float, float, float]] = {}
        label_bg_padding = 2
        for cluster in clusters:
            x0, y0, x1, y1 = cluster.bbox.as_tuple()
            x0 *= scale_x
            x1 *= scale_x
            y0 *= scale_x  # Keep consistent with draw_clusters current rendering.
            y1 *= scale_y
            x0, y0, x1, y1 = _as_valid_rect(x0, y0, x1, y1)

            label_text = f"{cluster.label.name} ({cluster.confidence:.2f})"
            text_bbox = draw.textbbox((x0, y0), label_text, font=label_font)
            label_box = (
                float(text_bbox[0] - label_bg_padding),
                float(text_bbox[1] - label_bg_padding),
                float(text_bbox[2] + label_bg_padding),
                float(text_bbox[3] + label_bg_padding),
            )
            label_boxes.append(label_box)
            label_boxes_by_cluster_id[id(cluster)] = label_box

        placed_badges: list[tuple[float, float, float, float]] = []

        for cluster in clusters:
            order = order_map.get(cluster.id)
            if order is None:
                continue

            x0, y0, x1, y1 = cluster.bbox.as_tuple()
            x0 *= scale_x
            x1 *= scale_x
            y0 *= scale_x  # Keep consistent with draw_clusters current rendering.
            y1 *= scale_y
            x0, y0, x1, y1 = _as_valid_rect(x0, y0, x1, y1)

            text = str(order)
            text_mask = _create_text_mask(text=text, font=order_font, min_height=20)
            text_w = float(text_mask.width)
            text_h = float(text_mask.height)
            pad = 2
            box_w = text_w + pad * 2
            box_h = text_h + pad * 2

            # Prefer positions right next to the item label text box.
            label_box = label_boxes_by_cluster_id.get(id(cluster))
            if label_box is None:
                label_box = (x0, y0, x0, y0)
            candidate_boxes = [
                _clamp_box(label_box[2] + 4, label_box[1], box_w, box_h),  # right of label
                _clamp_box(label_box[0] - box_w - 4, label_box[1], box_w, box_h),  # left of label
                _clamp_box(label_box[0], label_box[1] - box_h - 3, box_w, box_h),  # above label
                _clamp_box(label_box[0], label_box[3] + 3, box_w, box_h),  # below label
                _clamp_box(x1 + 3, y0 + 2, box_w, box_h),  # cluster fallback
                _clamp_box(x0 - box_w - 3, y0 + 2, box_w, box_h),
                _clamp_box(x0 + 2, y1 - box_h - 2, box_w, box_h),
                _clamp_box(x1 - box_w - 2, y1 - box_h - 2, box_w, box_h),
            ]

            selected_box = None
            best_score = None
            for candidate in candidate_boxes:
                overlap_count = 0
                overlap_area = 0.0

                for box in label_boxes:
                    if _boxes_intersect(candidate, box):
                        overlap_count += 1
                        overlap_area += _overlap_area(candidate, box)

                for box in placed_badges:
                    if _boxes_intersect(candidate, box):
                        overlap_count += 1
                        overlap_area += _overlap_area(candidate, box)

                score = (overlap_count, overlap_area)
                if selected_box is None or score < best_score:
                    selected_box = candidate
                    best_score = score
                    if overlap_count == 0:
                        break

            if selected_box is None:
                selected_box = _clamp_box(x0 + 2, y0 + 2, box_w, box_h)

            box_x0, box_y0, box_x1, box_y1 = selected_box

            draw.rounded_rectangle(
                [(box_x0, box_y0), (box_x1, box_y1)],
                radius=3,
                fill=(255, 240, 120, 230),
                outline=(0, 0, 0, 255),
                width=1,
            )
            text_w_px = text_mask.width
            text_h_px = text_mask.height
            text_x0 = int(round(box_x0 + pad))
            text_y0 = int(round(box_y0 + pad))
            text_x0 = min(max(text_x0, 0), max(image.width - text_w_px, 0))
            text_y0 = min(max(text_y0, 0), max(image.height - text_h_px, 0))
            image.paste(
                (0, 0, 0),
                (
                    text_x0,
                    text_y0,
                    text_x0 + text_w_px,
                    text_y0 + text_h_px,
                ),
                text_mask,
            )
            placed_badges.append(selected_box)

    def _process_page(self, conv_res: ConversionResult, page: Page) -> Page:
        assert page._backend is not None
        if not page._backend.is_valid():
            return page

        with TimeRecorder(conv_res, "layout"):
            assert page.size is not None
            page_image = page.get_image(scale=self.pipeline_options.images_scale)
            assert page_image is not None

            buffer = io.BytesIO()
            page_image.save(
                buffer, format="PNG"
            )  # PNG 형식으로 저장 (필요에 따라 JPEG 등 변경 가능)
            buffer.seek(0)

            # 바이트 스트림을 base64로 인코딩
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

            total_attempts = self.retry_count + 1
            response = None
            result = None
            usage = None
            _vlm_started_at = time.monotonic()
            for attempt in range(1, total_attempts + 1):
                try:
                    response_text, usage = call_vlm_server(
                        prompt=prompt,
                        base64_image=base64_image,
                        url=self.dotocr_endpoint,
                        api_key=self.api_key,
                        model=self.model,
                        max_completion_tokens=self.max_completion_tokens,
                        timeout=self.timeout,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        repetition_penalty=self.repetition_penalty,
                    )
                    if not isinstance(response_text, str) or not response_text.strip():
                        raise ValueError("Empty VLM response text")

                    # _log.debug(f"dotsocr raw response (page={page.page_no}, attempt={attempt}): {response_text}")
                    response = _parse_vlm_json_response(response_text)
                except Exception:
                    if attempt >= total_attempts:
                        raise
                    _log.warning(
                        "DotsOCR layout request failed (page=%s, attempt=%d/%d). Retrying...",
                        page.page_no,
                        attempt,
                        total_attempts,
                        exc_info=True,
                    )
                    continue

                result = _extract_layout_result_items(response)
                if isinstance(result, list):
                    if attempt > 1:
                        _log.info(
                            "DotsOCR layout request recovered after retry (page=%s, attempt=%d/%d).",
                            page.page_no,
                            attempt,
                            total_attempts,
                        )
                    break

                if attempt < total_attempts:
                    _log.warning(
                        "Unexpected VLM response schema (page=%s, attempt=%d/%d). Retrying. Parsed type=%s; value=%r",
                        page.page_no,
                        attempt,
                        total_attempts,
                        type(response).__name__,
                        response,
                    )
                    continue

                _log.warning(
                    "Unexpected VLM response schema after retries (page=%s, attempts=%d). Falling back to empty predictions. Parsed type=%s; value=%r",
                    page.page_no,
                    total_attempts,
                    type(response).__name__,
                    response,
                )
                result = []

            assert isinstance(result, list)
            _vlm_elapsed = time.monotonic() - _vlm_started_at
            _record_stage_elapsed(conv_res, "dotsocr_vlm_call", _vlm_elapsed)
            if isinstance(usage, dict):
                _log.info(
                    "DotsOCR VLM usage (page=%s): prompt_tokens=%s, completion_tokens=%s, total_tokens=%s, elapsed=%.3fs",
                    page.page_no,
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                    usage.get("total_tokens"),
                    _vlm_elapsed,
                )

            _parse_started_at = time.monotonic()
            clusters = []
            raw_table_html_by_cluster_id: dict[int, str] = {}
            raw_formula_latex_by_cluster_id: dict[int, str] = {}
            raw_text_by_cluster_id: dict[int, str] = {}
            for idx, pred_item in enumerate(result):
                if not isinstance(pred_item, dict):
                    _log.warning(
                        "Skipping non-dict layout item at index %d: %r",
                        idx,
                        pred_item,
                    )
                    continue

                category = pred_item.get("category")
                if category is None or pred_item.get("bbox") is None:
                    _log.warning(
                        "Skipping layout item missing required keys at index %d: %r",
                        idx,
                        pred_item,
                    )
                    continue

                try:
                    label = DocItemLabel(
                        str(category).lower().replace(" ", "_").replace("-", "_")
                    )  # Temporary, until docling-ibm-model uses docling-core types
                except ValueError:
                    _log.warning(
                        "Skipping unknown layout category '%s' at index %d",
                        category,
                        idx,
                    )
                    continue

                pred_item_text = _extract_layout_item_text(pred_item)
                if label in self.TABLE_LABELS:
                    if pred_item_text:
                        table_html = _extract_table_html(pred_item_text)
                        if table_html:
                            raw_table_html_by_cluster_id[idx] = table_html

                bbox_values = _rescale_dotsocr_bbox_to_page(
                    bbox=pred_item["bbox"],
                    source_width=page_image.width,
                    source_height=page_image.height,
                    page_width=page.size.width,
                    page_height=page.size.height,
                )
                if bbox_values is None:
                    _log.warning(
                        "Skipping invalid bbox at index %d: %r",
                        idx,
                        pred_item.get("bbox"),
                    )
                    continue
                bbox = {
                    "l": bbox_values[0],
                    "t": bbox_values[1],
                    "r": bbox_values[2],
                    "b": bbox_values[3],
                }
                cluster = Cluster(
                    id=idx,
                    label=label,
                    confidence=_coerce_confidence(pred_item.get("confidence")),
                    # bbox=BoundingBox.model_validate(pred_item),
                    bbox=BoundingBox.model_validate(bbox),
                    cells=[],
                )
                clusters.append(cluster)

                if label == self.FORMULA_LABEL:
                    if pred_item_text:
                        raw_formula_latex_by_cluster_id[idx] = pred_item_text
                elif (
                    label in self.TEXT_ELEM_LABELS
                    and pred_item_text
                    and label != self.FORMULA_LABEL
                ):
                    raw_text_by_cluster_id[idx] = pred_item_text

            # Preserve source(DotsOCR raw) order across debug views and
            # postprocessed cluster outputs.
            source_order_map = {
                cluster.id: order_idx + 1 for order_idx, cluster in enumerate(clusters)
            }

            if settings.debug.visualize_raw_layout:
                try:
                    self.draw_clusters_and_cells_side_by_side(
                        conv_res,
                        page,
                        clusters,
                        mode_prefix="1_raw_dotsocr",
                        cluster_order_map=source_order_map,
                    )
                except Exception:
                    _log.warning(
                        "Failed to render raw DotsOCR layout debug image (page=%s).",
                        page.page_no,
                        exc_info=True,
                    )

            # Run replacement before postprocess so clusters with empty parser text
            # can receive DotsOCR text and avoid being dropped as empty clusters.
            self._replace_glyph_text_with_dotsocr(
                page=page,
                clusters=clusters,
                dotsocr_text_by_cluster_id=raw_text_by_cluster_id,
            )
            _record_stage_elapsed(
                conv_res, "dotsocr_parse", time.monotonic() - _parse_started_at
            )

            _postprocess_started_at = time.monotonic()
            processed_clusters, processed_cells = LayoutPostprocessor(
                page, clusters, self.options
            ).postprocess()
            _record_stage_elapsed(
                conv_res,
                "dotsocr_postprocess",
                time.monotonic() - _postprocess_started_at,
            )

            # Note: LayoutPostprocessor updates page.cells and page.parsed_page internally
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "Mean of empty slice|invalid value encountered in scalar divide",
                    RuntimeWarning,
                    "numpy",
                )

                conv_res.confidence.pages[page.page_no].layout_score = float(
                    np.mean([c.confidence for c in processed_clusters])
                )

                conv_res.confidence.pages[page.page_no].ocr_score = float(
                    np.mean([c.confidence for c in processed_cells if c.from_ocr])
                )

            page.predictions.layout = LayoutPrediction(clusters=processed_clusters)

            equation_map: dict[int, TextElement] = {}
            for cluster in processed_clusters:
                if cluster.label != self.FORMULA_LABEL:
                    continue
                formula_latex = raw_formula_latex_by_cluster_id.get(cluster.id)
                if not formula_latex:
                    continue
                equation_map[cluster.id] = TextElement(
                    label=cluster.label,
                    id=cluster.id,
                    text=formula_latex,
                    page_no=page.page_no,
                    cluster=cluster,
                )
            page.predictions.equations_prediction = EquationPrediction(
                equation_count=len(equation_map),
                equation_map=equation_map,
            )

            if self._use_dotsocr_table_structure():
                _table_started_at = time.monotonic()
                page.predictions.tablestructure = self._build_tablestructure_from_dotsocr(
                    page=page,
                    clusters=processed_clusters,
                    table_html_by_cluster_id=raw_table_html_by_cluster_id,
                )
                _record_stage_elapsed(
                    conv_res,
                    "dotsocr_table_build",
                    time.monotonic() - _table_started_at,
                )

        if settings.debug.visualize_layout:
            try:
                self.draw_clusters_and_cells_side_by_side(
                    conv_res,
                    page,
                    processed_clusters,
                    mode_prefix="2_postprocessed_dotsocr",
                    cluster_order_map=source_order_map,
                )
            except Exception:
                _log.warning(
                    "Failed to render postprocessed DotsOCR layout debug image (page=%s).",
                    page.page_no,
                    exc_info=True,
                )

        return page

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        """
        여기서 conv_res.pages[0].predictions.layout.clusters 를 만듬
        나중에는
        conv_res.pages[0].predictions.layout.clusters -> conv_res.pages[0].assembled
        conv_res.pages[0].assembled -> conv_res.document
        """
        _log.info(f"Running GenosDotsOCRLayoutModel on {conv_res.input.file}")

        pages = list(page_batch)
        if not pages:
            return

        # Pre-create per-page sub-stage timing keys before spawning worker
        # threads so concurrent TimeRecorder/_record_stage_elapsed calls don't
        # race on dict initialization (list.append itself is atomic).
        if settings.debug.profile_pipeline_timings:
            for _key in (
                "layout",
                "dotsocr_vlm_call",
                "dotsocr_parse",
                "dotsocr_postprocess",
                "dotsocr_table_build",
            ):
                conv_res.timings.setdefault(
                    _key, ProfilingItem(scope=ProfilingScope.PAGE)
                )

        def _process(page: Page) -> Page:
            return self._process_page(conv_res, page)

        with TimeRecorder(
            conv_res, "dotsocr_layout_wallclock", scope=ProfilingScope.DOCUMENT
        ):
            with ThreadPoolExecutor(max_workers=len(pages)) as executor:
                yield from executor.map(_process, pages)


prompt = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

def call_vlm_server(
    prompt: str,
    base64_image: str,
    url: str,
    api_key: str,
    model: str,
    max_completion_tokens: int = 6000,
    timeout: int = 3600,
    temperature: float = 0.1,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
) -> tuple[str, dict | None]:
    image_data_url = f"data:image/png;base64,{base64_image}"
    prompt_with_image_token = f"<|img|><|imgpad|><|endofimg|>{prompt}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt_with_image_token,
                    },
                ],
            }
        ],
        "max_completion_tokens": max_completion_tokens,
        "temperature": temperature,
        "top_p": top_p,
        # VLM이 동일 토큰을 max_completion_tokens까지 무한 반복하는 degeneration 억제용.
        "repetition_penalty": repetition_penalty,
    }

    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if response.status_code == 400:
            fallback_payload = dict(payload)
            max_completion_tokens = fallback_payload.pop("max_completion_tokens", None)
            if max_completion_tokens is not None:
                fallback_payload["max_tokens"] = max_completion_tokens
                response = requests.post(
                    url, json=fallback_payload, headers=headers, timeout=timeout
                )

        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") if isinstance(data, dict) else None
        return data["choices"][0]["message"]["content"], usage
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HTTP 요청 오류: {e}") from e
    except KeyError as e:
        raise ValueError(f"응답 파싱 오류: {e}\n응답 본문: {response.text}") from e


def _extract_layout_result_items(response):
    if isinstance(response, dict):
        result = response.get("result")
        if result is None:
            # Fallback for providers that use a different list key.
            result = response.get("items")
    elif isinstance(response, list):
        result = response
    else:
        result = None

    if isinstance(result, str):
        nested = _parse_vlm_json_response(result)
        if isinstance(nested, dict):
            result = nested.get("result")
            if result is None:
                result = nested.get("items")
        elif isinstance(nested, list):
            result = nested

    return result


def _parse_vlm_json_response(response_text: str):
    if not isinstance(response_text, str):
        raise TypeError(
            f"VLM response must be str, got {type(response_text).__name__}: {response_text!r}"
        )

    text = response_text.strip()
    candidates = []

    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())
    candidates.append(text)
    candidates.extend(_extract_balanced_json_candidates(text))
    candidates = _dedupe_keep_order(candidates)
    candidates = sorted(
        candidates,
        key=lambda c: (
            bool(re.search(r"['\"]bbox['\"]", c, flags=re.IGNORECASE)),
            bool(re.search(r"['\"]category['\"]", c, flags=re.IGNORECASE)),
            len(c),
        ),
        reverse=True,
    )

    for candidate in candidates:
        parsed = _try_parse_json(candidate)
        if parsed is not None and _is_layout_json_container(parsed):
            return parsed

    recovered = _recover_layout_items_from_text(text)
    if recovered:
        _log.warning(
            "Recovered %d layout items from non-JSON VLM output via regex fallback.",
            len(recovered),
        )
        return recovered

    _log.warning(
        "Failed to parse VLM output as JSON after fallback. Returning empty predictions. raw=%r",
        response_text,
    )
    return []


def _try_parse_json(candidate: str):
    for payload in (
        candidate,
        re.sub(r",\s*([}\]])", r"\1", candidate),  # drop trailing commas
    ):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            continue
    return None


def _extract_balanced_json_candidates(text: str, max_candidates: int = 64):
    candidates = []
    for idx, ch in enumerate(text):
        if ch not in "{[":
            continue
        candidate = _extract_balanced_fragment(text, idx)
        if candidate:
            candidates.append(candidate.strip())
            if len(candidates) >= max_candidates:
                break
    return candidates


def _extract_balanced_fragment(text: str, start_idx: int):
    stack = []
    in_string = False
    escape = False

    for i in range(start_idx, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            stack.append("}")
            continue
        if ch == "[":
            stack.append("]")
            continue
        if ch in "}]":
            if not stack or ch != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return text[start_idx : i + 1]
    return None


def _is_layout_json_container(parsed) -> bool:
    if isinstance(parsed, dict):
        return True
    if isinstance(parsed, list):
        if not parsed:
            return True
        return any(isinstance(item, dict) for item in parsed)
    return False


def _recover_layout_items_from_text(text: str):
    items = []
    for segment in re.findall(r"\{[^{}]{1,4000}\}", text, flags=re.DOTALL):
        item = _extract_layout_item_from_text_segment(segment)
        if item is not None:
            items.append(item)

    if not items:
        patterns = [
            r"['\"]category['\"]\s*:\s*['\"](?P<category>[^'\"]+)['\"].{0,1200}?['\"]bbox['\"]\s*:\s*\[(?P<bbox>[^\]]+)\]",
            r"['\"]bbox['\"]\s*:\s*\[(?P<bbox>[^\]]+)\].{0,1200}?['\"]category['\"]\s*:\s*['\"](?P<category>[^'\"]+)['\"]",
            r"(?:['\"]?category['\"]?)\s*:\s*['\"]?(?P<category>[A-Za-z0-9_\- ]+)['\"]?.{0,1200}?(?:['\"]?bbox['\"]?)\s*:\s*\[(?P<bbox>[^\]]+)\]",
            r"(?:['\"]?bbox['\"]?)\s*:\s*\[(?P<bbox>[^\]]+)\].{0,1200}?(?:['\"]?category['\"]?)\s*:\s*['\"]?(?P<category>[A-Za-z0-9_\- ]+)['\"]?",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                item = _build_layout_item(
                    category=match.group("category"), bbox_value=match.group("bbox")
                )
                if item is not None:
                    items.append(item)

    deduped_items = []
    seen = set()
    for item in items:
        bbox = item["bbox"]
        key = (
            item["category"].strip().lower(),
            round(float(bbox[0]), 2),
            round(float(bbox[1]), 2),
            round(float(bbox[2]), 2),
            round(float(bbox[3]), 2),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped_items.append(item)
    return deduped_items


def _extract_layout_item_from_text_segment(segment: str):
    parsed_segment = _try_parse_json(segment)
    if isinstance(parsed_segment, dict):
        category = parsed_segment.get("category")
        bbox = parsed_segment.get("bbox")
        if category is not None and bbox is not None:
            return _build_layout_item(
                category=category,
                bbox_value=bbox,
                text=_extract_layout_item_text(parsed_segment),
            )

    category_match = re.search(
        r"(?:['\"]?category['\"]?)\s*:\s*['\"]?([^,'\"}\n]+)['\"]?",
        segment,
        flags=re.IGNORECASE,
    )
    bbox_match = re.search(
        r"(?:['\"]?bbox['\"]?)\s*:\s*\[([^\]]+)\]",
        segment,
        flags=re.IGNORECASE,
    )
    if category_match is None or bbox_match is None:
        return None
    return _build_layout_item(
        category=category_match.group(1), bbox_value=bbox_match.group(1)
    )


def _build_layout_item(category: str, bbox_value, text: Optional[str] = None):
    if isinstance(bbox_value, (list, tuple)):
        try:
            bbox_values = [float(v) for v in bbox_value[:4]]
        except (TypeError, ValueError):
            return None
    else:
        bbox_values = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", str(bbox_value))]

    if len(bbox_values) < 4:
        return None

    item = {"category": str(category).strip(), "bbox": bbox_values[:4]}
    if isinstance(text, str) and text.strip():
        item["text"] = text
    return item


def _dedupe_keep_order(values):
    deduped = []
    seen = set()
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _coerce_confidence(value) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 1.0
    if math.isnan(confidence) or math.isinf(confidence):
        return 1.0
    return max(0.0, min(confidence, 1.0))


def _extract_layout_item_text(layout_item: dict) -> Optional[str]:
    if not isinstance(layout_item, dict):
        return None

    for key in ("text", "html", "content"):
        value = layout_item.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return None


def _extract_table_html(raw_text: str) -> Optional[str]:
    if not isinstance(raw_text, str):
        return None

    text = raw_text.strip()
    if not text:
        return None

    soup = BeautifulSoup(text, "html.parser")
    first_table = soup.find("table")
    if isinstance(first_table, Tag):
        return str(first_table)

    # Fallback for malformed/non-HTML-like payloads:
    # find the first balanced <table>...</table> block without truncating nested tables.
    tag_pattern = re.compile(r"</?table\b[^>]*>", re.IGNORECASE)
    start_idx = None
    depth = 0

    for match in tag_pattern.finditer(text):
        token = match.group(0)
        is_close = token.startswith("</")
        if not is_close:
            if start_idx is None:
                start_idx = match.start()
                depth = 1
            else:
                depth += 1
            continue

        if start_idx is None:
            continue

        depth -= 1
        if depth == 0:
            return text[start_idx:match.end()]

    return None


def _parse_html_to_table_data(html: str) -> Optional[TableData]:
    from docling.backend.genos_vlm_html_backend import (
        GenosVlmHTMLDocumentBackend,
    )

    soup = BeautifulSoup(html, "html.parser")
    table_tag = soup.find("table")
    if table_tag is None or not isinstance(table_tag, Tag):
        return None

    # Backend parser rejects nested tables. Flatten them into plain text.
    nested_table = table_tag.find("table")
    while isinstance(nested_table, Tag):
        replacement_text = nested_table.get_text(" ", strip=True)
        if replacement_text:
            nested_table.replace_with(replacement_text)
        else:
            nested_table.decompose()
        nested_table = table_tag.find("table")

    return GenosVlmHTMLDocumentBackend.parse_table_data(table_tag)


def _rescale_dotsocr_bbox_to_page(
    bbox,
    source_width: float,
    source_height: float,
    page_width: float,
    page_height: float,
):
    raw_bbox = _extract_bbox_values(bbox)
    if raw_bbox is None:
        return None
    if source_width <= 0 or source_height <= 0 or page_width <= 0 or page_height <= 0:
        return None

    try:
        resized_h, resized_w = _smart_resize(
            height=int(round(source_height)),
            width=int(round(source_width)),
            factor=DOTSOCR_IMAGE_FACTOR,
            min_pixels=DOTSOCR_MIN_PIXELS,
            max_pixels=DOTSOCR_MAX_PIXELS,
        )
    except ValueError:
        resized_h, resized_w = source_height, source_width

    scale_x = float(resized_w) / float(source_width)
    scale_y = float(resized_h) / float(source_height)
    if scale_x <= 0 or scale_y <= 0:
        return None

    bbox_on_source = [
        raw_bbox[0] / scale_x,
        raw_bbox[1] / scale_y,
        raw_bbox[2] / scale_x,
        raw_bbox[3] / scale_y,
    ]
    bbox_on_source = _sanitize_bbox(bbox_on_source, source_width, source_height)
    if bbox_on_source is None:
        return None

    page_scale_x = float(page_width) / float(source_width)
    page_scale_y = float(page_height) / float(source_height)
    bbox_on_page = [
        bbox_on_source[0] * page_scale_x,
        bbox_on_source[1] * page_scale_y,
        bbox_on_source[2] * page_scale_x,
        bbox_on_source[3] * page_scale_y,
    ]
    return _sanitize_bbox(bbox_on_page, page_width, page_height)


def _extract_bbox_values(bbox):
    if isinstance(bbox, dict):
        candidate = [bbox.get("l"), bbox.get("t"), bbox.get("r"), bbox.get("b")]
    else:
        candidate = bbox
    if not isinstance(candidate, (list, tuple)) or len(candidate) < 4:
        return None
    try:
        return [float(candidate[0]), float(candidate[1]), float(candidate[2]), float(candidate[3])]
    except (TypeError, ValueError):
        return None


def _sanitize_bbox(bbox_values, width: float, height: float):
    try:
        l, t, r, b = [float(v) for v in bbox_values]
    except (TypeError, ValueError):
        return None

    if any(math.isnan(v) or math.isinf(v) for v in (l, t, r, b)):
        return None

    if r < l:
        l, r = r, l
    if b < t:
        t, b = b, t

    l = max(0.0, min(l, float(width)))
    t = max(0.0, min(t, float(height)))
    r = max(0.0, min(r, float(width)))
    b = max(0.0, min(b, float(height)))

    if r == l:
        if l >= float(width):
            l = max(0.0, float(width) - 1.0)
            r = float(width)
        else:
            r = min(float(width), l + 1.0)
    if b == t:
        if t >= float(height):
            t = max(0.0, float(height) - 1.0)
            b = float(height)
        else:
            b = min(float(height), t + 1.0)

    if r <= l or b <= t:
        return None
    return [l, t, r, b]


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize(
    height: int,
    width: int,
    factor: int = DOTSOCR_IMAGE_FACTOR,
    min_pixels: int = DOTSOCR_MIN_PIXELS,
    max_pixels: int = DOTSOCR_MAX_PIXELS,
):
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size: height={height}, width={width}")

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((h_bar * w_bar) / max_pixels)
            h_bar = max(factor, _floor_by_factor(h_bar / beta, factor))
            w_bar = max(factor, _floor_by_factor(w_bar / beta, factor))

    return int(h_bar), int(w_bar)
