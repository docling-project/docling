import io
import base64
import json
import math
import requests
import copy
import logging
import re
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import numpy as np
from docling_core.types.doc import DocItemLabel
from PIL import Image, ImageDraw, ImageFont

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import BoundingBox, Cluster, LayoutPrediction, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_V2, LayoutModelConfig
from docling.datamodel.pipeline_options import LayoutOptions
from docling.datamodel.settings import settings
from docling.models.base_model import BasePageModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device

from docling.utils.dotsocr_postprocessor import LayoutPostprocessor
from docling.utils.profiling import TimeRecorder
from docling.utils.visualization import draw_clusters
from docling.datamodel.pipeline_options import PdfPipelineOptions

_log = logging.getLogger(__name__)

DOTSOCR_IMAGE_FACTOR = 28
DOTSOCR_MIN_PIXELS = 3136
DOTSOCR_MAX_PIXELS = 11289600


class DotsOCRLayoutModel(BasePageModel):
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
        self.dotocr_endpoint = self.options.dotocr_endpoint
        self.api_key = self.options.dotocr_api_key
        self.max_completion_tokens = self.options.dotocr_max_completion_tokens
        self.timeout = self.options.dotocr_timeout

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
            image.paste(
                (0, 0, 0),
                (
                    int(round(box_x0 + pad)),
                    int(round(box_y0 + pad)),
                    int(round(box_x0 + pad + text_w)),
                    int(round(box_y0 + pad + text_h)),
                ),
                text_mask,
            )
            placed_badges.append(selected_box)

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        """
        여기서 conv_res.pages[0].predictions.layout.clusters 를 만듬
        나중에는
        conv_res.pages[0].predictions.layout.clusters -> conv_res.pages[0].assembled
        conv_res.pages[0].assembled -> conv_res.document
        """

        _log.info(f"Running DotsOCRLayoutModel on {conv_res.input.file}")

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "layout"):
                    assert page.size is not None
                    page_image = page.get_image(
                        scale=self.pipeline_options.images_scale
                    )
                    assert page_image is not None

                    buffer = io.BytesIO()
                    page_image.save(
                        buffer, format="PNG"
                    )  # PNG 형식으로 저장 (필요에 따라 JPEG 등 변경 가능)
                    buffer.seek(0)

                    # 바이트 스트림을 base64로 인코딩
                    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

                    response_text = call_vlm_server(
                        prompt=prompt,
                        base64_image=base64_image,
                        url=self.dotocr_endpoint,
                        api_key=self.api_key,
                        model="model",
                        max_completion_tokens=self.max_completion_tokens,
                        timeout=self.timeout,
                    )
                    response = _parse_vlm_json_response(response_text)
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

                    if not isinstance(result, list):
                        _log.warning(
                            "Unexpected VLM response schema. Parsed type=%s; falling back to empty predictions. value=%r",
                            type(response).__name__,
                            response,
                        )
                        result = []

                    clusters = []
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
                                str(category)
                                .lower()
                                .replace(" ", "_")
                                .replace("-", "_")
                            )  # Temporary, until docling-ibm-model uses docling-core types
                        except ValueError:
                            _log.warning(
                                "Skipping unknown layout category '%s' at index %d",
                                category,
                                idx,
                            )
                            continue

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

                    # Preserve source(DotsOCR raw) order across debug views and
                    # postprocessed cluster outputs.
                    source_order_map = {
                        cluster.id: order_idx + 1
                        for order_idx, cluster in enumerate(clusters)
                    }

                    if settings.debug.visualize_raw_layout:
                        self.draw_clusters_and_cells_side_by_side(
                            conv_res,
                            page,
                            clusters,
                            mode_prefix="1_raw_dotsocr",
                            cluster_order_map=source_order_map,
                        )

                    # processed_clusters

                    processed_clusters, processed_cells = LayoutPostprocessor(
                        page, clusters, self.options
                    ).postprocess()

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
                            np.mean(
                                [c.confidence for c in processed_cells if c.from_ocr]
                            )
                        )

                    page.predictions.layout = LayoutPrediction(
                        clusters=processed_clusters
                    )

                if settings.debug.visualize_layout:
                    self.draw_clusters_and_cells_side_by_side(
                        conv_res,
                        page,
                        processed_clusters,
                        mode_prefix="2_postprocessed_dotsocr",
                        cluster_order_map=source_order_map,
                    )
                yield page


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
) -> str:
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
        "temperature": 0.0,
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
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HTTP 요청 오류: {e}") from e
    except KeyError as e:
        raise ValueError(f"응답 파싱 오류: {e}\n응답 본문: {response.text}") from e


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
                    category=match.group("category"), bbox_text=match.group("bbox")
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
        category=category_match.group(1), bbox_text=bbox_match.group(1)
    )


def _build_layout_item(category: str, bbox_text: str):
    bbox_values = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", bbox_text)]
    if len(bbox_values) < 4:
        return None
    return {"category": category.strip(), "bbox": bbox_values[:4]}


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
