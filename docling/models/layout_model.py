import copy
import logging
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
from docling.utils.layout_postprocessor import LayoutPostprocessor
from docling.utils.profiling import TimeRecorder
from docling.utils.visualization import draw_clusters

_log = logging.getLogger(__name__)


class LayoutModel(BasePageModel):
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

    def __init__(
        self,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        options: LayoutOptions,
    ):
        from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor

        self.options = options

        device = decide_device(accelerator_options.device)
        layout_model_config = options.model_spec
        model_repo_folder = layout_model_config.model_repo_folder
        model_path = layout_model_config.model_path

        if artifacts_path is None:
            artifacts_path = (
                self.download_models(layout_model_config=layout_model_config)
                / model_path
            )
        else:
            if (artifacts_path / model_repo_folder).exists():
                artifacts_path = artifacts_path / model_repo_folder / model_path
            elif (artifacts_path / model_path).exists():
                warnings.warn(
                    "The usage of artifacts_path containing directly "
                    f"{model_path} is deprecated. Please point "
                    "the artifacts_path to the parent containing "
                    f"the {model_repo_folder} folder.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                artifacts_path = artifacts_path / model_path

        self.layout_predictor = LayoutPredictor(
            artifact_path=str(artifacts_path),
            device=device,
            num_threads=accelerator_options.num_threads,
        )

    @staticmethod
    def download_models(
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
        layout_model_config: LayoutModelConfig = DOCLING_LAYOUT_V2,
    ) -> Path:
        return download_hf_model(
            repo_id=layout_model_config.repo_id,
            revision=layout_model_config.revision,
            local_dir=local_dir,
            force=force,
            progress=progress,
        )

    def draw_clusters_and_cells_side_by_side(
        self, conv_res, page, clusters, mode_prefix: str, show: bool = False
    ):
        """
        Draws all layout clusters on a single page image for debug visualization.
        """
        scale_x = page.image.width / page.size.width
        scale_y = page.image.height / page.size.height

        order_map = {cluster.id: idx + 1 for idx, cluster in enumerate(clusters)}

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
            y0 *= scale_y
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
            y0 *= scale_y
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
        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "layout"):
                    assert page.size is not None
                    page_image = page.get_image(scale=1.0)
                    assert page_image is not None

                    clusters = []
                    for ix, pred_item in enumerate(
                        self.layout_predictor.predict(page_image)
                    ):
                        label = DocItemLabel(
                            pred_item["label"]
                            .lower()
                            .replace(" ", "_")
                            .replace("-", "_")
                        )  # Temporary, until docling-ibm-model uses docling-core types
                        cluster = Cluster(
                            id=ix,
                            label=label,
                            confidence=pred_item["confidence"],
                            bbox=BoundingBox.model_validate(pred_item),
                            cells=[],
                        )
                        clusters.append(cluster)

                    if settings.debug.visualize_raw_layout:
                        self.draw_clusters_and_cells_side_by_side(
                            conv_res, page, clusters, mode_prefix="raw"
                        )

                    # Apply postprocessing

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
                        conv_res, page, processed_clusters, mode_prefix="postprocessed"
                    )

                yield page
