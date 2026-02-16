import logging
from collections.abc import Iterable, Sequence
from itertools import groupby
from pathlib import Path
from typing import Optional

import numpy
import torch
import torchvision.transforms as T
from docling_core.types.doc import BoundingBox, DocItemLabel, TableCell
from docling_core.types.doc.page import (
    BoundingRectangle,
    TextCellUnit,
)
from PIL import Image, ImageDraw
from transformers import AutoTokenizer

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import Page, Table, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import TableStructureV2Options
from docling.datamodel.settings import settings
from docling.models.base_table_model import BaseTableStructureModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class TableStructureModelV2(BaseTableStructureModel):
    """TableFormerV2 model for table structure recognition."""

    _model_repo_id = "docling-project/TableFormerV2"
    _model_repo_folder = "docling-project--TableFormerV2"

    # Cell tokens that produce bboxes
    _cell_tokens = ["fcel", "ecel", "ched", "rhed", "srow"]

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: TableStructureV2Options,
        accelerator_options: AcceleratorOptions,
    ):
        self.options = options
        self.do_cell_matching = self.options.do_cell_matching
        self.enabled = enabled

        if self.enabled:
            # Determine model path
            if artifacts_path is None:
                model_path = self.download_models()
            elif (artifacts_path / self._model_repo_folder).exists():
                model_path = artifacts_path / self._model_repo_folder
            else:
                model_path = artifacts_path

            # Determine device
            device = decide_device(accelerator_options.device)
            if device == AcceleratorDevice.MPS.value:
                device = AcceleratorDevice.CPU.value
            self.device = device

            # Set number of threads for CPU inference
            if device == "cpu":
                torch.set_num_threads(accelerator_options.num_threads)

            # Load model and tokenizer
            from docling_ibm_models.tableformer_v2 import TableFormerV2

            self.model = TableFormerV2.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Image preprocessing
            self.transform = T.Compose(
                [
                    T.Resize((448, 448)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            self.scale = 2.0  # Scale up table input images to 144 dpi

    @classmethod
    def get_options_type(cls) -> type[TableStructureV2Options]:
        return TableStructureV2Options

    @staticmethod
    def download_models(
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
    ) -> Path:
        return download_hf_model(
            repo_id=TableStructureModelV2._model_repo_id,
            local_dir=local_dir,
            force=force,
            progress=progress,
        )

    def _decode_otsl_sequence(self, token_ids: torch.Tensor) -> list[str]:
        """
        Decode token IDs to OTSL tag sequence.

        Strips angle brackets from tokens (e.g., <fcel> -> fcel) to match
        the format expected by otsl_to_html.
        """
        tags = []
        skip_tokens = {"<pad>", "[UNK]", "<start>", "<end>"}

        for tid in token_ids.tolist():
            token = self.tokenizer.decode([tid]).strip()
            if token in skip_tokens:
                continue
            # Strip angle brackets: <fcel> -> fcel
            if token.startswith("<") and token.endswith(">"):
                token = token[1:-1]
            tags.append(token)
        return tags

    def _build_table_cells(
        self,
        otsl_seq: list[str],
        bboxes: torch.Tensor,
        table_bbox: list[float],
    ) -> tuple[list[dict], int, int]:
        """
        Build table cell structures from OTSL sequence and bboxes.

        Parameters
        ----------
        otsl_seq : list[str]
            OTSL tag sequence (fcel, ecel, nl, lcel, ucel, xcel, etc.)
        bboxes : torch.Tensor
            Predicted bboxes in xyxy format [0, 1], shape (num_cells, 4)
        table_bbox : list[float]
            Table bounding box [x1, y1, x2, y2] in page coordinates

        Returns
        -------
        tuple
            (table_cells, num_rows, num_cols)
        """
        # Split into rows by "nl" token
        rows = [
            list(group) for k, group in groupby(otsl_seq, lambda x: x == "nl") if not k
        ]

        if not rows:
            return [], 0, 0

        num_rows = len(rows)
        num_cols = max(len(row) for row in rows) if rows else 0

        # Build 2D grid for span detection
        grid = []
        for row in rows:
            grid.append(row + [""] * (num_cols - len(row)))

        # Build cells
        table_cells = []
        bbox_idx = 0
        t_x1, t_y1, t_x2, t_y2 = table_bbox
        t_w = t_x2 - t_x1
        t_h = t_y2 - t_y1

        for row_idx, row in enumerate(grid):
            for col_idx, tag in enumerate(row):
                if tag not in self._cell_tokens:
                    continue

                # Get bbox
                cell_bbox = None
                if bbox_idx < bboxes.shape[0]:
                    bbox = bboxes[bbox_idx].tolist()
                    cell_bbox = {
                        "l": t_x1 + bbox[0] * t_w,
                        "t": t_y1 + bbox[1] * t_h,
                        "r": t_x1 + bbox[2] * t_w,
                        "b": t_y1 + bbox[3] * t_h,
                    }
                    bbox_idx += 1

                # Detect colspan (lcel to the right)
                colspan = 1
                for c in range(col_idx + 1, num_cols):
                    if grid[row_idx][c] == "lcel":
                        colspan += 1
                    else:
                        break

                # Detect rowspan (ucel below)
                rowspan = 1
                for r in range(row_idx + 1, num_rows):
                    if grid[r][col_idx] == "ucel":
                        rowspan += 1
                    else:
                        break

                cell = {
                    "bbox": cell_bbox,
                    "row_span": rowspan,
                    "col_span": colspan,
                    "start_row_offset_idx": row_idx,
                    "end_row_offset_idx": row_idx + rowspan,
                    "start_col_offset_idx": col_idx,
                    "end_col_offset_idx": col_idx + colspan,
                    "column_header": tag == "ched",
                    "row_header": tag == "rhed",
                    "row_section": tag == "srow",
                }
                table_cells.append(cell)

        return table_cells, num_rows, num_cols

    def draw_table_and_cells(
        self,
        conv_res: ConversionResult,
        page: Page,
        tbl_list: Iterable[Table],
        show: bool = False,
    ):
        assert page._backend is not None
        assert page.size is not None

        image = page._backend.get_page_image()

        scale_x = image.width / page.size.width
        scale_y = image.height / page.size.height

        draw = ImageDraw.Draw(image)

        for table_element in tbl_list:
            x0, y0, x1, y1 = table_element.cluster.bbox.as_tuple()
            draw.rectangle(
                [(x0 * scale_x, y0 * scale_y), (x1 * scale_x, y1 * scale_y)],
                outline="red",
            )

            for tc in table_element.table_cells:
                if tc.bbox is not None:
                    x0, y0, x1, y1 = tc.bbox.as_tuple()
                    width = 3 if tc.column_header else 1
                    draw.rectangle(
                        [(x0 * scale_x, y0 * scale_y), (x1 * scale_x, y1 * scale_y)],
                        outline="blue",
                        width=width,
                    )
                    draw.text(
                        (x0 * scale_x + 3, y0 * scale_y + 3),
                        text=f"{tc.start_row_offset_idx}, {tc.start_col_offset_idx}",
                        fill="black",
                    )

        if show:
            image.show()
        else:
            out_path: Path = (
                Path(settings.debug.debug_output_path)
                / f"debug_{conv_res.input.file.stem}"
            )
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / f"table_struct_v2_page_{page.page_no:05}.png"
            image.save(str(out_file), format="png")

    def predict_tables(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[TableStructurePrediction]:
        pages = list(pages)
        predictions: list[TableStructurePrediction] = []

        for page in pages:
            assert page._backend is not None
            if not page._backend.is_valid():
                existing_prediction = (
                    page.predictions.tablestructure or TableStructurePrediction()
                )
                page.predictions.tablestructure = existing_prediction
                predictions.append(existing_prediction)
                continue

            with TimeRecorder(conv_res, "table_structure"):
                assert page.predictions.layout is not None
                assert page.size is not None

                table_prediction = TableStructurePrediction()
                page.predictions.tablestructure = table_prediction

                in_tables = [
                    (
                        cluster,
                        [
                            round(cluster.bbox.l) * self.scale,
                            round(cluster.bbox.t) * self.scale,
                            round(cluster.bbox.r) * self.scale,
                            round(cluster.bbox.b) * self.scale,
                        ],
                    )
                    for cluster in page.predictions.layout.clusters
                    if cluster.label
                    in [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]
                ]

                if not in_tables:
                    predictions.append(table_prediction)
                    continue

                page_image = numpy.asarray(page.get_image(scale=self.scale))

                for table_cluster, tbl_box in in_tables:
                    # Crop table region
                    x1, y1, x2, y2 = [int(v) for v in tbl_box]
                    table_image = page_image[y1:y2, x1:x2]

                    # Convert to PIL and preprocess
                    pil_image = Image.fromarray(table_image).convert("RGB")
                    image_tensor = (
                        self.transform(pil_image).unsqueeze(0).to(self.device)
                    )

                    # Run inference
                    with torch.no_grad():
                        output = self.model.generate(
                            image_tensor, self.tokenizer, max_length=512
                        )

                    # Decode tokens to OTSL sequence
                    generated_ids = output["generated_ids"][0]
                    otsl_seq = self._decode_otsl_sequence(generated_ids)

                    # Get bboxes
                    pred_bboxes = output["predicted_bboxes"]
                    if pred_bboxes is not None:
                        pred_bboxes = pred_bboxes[0]
                        valid_mask = pred_bboxes.sum(dim=-1) > 0
                        pred_bboxes = pred_bboxes[valid_mask]
                    else:
                        pred_bboxes = torch.empty(0, 4)

                    # Build table cells
                    table_bbox = [v / self.scale for v in tbl_box]
                    cell_data, num_rows, num_cols = self._build_table_cells(
                        otsl_seq, pred_bboxes, table_bbox
                    )

                    # Build TableCell objects
                    table_cells = []
                    for element in cell_data:
                        if element["bbox"] is not None:
                            bbox = BoundingBox.model_validate(element["bbox"])
                            # Always extract text from the PDF backend for V2
                            # (V2 doesn't have a separate cell matching pipeline like V1)
                            text_piece = page._backend.get_text_in_rect(bbox)
                            element["bbox"]["token"] = text_piece
                        tc = TableCell.model_validate(element)
                        table_cells.append(tc)

                    tbl = Table(
                        otsl_seq=otsl_seq,
                        table_cells=table_cells,
                        num_rows=num_rows,
                        num_cols=num_cols,
                        id=table_cluster.id,
                        page_no=page.page_no,
                        cluster=table_cluster,
                        label=table_cluster.label,
                    )

                    table_prediction.table_map[table_cluster.id] = tbl

                if settings.debug.visualize_tables:
                    self.draw_table_and_cells(
                        conv_res,
                        page,
                        page.predictions.tablestructure.table_map.values(),
                    )

                predictions.append(table_prediction)

        return predictions
