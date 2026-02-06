"""Table structure stage powered by object-detection runtimes."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence

from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, TableCell, TableData
from docling_core.types.doc.page import TextCell, TextCellUnit
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Cluster, Page, Table, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import TableStructureObjectDetectionOptions
from docling.models.base_table_model import BaseTableStructureModel
from docling.models.inference_engines.object_detection import (
    BaseObjectDetectionEngine,
    ObjectDetectionEngineInput,
    ObjectDetectionEnginePrediction,
    create_object_detection_engine,
)
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class TableObjectDetectionModel(BaseTableStructureModel):
    """Detect table structure using the shared object-detection runtime."""

    SCORE_THRESHOLD = 0.6

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        options: TableStructureObjectDetectionOptions,
    ) -> None:
        del accelerator_options
        self.enabled = enabled
        self.options = options
        self.scale = options.scale
        self.concurrency = options.concurrency

        self.engine: Optional[BaseObjectDetectionEngine] = None
        if self.enabled:
            engine_options = options.engine_options.model_copy(deep=True)
            if artifacts_path is not None:
                engine_options.artifacts_path = artifacts_path

            self.engine = create_object_detection_engine(
                engine_options,
                model_spec=self.options.model_spec,
            )
            self.engine.initialize()

    @classmethod
    def get_options_type(cls) -> type[TableStructureObjectDetectionOptions]:
        return TableStructureObjectDetectionOptions

    def predict_tables(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[TableStructurePrediction]:
        if not self.enabled or self.engine is None:
            return [TableStructurePrediction() for _ in pages]

        def process_one_page(page: Page) -> TableStructurePrediction:
            assert page._backend is not None
            if not page._backend.is_valid():
                existing = page.predictions.tablestructure or TableStructurePrediction()
                page.predictions.tablestructure = existing
                return existing

            if page.predictions.layout is None:
                empty_prediction = TableStructurePrediction()
                page.predictions.tablestructure = empty_prediction
                return empty_prediction

            prediction = TableStructurePrediction()
            page.predictions.tablestructure = prediction

            table_clusters = [
                cluster
                for cluster in page.predictions.layout.clusters
                if cluster.label in {DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX}
            ]
            if not table_clusters:
                return prediction

            with TimeRecorder(conv_res, "table_structure"):
                for cluster in table_clusters:
                    crop_image = page.get_image(scale=self.scale, cropbox=cluster.bbox)
                    if crop_image is None:
                        continue

                    engine_output = self.engine.predict(
                        ObjectDetectionEngineInput(
                            image=crop_image,
                            metadata={"page_no": page.page_no, "cluster_id": cluster.id},
                        )
                    )

                    table = self._predictions_to_table(
                        page=page,
                        table_cluster=cluster,
                        crop_image=crop_image,
                        predictions=engine_output.predictions,
                    )
                    if table is not None:
                        prediction.table_map[cluster.id] = table

            return prediction

        if self.concurrency > 1:
            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                return list(executor.map(process_one_page, pages))
        return [process_one_page(page) for page in pages]

    def _predictions_to_table(
        self,
        page: Page,
        table_cluster: Cluster,
        crop_image: Image.Image,
        predictions: List[ObjectDetectionEnginePrediction],
    ) -> Optional[Table]:
        if not predictions:
            return None

        page_bbox = table_cluster.bbox
        table_width = page_bbox.width
        table_height = page_bbox.height
        image_width, image_height = crop_image.size

        rows: list[BoundingBox] = []
        cols: list[BoundingBox] = []
        cells: list[TableCell] = []

        for pred in predictions:
            if pred.score < self.SCORE_THRESHOLD:
                continue

            doc_bbox = self._image_bbox_to_page_bbox(
                pred.bbox,
                page_bbox,
                image_width,
                image_height,
            )

            cls_id = pred.label_id
            if cls_id == 0:
                table_bbox = doc_bbox
                page_bbox = table_bbox
            elif cls_id == 1:
                rows.append(doc_bbox)
            elif cls_id == 2:
                cols.append(doc_bbox)
            elif cls_id in {3, 4, 5, 6}:
                cells.append(
                    TableCell(
                        bbox=doc_bbox,
                        row_span=-1,
                        col_span=-1,
                        start_row_offset_idx=-1,
                        end_row_offset_idx=-1,
                        start_col_offset_idx=-1,
                        end_col_offset_idx=-1,
                        text="",
                        column_header=cls_id == 3,
                        row_header=cls_id == 4,
                        row_section=cls_id == 5,
                        fillable=False,
                    )
                )

        if not rows or not cols or not cells:
            return None

        rows.sort(key=lambda bbox: bbox.t)
        cols.sort(key=lambda bbox: bbox.l)

        self._assign_spans(cells, rows, axis="row")
        self._assign_spans(cells, cols, axis="col")

        table_data = TableData(table_cells=cells, num_rows=len(rows), num_cols=len(cols))
        table = Table(
            otsl_seq=[],
            table_cells=table_data.table_cells,
            num_rows=table_data.num_rows,
            num_cols=table_data.num_cols,
            id=table_cluster.id,
            page_no=page.page_no,
            cluster=table_cluster,
            label=table_cluster.label,
        )

        if self.options.do_cell_matching:
            text_cells = self._get_text_cells_for_table(table_cluster, page)
            self._assign_text_to_table_cells(table, text_cells)

        return table

    def _assign_spans(
        self,
        cells: List[TableCell],
        bounds: List[BoundingBox],
        axis: str,
    ) -> None:
        for idx, bound in enumerate(bounds):
            for cell in cells:
                if cell.bbox is None:
                    continue
                if cell.bbox.overlaps(bound):
                    if axis == "row":
                        if cell.start_row_offset_idx == -1:
                            cell.start_row_offset_idx = idx
                            cell.end_row_offset_idx = idx + 1
                            cell.row_span = 1
                        else:
                            cell.start_row_offset_idx = min(
                                cell.start_row_offset_idx, idx
                            )
                            cell.end_row_offset_idx = max(cell.end_row_offset_idx, idx + 1)
                            cell.row_span = cell.end_row_offset_idx - cell.start_row_offset_idx
                    else:
                        if cell.start_col_offset_idx == -1:
                            cell.start_col_offset_idx = idx
                            cell.end_col_offset_idx = idx + 1
                            cell.col_span = 1
                        else:
                            cell.start_col_offset_idx = min(
                                cell.start_col_offset_idx, idx
                            )
                            cell.end_col_offset_idx = max(cell.end_col_offset_idx, idx + 1)
                            cell.col_span = cell.end_col_offset_idx - cell.start_col_offset_idx

    def _image_bbox_to_page_bbox(
        self,
        image_bbox: List[float],
        table_bbox: BoundingBox,
        image_width: int,
        image_height: int,
    ) -> BoundingBox:
        table_width = table_bbox.width
        table_height = table_bbox.height

        left = table_bbox.l + (image_bbox[0] / image_width) * table_width
        right = table_bbox.l + (image_bbox[2] / image_width) * table_width
        top = table_bbox.t + (image_bbox[1] / image_height) * table_height
        bottom = table_bbox.t + (image_bbox[3] / image_height) * table_height

        return BoundingBox(
            l=left,
            t=top,
            r=right,
            b=bottom,
            coord_origin=CoordOrigin.TOPLEFT,
        )

    def _get_text_cells_for_table(
        self,
        table_cluster: Cluster,
        page: Page,
    ) -> List[TextCell]:
        assert page._backend is not None
        segmented_page = page._backend.get_segmented_page()
        if segmented_page is not None:
            cells = segmented_page.get_cells_in_bbox(
                cell_unit=TextCellUnit.WORD,
                bbox=table_cluster.bbox,
            )
            if cells:
                return [cell for cell in cells if cell.text.strip()]
        if table_cluster.cells:
            return [cell for cell in table_cluster.cells if cell.text.strip()]
        return []

    def _assign_text_to_table_cells(
        self,
        table: Table,
        text_cells: List[TextCell],
    ) -> None:
        for table_cell in table.table_cells:
            if table_cell.bbox is None:
                continue

            overlaps: List[str] = []
            for text_cell in text_cells:
                text_bbox = text_cell.rect.to_bounding_box()
                if table_cell.bbox.get_intersection_bbox(text_bbox) is None:
                    continue
                overlap = text_bbox.intersection_over_self(table_cell.bbox)
                if overlap > 0.5:
                    overlaps.append(text_cell.text.strip())

            if overlaps:
                table_cell.text = " ".join(overlaps)
