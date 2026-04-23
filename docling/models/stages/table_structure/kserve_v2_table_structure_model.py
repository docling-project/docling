"""Remote table-structure stage backed by a KServe v2 HTTP endpoint."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
from docling_core.types.doc import BoundingBox, DocItemLabel, TableCell
from docling_core.types.doc.page import TextCellUnit
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import (
    Cluster,
    Page,
    Table,
    TableStructurePrediction,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.kserve_transport_utils import resolve_kserve_transport_base_url
from docling.datamodel.pipeline_options import KserveV2TableStructureOptions
from docling.datamodel.settings import settings
from docling.exceptions import OperationNotAllowed
from docling.models.base_table_model import BaseTableStructureModel
from docling.models.inference_engines.common import KserveV2HttpClient
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class _RemoteTableRequest(BaseModel):
    schema_version: int = 1
    do_cell_matching: bool
    tokens: list[dict[str, Any]] = Field(default_factory=list)


class _RemoteTableResponse(BaseModel):
    otsl_seq: list[str] = Field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
    table_cells: list[dict[str, Any]] = Field(default_factory=list)


class KserveV2TableStructureModel(BaseTableStructureModel):
    """Table-structure stage calling a remote HTTP service for inference.

    The remote endpoint receives:
    - `image`: a `(1, H, W, 3)` UINT8 RGB tensor containing the table crop
    - `request_json`: a `(1,)` BYTES tensor containing the JSON-serialized
      `_RemoteTableRequest`

    It must return:
    - `response_json`: a `(1,)` BYTES tensor containing the JSON-serialized
      `_RemoteTableResponse`

    Bounding boxes in the response are expected to be in crop-local pixels at
    the same scale as the input image. They are converted back to page
    coordinates by this stage.
    """

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: KserveV2TableStructureOptions,
        accelerator_options: AcceleratorOptions,
        enable_remote_services: bool = False,
    ):
        _ = artifacts_path, accelerator_options

        self.enabled = enabled
        self.options = options
        self._kserve_client: Optional[KserveV2HttpClient] = None

        if self.enabled and not enable_remote_services:
            raise OperationNotAllowed(
                "Connections to remote services are only allowed when set explicitly. "
                "pipeline_options.enable_remote_services=True."
            )

        if self.enabled:
            self._initialize_client()

    @classmethod
    def get_options_type(cls) -> type[KserveV2TableStructureOptions]:
        return KserveV2TableStructureOptions

    def _initialize_client(self) -> None:
        base_url = resolve_kserve_transport_base_url(
            url=self.options.url,
            transport=self.options.transport,
        )
        self._kserve_client = KserveV2HttpClient(
            base_url=base_url,
            model_name=self.options.model_name,
            model_version=self.options.model_version,
            timeout=self.options.timeout,
            headers=self.options.headers,
        )
        _log.info(
            "KServe v2 table structure client initialized: url=%s, model=%s",
            self.options.url,
            self.options.model_name,
        )

    def close(self) -> None:
        if self._kserve_client is None:
            return
        self._kserve_client.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return np.expand_dims(image_array, axis=0)

    def _build_tokens(
        self,
        page: Page,
        table_cluster: Cluster,
    ) -> list[dict[str, Any]]:
        assert page._backend is not None

        segmented_page = page._backend.get_segmented_page()
        if segmented_page is not None:
            token_cells = segmented_page.get_cells_in_bbox(
                cell_unit=TextCellUnit.WORD,
                bbox=table_cluster.bbox,
            )
            if len(token_cells) == 0:
                token_cells = table_cluster.cells
        else:
            token_cells = table_cluster.cells

        tokens: list[dict[str, Any]] = []
        for token_id, cell in enumerate(token_cells):
            if len(cell.text.strip()) == 0:
                continue

            cell_bbox = cell.rect.to_bounding_box()
            local_bbox = BoundingBox(
                l=(cell_bbox.l - table_cluster.bbox.l) * self.options.scale,
                t=(cell_bbox.t - table_cluster.bbox.t) * self.options.scale,
                r=(cell_bbox.r - table_cluster.bbox.l) * self.options.scale,
                b=(cell_bbox.b - table_cluster.bbox.t) * self.options.scale,
                coord_origin=cell_bbox.coord_origin,
            )
            tokens.append(
                {
                    "id": token_id,
                    "text": cell.text,
                    "bbox": local_bbox.model_dump(),
                }
            )

        return tokens

    def _build_request_payload(
        self,
        page: Page,
        table_cluster: Cluster,
    ) -> np.ndarray:
        request = _RemoteTableRequest(
            do_cell_matching=self.options.do_cell_matching,
            tokens=self._build_tokens(page, table_cluster),
        )
        return np.asarray([request.model_dump_json()], dtype=object)

    def _decode_response_payload(
        self, outputs: dict[str, np.ndarray]
    ) -> _RemoteTableResponse:
        try:
            payload = outputs[self.options.response_output_name]
        except KeyError as exc:
            raise RuntimeError(
                "Missing expected KServe v2 table output "
                f"{self.options.response_output_name!r}."
            ) from exc

        if payload.size == 0:
            raise RuntimeError("Remote table service returned an empty response tensor.")

        raw_payload = payload.reshape(-1)[0]
        if isinstance(raw_payload, bytes):
            payload_json = raw_payload.decode("utf-8")
        else:
            payload_json = str(raw_payload)

        return _RemoteTableResponse.model_validate_json(payload_json)

    def _crop_bbox_to_page_bbox(
        self,
        bbox: BoundingBox,
        table_cluster: Cluster,
    ) -> BoundingBox:
        return BoundingBox(
            l=(bbox.l / self.options.scale) + table_cluster.bbox.l,
            t=(bbox.t / self.options.scale) + table_cluster.bbox.t,
            r=(bbox.r / self.options.scale) + table_cluster.bbox.l,
            b=(bbox.b / self.options.scale) + table_cluster.bbox.t,
            coord_origin=bbox.coord_origin,
        )

    def _build_table_cells(
        self,
        page: Page,
        table_cluster: Cluster,
        response: _RemoteTableResponse,
    ) -> list[TableCell]:
        assert page._backend is not None

        table_cells: list[TableCell] = []
        for raw_cell in response.table_cells:
            element = dict(raw_cell)
            raw_bbox = element.get("bbox")

            if raw_bbox is not None:
                bbox_token = raw_bbox.get("token")
                bbox = BoundingBox.model_validate(
                    {key: value for key, value in raw_bbox.items() if key != "token"}
                )
                page_bbox = self._crop_bbox_to_page_bbox(bbox, table_cluster)
                page_bbox_payload = page_bbox.model_dump()

                if bbox_token is None or len(str(bbox_token).strip()) == 0:
                    bbox_token = page._backend.get_text_in_rect(page_bbox)
                page_bbox_payload["token"] = bbox_token
                element["bbox"] = page_bbox_payload

            table_cells.append(TableCell.model_validate(element))

        return table_cells

    def draw_table_and_cells(
        self,
        conv_res: ConversionResult,
        page: Page,
        tbl_list: Iterable[Table],
        show: bool = False,
    ) -> None:
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
                if tc.bbox is None:
                    continue
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
            return

        out_path: Path = (
            Path(settings.debug.debug_output_path) / f"debug_{conv_res.input.file.stem}"
        )
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / f"table_struct_remote_page_{page.page_no:05}.png"
        image.save(str(out_file), format="png")

    def predict_tables(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[TableStructurePrediction]:
        pages = list(pages)
        predictions: list[TableStructurePrediction] = []

        if not self.enabled or self._kserve_client is None:
            for page in pages:
                existing_prediction = (
                    page.predictions.tablestructure or TableStructurePrediction()
                )
                page.predictions.tablestructure = existing_prediction
                predictions.append(existing_prediction)
            return predictions

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

                table_prediction = TableStructurePrediction()
                page.predictions.tablestructure = table_prediction

                in_tables = [
                    cluster
                    for cluster in page.predictions.layout.clusters
                    if cluster.label
                    in [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]
                ]

                for table_cluster in in_tables:
                    table_image = page.get_image(
                        scale=self.options.scale,
                        cropbox=table_cluster.bbox,
                    )
                    if table_image is None:
                        continue

                    outputs = self._kserve_client.infer(
                        inputs={
                            self.options.image_input_name: self._preprocess_image(
                                table_image
                            ),
                            self.options.request_input_name: self._build_request_payload(
                                page, table_cluster
                            ),
                        },
                        output_names=[self.options.response_output_name],
                        request_parameters=self.options.request_parameters,
                    )

                    response = self._decode_response_payload(outputs)
                    table_cells = self._build_table_cells(
                        page=page,
                        table_cluster=table_cluster,
                        response=response,
                    )

                    table_prediction.table_map[table_cluster.id] = Table(
                        otsl_seq=response.otsl_seq,
                        table_cells=table_cells,
                        num_rows=response.num_rows,
                        num_cols=response.num_cols,
                        id=table_cluster.id,
                        page_no=page.page_no,
                        cluster=table_cluster,
                        label=table_cluster.label,
                    )

                if settings.debug.visualize_tables:
                    self.draw_table_and_cells(
                        conv_res,
                        page,
                        page.predictions.tablestructure.table_map.values(),
                    )

                predictions.append(table_prediction)

        return predictions
