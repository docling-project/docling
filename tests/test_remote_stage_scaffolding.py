"""Tests for remote OCR, layout, table stages and service wrappers."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, Size
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import (
    Cluster,
    ConfidenceReport,
    LayoutPrediction,
    Page,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    KserveV2LayoutOptions,
    KserveV2OcrOptions,
    KserveV2TableStructureOptions,
    LayoutOptions,
    RapidOcrOptions,
    TableStructureOptions,
)
from docling.exceptions import OperationNotAllowed
from docling.models.stages.layout.kserve_v2_layout_model import KserveV2LayoutModel
from docling.models.stages.ocr.kserve_v2_ocr_model import KserveV2OcrModel
from docling.models.stages.table_structure import (
    kserve_v2_table_structure_model as remote_table_module,
)
from docling.models.stages.table_structure.kserve_v2_table_structure_model import (
    KserveV2TableStructureModel,
)
from docling.utils.kserve_v2_compat_server import (
    LayoutCompatModel,
    RapidOcrCompatModel,
    TableStructureCompatModel,
    build_standard_pipeline_compat_server,
)


class _FakeBackend:
    def __init__(self, width: int = 100, height: int = 80) -> None:
        self.width = width
        self.height = height
        self.segmented_cells = []

    def is_valid(self) -> bool:
        return True

    def get_page_image(
        self,
        scale: float = 1.0,
        cropbox: BoundingBox | None = None,
    ) -> Image.Image:
        if cropbox is None:
            width = int(round(self.width * scale))
            height = int(round(self.height * scale))
        else:
            width = max(1, int(round((cropbox.r - cropbox.l) * scale)))
            height = max(1, int(round((cropbox.b - cropbox.t) * scale)))
        return Image.new("RGB", (width, height), "white")

    def get_bitmap_rects(self, scale: float = 1.0):
        yield BoundingBox(
            l=0,
            t=0,
            r=self.width * scale,
            b=self.height * scale,
            coord_origin=CoordOrigin.TOPLEFT,
        )

    def get_segmented_page(self):
        if not self.segmented_cells:
            return None

        class _SegmentedPage:
            def __init__(self, cells):
                self._cells = cells

            def get_cells_in_bbox(self, cell_unit, bbox):
                _ = cell_unit, bbox
                return list(self._cells)

        return _SegmentedPage(self.segmented_cells)

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        return f"text@{bbox.l:.0f},{bbox.t:.0f}"

    def unload(self) -> None:
        return None


class _FakeRapidResult:
    def __init__(self) -> None:
        self.boxes = np.asarray(
            [
                [
                    [1.0, 2.0],
                    [11.0, 2.0],
                    [11.0, 8.0],
                    [1.0, 8.0],
                ]
            ],
            dtype=np.float32,
        )
        self.txts = ["WrappedText"]
        self.scores = [0.97]


class _FakeRapidOcrStage:
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        options: RapidOcrOptions,
        accelerator_options: AcceleratorOptions,
    ) -> None:
        self.enabled = enabled
        self.artifacts_path = artifacts_path
        self.options = options
        self.accelerator_options = accelerator_options

    def reader(self, image, use_det=None, use_cls=None, use_rec=None):
        _ = image, use_det, use_cls, use_rec
        return _FakeRapidResult()


class _FakeLayoutPredictor:
    def predict_batch(self, images):
        image = images[0]
        return [
            [
                {
                    "label": "table",
                    "confidence": 0.93,
                    "l": 5.0,
                    "t": 6.0,
                    "r": float(image.width - 5),
                    "b": float(image.height - 6),
                }
            ]
        ]


class _FakeLayoutStage:
    def __init__(
        self,
        artifacts_path: Path | None,
        accelerator_options: AcceleratorOptions,
        options: LayoutOptions,
    ) -> None:
        self.artifacts_path = artifacts_path
        self.accelerator_options = accelerator_options
        self.options = options
        self.layout_predictor = _FakeLayoutPredictor()


class _FakeTablePredictor:
    def __init__(self) -> None:
        self.last_page_input = None
        self.last_tbl_boxes = None
        self.last_do_matching = None

    def multi_table_predict(self, page_input, tbl_boxes, do_matching):
        self.last_page_input = page_input
        self.last_tbl_boxes = tbl_boxes
        self.last_do_matching = do_matching
        return [
            {
                "tf_responses": [
                    {
                        "bbox": {
                            "l": np.float32(0.0),
                            "t": np.float32(0.0),
                            "r": np.float32(page_input["width"]),
                            "b": np.float32(page_input["height"]),
                        },
                        "row_span": 1,
                        "col_span": 1,
                        "start_row_offset_idx": 0,
                        "end_row_offset_idx": 1,
                        "start_col_offset_idx": 0,
                        "end_col_offset_idx": 1,
                    }
                ],
                "predict_details": {
                    "num_rows": np.int64(1),
                    "num_cols": np.int64(1),
                    "prediction": {"rs_seq": ["fcel"]},
                },
            }
        ]


class _FakeTableStage:
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        options: TableStructureOptions,
        accelerator_options: AcceleratorOptions,
    ) -> None:
        self.enabled = enabled
        self.artifacts_path = artifacts_path
        self.options = options
        self.accelerator_options = accelerator_options
        self.tf_predictor = _FakeTablePredictor()


class RemoteStageScaffoldingTest(unittest.TestCase):
    def test_kserve_v2_ocr_requires_remote_enablement(self) -> None:
        options = KserveV2OcrOptions(
            url="http://localhost:8000",
            model_name="ocr_model",
            transport="http",
        )

        with self.assertRaises(OperationNotAllowed):
            KserveV2OcrModel(
                enabled=True,
                artifacts_path=None,
                options=options,
                accelerator_options=AcceleratorOptions(),
                enable_remote_services=False,
            )

    def test_kserve_v2_table_requires_remote_enablement(self) -> None:
        options = KserveV2TableStructureOptions(
            url="http://localhost:8000",
            model_name="table_model",
        )

        with self.assertRaises(OperationNotAllowed):
            KserveV2TableStructureModel(
                enabled=True,
                artifacts_path=None,
                options=options,
                accelerator_options=AcceleratorOptions(),
                enable_remote_services=False,
            )

    def test_kserve_v2_layout_requires_remote_enablement(self) -> None:
        options = KserveV2LayoutOptions(
            url="http://localhost:8000",
            model_name="layout_model",
        )

        with self.assertRaises(OperationNotAllowed):
            KserveV2LayoutModel(
                artifacts_path=None,
                accelerator_options=AcceleratorOptions(),
                options=options,
                enable_remote_services=False,
            )

    def test_kserve_v2_table_model_converts_remote_response(self) -> None:
        class _FakeClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def infer(self, *, inputs, output_names, request_parameters=None):
                _ = request_parameters
                self.assertEqual(output_names, ["response_json"])
                request_payload = inputs["request_json"].reshape(-1)[0]
                self.assertEqual(
                    json.loads(str(request_payload)),
                    {
                        "schema_version": 1,
                        "do_cell_matching": False,
                        "tokens": [],
                    },
                )
                return {
                    "response_json": np.asarray(
                        [
                            json.dumps(
                                {
                                    "otsl_seq": ["fcel"],
                                    "num_rows": 1,
                                    "num_cols": 1,
                                    "table_cells": [
                                        {
                                            "bbox": {
                                                "l": 0,
                                                "t": 0,
                                                "r": 20,
                                                "b": 10,
                                            },
                                            "row_span": 1,
                                            "col_span": 1,
                                            "start_row_offset_idx": 0,
                                            "end_row_offset_idx": 1,
                                            "start_col_offset_idx": 0,
                                            "end_col_offset_idx": 1,
                                        }
                                    ],
                                }
                            )
                        ],
                        dtype=object,
                    )
                }

            def close(self) -> None:
                return None

        fake_client_cls = _FakeClient
        fake_client_cls.assertEqual = self.assertEqual

        with patch.object(
            remote_table_module,
            "KserveV2HttpClient",
            fake_client_cls,
        ):
            options = KserveV2TableStructureOptions(
                url="http://localhost:8000",
                model_name="table_model",
                do_cell_matching=False,
            )
            model = KserveV2TableStructureModel(
                enabled=True,
                artifacts_path=None,
                options=options,
                accelerator_options=AcceleratorOptions(),
                enable_remote_services=True,
            )

        page = Page(page_no=1, size=Size(width=100, height=80))
        page._backend = _FakeBackend()
        page.predictions.layout = LayoutPrediction(
            clusters=[
                Cluster(
                    id=7,
                    label=DocItemLabel.TABLE,
                    bbox=BoundingBox(
                        l=10,
                        t=20,
                        r=30,
                        b=30,
                        coord_origin=CoordOrigin.TOPLEFT,
                    ),
                    cells=[],
                )
            ]
        )

        conv_res = ConversionResult.model_construct(
            input=SimpleNamespace(file=Path("dummy.pdf")),
            pages=[page],
            confidence=ConfidenceReport(),
            timings={},
        )

        predictions = model.predict_tables(conv_res, [page])

        self.assertEqual(len(predictions), 1)
        self.assertIn(7, predictions[0].table_map)

        table = predictions[0].table_map[7]
        self.assertEqual(table.otsl_seq, ["fcel"])
        self.assertEqual(table.num_rows, 1)
        self.assertEqual(table.num_cols, 1)
        self.assertEqual(len(table.table_cells), 1)

        cell = table.table_cells[0]
        self.assertEqual(cell.text, "text@10,20")
        self.assertIsNotNone(cell.bbox)
        assert cell.bbox is not None
        self.assertAlmostEqual(cell.bbox.l, 10.0)
        self.assertAlmostEqual(cell.bbox.t, 20.0)
        self.assertAlmostEqual(cell.bbox.r, 20.0)
        self.assertAlmostEqual(cell.bbox.b, 25.0)

    def test_remote_stage_chain_works_with_live_http_service(self) -> None:
        with build_standard_pipeline_compat_server(cell_text="RemoteCell") as server:
            page = Page(page_no=1, size=Size(width=100, height=80))
            page._backend = _FakeBackend()
            page.parsed_page = SimpleNamespace(
                textline_cells=[],
                word_cells=[],
                char_cells=[],
                has_lines=False,
                has_words=False,
                has_chars=False,
            )

            conv_res = ConversionResult.model_construct(
                input=SimpleNamespace(file=Path("dummy.png")),
                pages=[page],
                confidence=ConfidenceReport(),
                timings={},
            )

            ocr_model = KserveV2OcrModel(
                enabled=True,
                artifacts_path=None,
                options=KserveV2OcrOptions(
                    url=server.base_url,
                    model_name="rapidocr",
                    transport="http",
                    lang=["english"],
                    force_full_page_ocr=True,
                ),
                accelerator_options=AcceleratorOptions(),
                enable_remote_services=True,
            )
            layout_model = KserveV2LayoutModel(
                artifacts_path=None,
                accelerator_options=AcceleratorOptions(),
                options=KserveV2LayoutOptions(
                    url=server.base_url,
                    model_name="layout-heron",
                ),
                enable_remote_services=True,
            )
            table_model = KserveV2TableStructureModel(
                enabled=True,
                artifacts_path=None,
                options=KserveV2TableStructureOptions(
                    url=server.base_url,
                    model_name="table-structure",
                    do_cell_matching=True,
                ),
                accelerator_options=AcceleratorOptions(),
                enable_remote_services=True,
            )

            page = list(ocr_model(conv_res, [page]))[0]
            page._backend.segmented_cells = list(page.cells)
            page = list(layout_model(conv_res, [page]))[0]
            page = list(table_model(conv_res, [page]))[0]

        self.assertTrue(
            any(cell.from_ocr and cell.text == "RemoteCell" for cell in page.cells)
        )
        self.assertIsNotNone(page.predictions.layout)
        assert page.predictions.layout is not None
        self.assertEqual(len(page.predictions.layout.clusters), 1)
        self.assertIsNotNone(page.predictions.tablestructure)
        assert page.predictions.tablestructure is not None
        self.assertEqual(len(page.predictions.tablestructure.table_map), 1)

        table_model_impl = server.models["table-structure"]
        self.assertIsNotNone(table_model_impl.last_request)
        assert table_model_impl.last_request is not None
        self.assertTrue(table_model_impl.last_request["tokens"])
        self.assertEqual(table_model_impl.last_request["tokens"][0]["text"], "RemoteCell")


class RuntimeServiceWrapperTest(unittest.TestCase):
    def test_rapidocr_wrapper_exports_kserve_outputs(self) -> None:
        with patch(
            "docling.models.stages.ocr.rapid_ocr_model.RapidOcrModel",
            _FakeRapidOcrStage,
        ):
            model = RapidOcrCompatModel(
                accelerator_options=AcceleratorOptions(),
                options=RapidOcrOptions(lang=["english"]),
            )

        outputs = model.infer(
            inputs={"image": np.zeros((1, 24, 48, 3), dtype=np.uint8)},
            output_names=["boxes", "txts", "scores"],
        )

        self.assertEqual(outputs["boxes"].shape, (1, 4, 2))
        self.assertEqual(outputs["txts"].tolist(), ["WrappedText"])
        self.assertEqual(outputs["scores"].tolist(), [0.9700000286102295])

    def test_layout_wrapper_exports_kserve_outputs(self) -> None:
        with patch(
            "docling.models.stages.layout.layout_model.LayoutModel",
            _FakeLayoutStage,
        ):
            model = LayoutCompatModel(
                accelerator_options=AcceleratorOptions(),
                options=LayoutOptions(),
            )

        outputs = model.infer(
            inputs={"image": np.zeros((1, 30, 60, 3), dtype=np.uint8)},
            output_names=["label_names", "boxes", "scores"],
        )

        self.assertEqual(outputs["label_names"].tolist(), ["table"])
        self.assertEqual(outputs["boxes"].shape, (1, 4))
        self.assertEqual(outputs["scores"].tolist(), [0.9300000071525574])

    def test_table_wrapper_exports_remote_contract(self) -> None:
        with patch(
            "docling.models.stages.table_structure.table_structure_model.TableStructureModel",
            _FakeTableStage,
        ):
            model = TableStructureCompatModel(
                accelerator_options=AcceleratorOptions(),
                options=TableStructureOptions(do_cell_matching=True),
            )

        request_json = np.asarray(
            [
                json.dumps(
                    {
                        "schema_version": 1,
                        "do_cell_matching": True,
                        "tokens": [
                            {
                                "id": 0,
                                "text": "RemoteCell",
                                "bbox": {"l": 1, "t": 2, "r": 3, "b": 4},
                            }
                        ],
                    }
                )
            ],
            dtype=object,
        )
        outputs = model.infer(
            inputs={
                "image": np.zeros((1, 40, 80, 3), dtype=np.uint8),
                "request_json": request_json,
            },
            output_names=["response_json"],
        )

        response_payload = json.loads(outputs["response_json"].reshape(-1)[0])
        self.assertEqual(response_payload["otsl_seq"], ["fcel"])
        self.assertEqual(response_payload["num_rows"], 1)
        self.assertEqual(response_payload["num_cols"], 1)
        self.assertEqual(len(response_payload["table_cells"]), 1)
        self.assertEqual(model.last_request["tokens"][0]["text"], "RemoteCell")
        self.assertEqual(model.last_response["num_rows"], 1)


if __name__ == "__main__":
    unittest.main()
