"""End-to-end remote standard pipeline test covering OCR, layout, and tables."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from docling_core.types.doc import BoundingBox, CoordOrigin, Size

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import AssembledUnit, ConfidenceReport, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    KserveV2LayoutOptions,
    KserveV2OcrOptions,
    KserveV2TableStructureOptions,
)
from docling.models.stages.layout.kserve_v2_layout_model import KserveV2LayoutModel
from docling.models.stages.ocr.kserve_v2_ocr_model import KserveV2OcrModel
from docling.models.stages.page_assemble.page_assemble_model import (
    PageAssembleModel,
    PageAssembleOptions,
)
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)
from docling.models.stages.table_structure.kserve_v2_table_structure_model import (
    KserveV2TableStructureModel,
)
from docling.utils.kserve_v2_compat_server import build_standard_pipeline_compat_server


class _FakeBackend:
    def __init__(self, width: int = 360, height: int = 180) -> None:
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


class RemoteStandardPipelineE2ETest(unittest.TestCase):
    def test_remote_standard_pipeline_end_to_end(self) -> None:
        page = Page(page_no=1, size=Size(width=360, height=180))
        page._backend = _FakeBackend()
        page.parsed_page = SimpleNamespace(
            textline_cells=[],
            word_cells=[],
            char_cells=[],
            has_lines=False,
            has_words=False,
            has_chars=False,
            hyperlinks=[],
        )

        conv_res = ConversionResult.model_construct(
            input=SimpleNamespace(
                file=Path("remote-standard.png"),
                document_hash="0" * 64,
            ),
            pages=[page],
            confidence=ConfidenceReport(),
            timings={},
            assembled=AssembledUnit(),
        )

        with build_standard_pipeline_compat_server(cell_text="RemoteCell") as server:
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
            assemble_model = PageAssembleModel(options=PageAssembleOptions())
            reading_order_model = ReadingOrderModel(options=ReadingOrderOptions())

            page = list(ocr_model(conv_res, [page]))[0]
            page._backend.segmented_cells = list(page.cells)
            page = list(layout_model(conv_res, [page]))[0]
            page = list(table_model(conv_res, [page]))[0]
            page = list(assemble_model(conv_res, [page]))[0]

            conv_res.pages = [page]
            conv_res.assembled = page.assembled
            conv_res.document = reading_order_model(conv_res)

        ocr_service = server.models["rapidocr"]
        layout_service = server.models["layout-heron"]
        table_service = server.models["table-structure"]
        self.assertGreaterEqual(ocr_service.infer_calls, 1)
        self.assertGreaterEqual(layout_service.infer_calls, 1)
        self.assertGreaterEqual(table_service.infer_calls, 1)

        self.assertTrue(
            any(cell.from_ocr and cell.text == "RemoteCell" for cell in page.cells)
        )
        self.assertIsNotNone(page.predictions.layout)
        self.assertIsNotNone(page.predictions.tablestructure)
        self.assertIsNotNone(page.assembled)

        assert page.predictions.layout is not None
        assert page.predictions.tablestructure is not None
        assert page.assembled is not None
        self.assertEqual(len(page.predictions.layout.clusters), 1)
        self.assertEqual(len(page.predictions.tablestructure.table_map), 1)

        self.assertIsNotNone(table_service.last_request)
        assert table_service.last_request is not None
        self.assertEqual(table_service.last_request["tokens"][0]["text"], "RemoteCell")

        self.assertEqual(len(conv_res.document.tables), 1)
        table_item = conv_res.document.tables[0]
        table_df = table_item.export_to_dataframe(doc=conv_res.document)
        self.assertEqual(table_df.shape, (1, 1))
        self.assertEqual(table_df.iloc[0, 0], "RemoteCell")


if __name__ == "__main__":
    unittest.main()
