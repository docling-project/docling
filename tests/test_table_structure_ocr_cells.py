# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, Mock, patch

import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel
from docling_core.types.doc.page import BoundingRectangle, SegmentedPdfPage, TextCell
from PIL import Image

from docling.datamodel.base_models import (
    Cluster,
    LayoutPrediction,
    Page,
    PagePredictions,
    Size,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    TableFormerMode,
    TableStructureOptions,
    TableStructureV2Options,
)
from docling.models.base_ocr_model import _empty_segmented_page
from docling.models.stages.table_structure.table_structure_model import (
    TableStructureModel,
)
from docling.models.stages.table_structure.table_structure_model_v2 import (
    TableStructureModelV2,
)


def _ocr_cell(index: int, bbox: BoundingBox, text: str) -> TextCell:
    return TextCell(
        index=index,
        rect=BoundingRectangle.from_bounding_box(bbox),
        text=text,
        orig=text,
        from_ocr=True,
    )


def _native_cell(index: int, bbox: BoundingBox, text: str) -> TextCell:
    return TextCell(
        index=index,
        rect=BoundingRectangle.from_bounding_box(bbox),
        text=text,
        orig=text,
        from_ocr=False,
    )


def _table_page(ocr_mode: bool = True) -> Page:
    page = Page(page_no=1)
    page.size = Size(width=500.0, height=500.0)

    # Mock backend
    backend = Mock()
    backend.is_valid.return_value = True
    backend.get_page_image.return_value = Image.new("RGB", (500, 500), color="white")

    backend_sp = _empty_segmented_page(page)
    backend_sp.word_cells = [
        _native_cell(0, BoundingBox(l=50, t=50, r=150, b=100), "CORRUPT_NATIVE_TEXT")
    ]
    backend_sp.has_words = True
    backend.get_segmented_page.return_value = backend_sp
    backend.get_text_in_rect.return_value = "CORRUPT_NATIVE_TEXT"
    page._backend = backend

    # Setup layout table cluster
    tbl_bbox = BoundingBox(l=50, t=50, r=400, b=300)
    if ocr_mode:
        cluster_cells = [
            _ocr_cell(0, BoundingBox(l=60, t=60, r=180, b=90), "OCR Cell 1"),
            _ocr_cell(1, BoundingBox(l=200, t=60, r=350, b=90), "OCR Cell 2"),
        ]
    else:
        cluster_cells = [
            _native_cell(0, BoundingBox(l=60, t=60, r=180, b=90), "Native Cell 1"),
            _native_cell(1, BoundingBox(l=200, t=60, r=350, b=90), "Native Cell 2"),
        ]

    cluster = Cluster(
        id=0,
        label=DocItemLabel.TABLE,
        bbox=tbl_bbox,
        confidence=0.95,
        cells=cluster_cells,
    )
    page.predictions = PagePredictions()
    page.predictions.layout = LayoutPrediction(clusters=[cluster])
    return page


def test_table_structure_model_v1_prioritizes_ocr_cells() -> None:
    page = _table_page(ocr_mode=True)
    assert page.predictions.layout is not None
    # parsed_page has no words (as in full-page OCR)
    sp = _empty_segmented_page(page)
    sp.textline_cells = page.predictions.layout.clusters[0].cells
    page.parsed_page = sp

    model = object.__new__(TableStructureModel)
    model.scale = 1.0
    model.do_cell_matching = True
    model.tf_predictor = Mock()
    model.tf_predictor.multi_table_predict.return_value = [
        {
            "tf_responses": [
                {
                    "bbox": {
                        "l": 60,
                        "t": 60,
                        "r": 180,
                        "b": 90,
                        "token": "OCR Cell 1",
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
                "num_rows": 1,
                "num_cols": 1,
                "prediction": {"rs_seq": ["fcel"]},
            },
        }
    ]

    with patch.object(model, "draw_table_and_cells", return_value=None):
        conv_res = cast(
            ConversionResult,
            SimpleNamespace(confidence=SimpleNamespace(pages={}), timings={}),
        )
        preds = model.predict_tables(conv_res, [page])

    assert len(preds) == 1
    tbl = preds[0].table_map[0]
    assert len(tbl.table_cells) == 1
    # Verify tokens passed to predictor came from OCR cells, not backend get_segmented_page
    called_page_input = model.tf_predictor.multi_table_predict.call_args[0][0]
    tokens = [tok["text"] for tok in called_page_input["tokens"]]
    assert "OCR Cell 1" in tokens
    assert "OCR Cell 2" in tokens
    assert "CORRUPT_NATIVE_TEXT" not in tokens
    # Backend get_segmented_page should NOT be called when parsed_page is available
    assert page._backend is not None
    cast(Any, page._backend.get_segmented_page).assert_not_called()


def test_table_structure_model_v1_no_matching_uses_ocr_cells() -> None:
    page = _table_page(ocr_mode=True)
    assert page.predictions.layout is not None
    sp = _empty_segmented_page(page)
    sp.textline_cells = page.predictions.layout.clusters[0].cells
    page.parsed_page = sp

    model = object.__new__(TableStructureModel)
    model.scale = 1.0
    model.do_cell_matching = False
    model.tf_predictor = Mock()
    model.tf_predictor.multi_table_predict.return_value = [
        {
            "tf_responses": [
                {
                    "bbox": {"l": 55, "t": 55, "r": 185, "b": 95},
                    "row_span": 1,
                    "col_span": 1,
                    "start_row_offset_idx": 0,
                    "end_row_offset_idx": 1,
                    "start_col_offset_idx": 0,
                    "end_col_offset_idx": 1,
                }
            ],
            "predict_details": {
                "num_rows": 1,
                "num_cols": 1,
                "prediction": {"rs_seq": ["fcel"]},
            },
        }
    ]

    with patch.object(model, "draw_table_and_cells", return_value=None):
        conv_res = cast(
            ConversionResult,
            SimpleNamespace(confidence=SimpleNamespace(pages={}), timings={}),
        )
        preds = model.predict_tables(conv_res, [page])

    assert len(preds) == 1
    tbl = preds[0].table_map[0]
    assert tbl.table_cells[0].text == "OCR Cell 1"
    # Backend get_text_in_rect should NOT be called when OCR cells exist
    assert page._backend is not None
    cast(Any, page._backend.get_text_in_rect).assert_not_called()


def test_table_structure_model_v2_no_fallback_to_backend_for_ocr() -> None:
    page = _table_page(ocr_mode=True)

    import torch

    model = object.__new__(TableStructureModelV2)
    model.scale = 1.0
    model.do_cell_matching = True
    model._cell_tokens = {"fcel", "ecel", "lcel", "ucel", "xcel"}
    model.model = Mock()
    model.tokenizer = Mock()
    model.transform = Mock(return_value=torch.zeros((3, 224, 224)))
    model.device = "cpu"
    model._decode_otsl_sequence = Mock(return_value=["fcel"])
    model.model.generate.return_value = {
        "generated_ids": [torch.tensor([1])],
        "predicted_bboxes": torch.tensor([[[0.1, 0.1, 0.4, 0.4]]]),
    }
    model._match_texts = Mock(return_value=[""])  # Simulates empty match in a cell

    with patch.object(model, "_build_table_cells") as mock_build:
        mock_build.return_value = (
            [
                {
                    "bbox": {"l": 60.0, "t": 60.0, "r": 180.0, "b": 90.0},
                    "row_span": 1,
                    "col_span": 1,
                    "start_row_offset_idx": 0,
                    "end_row_offset_idx": 1,
                    "start_col_offset_idx": 0,
                    "end_col_offset_idx": 1,
                }
            ],
            1,
            1,
        )
        model.predict_tables = TableStructureModelV2.predict_tables.__get__(model)

        with patch.object(model, "draw_table_and_cells", return_value=None):
            conv_res = cast(
                ConversionResult,
                SimpleNamespace(confidence=SimpleNamespace(pages={}), timings={}),
            )
            preds = model.predict_tables(conv_res, [page])

    assert len(preds) == 1
    # Backend get_text_in_rect should NOT be called to populate empty cell when OCR cells exist
    assert page._backend is not None
    cast(Any, page._backend.get_text_in_rect).assert_not_called()
