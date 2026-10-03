# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Regression tests for chart extraction presets and CSV table semantics."""

import sys
from types import ModuleType

import pandas as pd
import pytest
from docling_core.types.doc import (
    DescriptionMetaField,
    DoclingDocument,
    PictureClassificationMetaField,
    PictureMeta,
    TableData,
    TabularChartMetaField,
)
from docling_core.types.doc.document import PictureClassificationPrediction
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.datamodel.chart_extraction_options import ChartExtractionVlmEngineOptions
from docling.datamodel.vlm_engine_options import (
    AutoInlineVlmEngineOptions,
    MlxVlmEngineOptions,
)
from docling.models.inference_engines.vlm.auto_inline_engine import (
    AutoInlineVlmEngine,
)
from docling.models.inference_engines.vlm.base import VlmEngineType
from docling.models.stages.chart_extraction.granite_vision import (
    ChartExtractionVlmEngineModel,
    _dataframe_to_tabledata,
    _extract_csv_to_dataframe,
)


def test_granite_vision_v4_mlx_preset_uses_official_model() -> None:
    mlx_options = ChartExtractionVlmEngineOptions.from_preset("granite_vision_v4_mlx")
    default_options = ChartExtractionVlmEngineOptions.from_preset("granite_vision_v4")

    assert isinstance(mlx_options.engine_options, MlxVlmEngineOptions)
    assert isinstance(default_options.engine_options, AutoInlineVlmEngineOptions)
    assert mlx_options.model_spec.is_engine_supported(VlmEngineType.MLX)
    assert mlx_options.model_spec.get_engine_config(VlmEngineType.MLX).repo_id == (
        "ibm-granite/granite-vision-4.1-4b"
    )
    assert mlx_options.model_spec.get_engine_config(VlmEngineType.MLX).revision == (
        default_options.model_spec.revision
    )
    assert mlx_options.output_format == default_options.output_format
    assert (
        default_options.model_spec.get_engine_config(
            VlmEngineType.MLX
        ).min_engine_version
        == "0.7.0"
    )
    assert "granite_vision_v4_mlx" in ChartExtractionVlmEngineOptions.list_preset_ids()


@pytest.mark.parametrize(
    ("system", "device", "mlx_version_ok", "expected_engine"),
    [
        ("Darwin", "mps", True, VlmEngineType.MLX),
        ("Darwin", "mps", False, VlmEngineType.TRANSFORMERS),
        ("Darwin", "cpu", True, VlmEngineType.TRANSFORMERS),
        ("Linux", "cpu", True, VlmEngineType.TRANSFORMERS),
    ],
)
def test_granite_vision_v4_auto_selects_local_engine(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    device: str,
    mlx_version_ok: bool,
    expected_engine: VlmEngineType,
) -> None:
    options = ChartExtractionVlmEngineOptions.from_preset("granite_vision_v4")
    assert isinstance(options.engine_options, AutoInlineVlmEngineOptions)
    engine = AutoInlineVlmEngine(
        options=options.engine_options,
        accelerator_options=AcceleratorOptions(),
        artifacts_path=None,
    )
    engine.model_spec = options.model_spec

    monkeypatch.setattr("platform.system", lambda: system)
    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.auto_inline_engine.decide_device",
        lambda *args, **kwargs: device,
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm", ModuleType("mlx_vlm"))

    def version_satisfied(engine_type: VlmEngineType, min_version: str | None) -> bool:
        assert engine_type == VlmEngineType.MLX
        assert min_version == "0.7.0"
        return mlx_version_ok

    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.auto_inline_engine.engine_version_satisfied",
        version_satisfied,
    )

    assert engine._select_engine() == expected_engine


@pytest.mark.parametrize(
    ("csv_text", "header_count", "row_header_coords"),
    [
        ("Category,2020,2021\nNorth,85,15\nSouth,10,60", 3, {(1, 0), (2, 0)}),
        ("Category,1,2,3\nInstitutional,85,15,0", 4, {(1, 0)}),
        ("2020,2021,2022\n85,15,0\n10,60,30", 3, set()),
        ("1,2,3\n10,3,4", 3, set()),
        ("1,2,3\n4,5,6", 0, set()),
        ("1,2,3", 0, set()),
        ("2020,2021,2022\nNorth,12,13", 3, {(1, 0)}),
        ("1,2.5,3\n4,5,6", 0, set()),
    ],
)
def test_chart_csv_table_header_classification(
    csv_text: str, header_count: int, row_header_coords: set[tuple[int, int]]
) -> None:
    table = _dataframe_to_tabledata(_extract_csv_to_dataframe(csv_text))

    assert len(table.table_cells) == table.num_rows * table.num_cols
    assert table.num_rows == csv_text.count("\n") + 1
    assert sum(cell.column_header for cell in table.table_cells) == header_count
    assert {
        (cell.start_row_offset_idx, cell.start_col_offset_idx)
        for cell in table.table_cells
        if cell.row_header
    } == row_header_coords
    assert all(
        cell.start_row_offset_idx == 0
        for cell in table.table_cells
        if cell.column_header
    )


@pytest.mark.parametrize(
    ("csv_text", "header_texts"),
    [
        (",2020,2021\nNorth,1,2\nSouth,3,4", ["", "2020", "2021"]),
        ("1,,3\n4,5,6\n7,8,9", []),
    ],
)
def test_blank_cells_in_first_row(csv_text: str, header_texts: list[str]) -> None:
    table = _dataframe_to_tabledata(_extract_csv_to_dataframe(csv_text))

    assert [c.text for c in table.table_cells if c.column_header] == header_texts


def test_text_in_data_columns_is_not_a_row_header() -> None:
    csv_text = "Category,Year,Value\nNorth,2024,unknown\nSouth,2025,3"
    table = _dataframe_to_tabledata(_extract_csv_to_dataframe(csv_text))
    cells = {
        (cell.start_row_offset_idx, cell.start_col_offset_idx): cell
        for cell in table.table_cells
    }

    assert cells[1, 0].row_header is True
    assert cells[1, 2].text == "unknown"
    assert cells[1, 2].row_header is False
    assert cells[2, 0].row_header is True


def test_blank_first_column_cells_are_not_row_headers() -> None:
    df = pd.DataFrame([["Category", "Value"], [None, "unknown"], ["   ", "other"]])
    table = _dataframe_to_tabledata(df)
    cells = {
        (cell.start_row_offset_idx, cell.start_col_offset_idx): cell
        for cell in table.table_cells
    }

    assert cells[0, 0].column_header is True
    assert cells[1, 0].text == ""
    assert cells[1, 0].row_header is False
    assert cells[2, 0].text == "   "
    assert cells[2, 0].row_header is False
    assert cells[1, 1].row_header is False


def test_empty_chart_table_has_no_headers() -> None:
    table = _dataframe_to_tabledata(pd.DataFrame())

    assert table.num_rows == 0
    assert table.num_cols == 0
    assert table.table_cells == []


def test_chart_enrichment_runs_only_missing_outputs_after_classification() -> None:
    class Engine:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def predict_batch(self, inputs):
            self.prompts = [item.prompt for item in inputs]
            return [type("Output", (), {"text": "```python\npass\n```"})()]

        def cleanup(self) -> None:
            pass

    options = ChartExtractionVlmEngineOptions.from_preset("granite_vision_v4")
    options.chart2summary = True
    options.chart2code = True
    model = ChartExtractionVlmEngineModel.__new__(ChartExtractionVlmEngineModel)
    model.enabled = True
    model.options = options
    engine = Engine()
    model.engine = engine

    chart_data = TabularChartMetaField(
        chart_data=TableData(num_rows=0, num_cols=0, table_cells=[])
    )
    description = DescriptionMetaField(text="Provided by VLM")
    doc = DoclingDocument(name="chart")
    picture = doc.add_picture()
    picture.meta = PictureMeta(tabular_chart=chart_data, description=description)
    assert not model.is_processable(doc, picture)

    picture.meta.classification = PictureClassificationMetaField(
        predictions=[PictureClassificationPrediction(class_name="bar_chart")]
    )
    assert model.is_processable(doc, picture)

    image = Image.new("RGB", (10, 10), "white")
    result = list(
        model(
            doc,
            [ItemAndImageEnrichmentElement(item=picture, image=image)],
        )
    )

    assert result == [picture]
    assert engine.prompts == ["<chart2code>"]
    assert picture.meta.tabular_chart is chart_data
    assert picture.meta.description is description
    assert picture.meta.code is not None


def test_legacy_chart_preset_uses_its_csv_prompt_and_parser() -> None:
    class Engine:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def predict_batch(self, inputs):
            self.prompts = [item.prompt for item in inputs]
            return [type("Output", (), {"text": "Category,Value\nNorth,10"})()]

        def cleanup(self) -> None:
            pass

    options = ChartExtractionVlmEngineOptions.from_preset("granite_vision")
    assert (
        options.model_spec.default_repo_id
        == "ibm-granite/granite-vision-3.3-2b-chart2csv-preview"
    )
    with pytest.raises(ValueError, match="supports CSV output only"):
        ChartExtractionVlmEngineOptions.from_preset(
            "granite_vision", chart2summary=True
        )

    model = ChartExtractionVlmEngineModel.__new__(ChartExtractionVlmEngineModel)
    model.enabled = True
    model.options = options
    engine = Engine()
    model.engine = engine

    doc = DoclingDocument(name="chart")
    picture = doc.add_picture()
    picture.meta = PictureMeta(
        classification=PictureClassificationMetaField(
            predictions=[PictureClassificationPrediction(class_name="bar_chart")]
        )
    )
    assert model.is_processable(doc, picture)

    result = list(
        model(
            doc,
            [
                ItemAndImageEnrichmentElement(
                    item=picture, image=Image.new("RGB", (10, 10), "white")
                )
            ],
        )
    )

    assert result == [picture]
    assert engine.prompts == [options.model_spec.prompt]
    assert picture.meta.tabular_chart is not None
    assert picture.meta.tabular_chart.chart_data.num_rows == 2
