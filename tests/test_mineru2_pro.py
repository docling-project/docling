# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the MinerU2-Pro two-step VLM preset."""

import json
import sys
from types import ModuleType, SimpleNamespace

from docling_core.types.doc import DocItemLabel, Size
from PIL import Image

from docling.datamodel.base_models import Page
from docling.datamodel.pipeline_options import VlmConvertOptions
from docling.datamodel.pipeline_options_vlm_model import (
    ResponseFormat,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.datamodel.stage_model_specs import EngineModelConfig
from docling.datamodel.vlm_engine_options import AutoInlineVlmEngineOptions
from docling.models.inference_engines.vlm.base import VlmEngineOutput, VlmEngineType
from docling.models.inference_engines.vlm.mlx_engine import MlxVlmEngine
from docling.models.stages.vlm_convert.vlm_convert_model import VlmConvertModel
from docling.utils.mineru_utils import (
    MINERU2_LAYOUT_PROMPT,
    MinerU2Region,
    parse_mineru2,
    parse_mineru2_layout,
    prepare_mineru2_crops,
    serialize_mineru2_regions,
)


def test_mineru2_pro_preset_and_engine_configs() -> None:
    preset = VlmConvertOptions.get_preset("mineru2_pro")

    assert preset.name == "MinerU2.5-Pro"
    assert preset.default_engine_type == VlmEngineType.AUTO_INLINE
    assert "mineru2_pro" in VlmConvertOptions.list_preset_ids()

    spec = preset.model_spec
    assert spec.default_repo_id == "opendatalab/MinerU2.5-Pro-2604-1.2B"
    assert spec.prompt == MINERU2_LAYOUT_PROMPT
    assert spec.response_format == ResponseFormat.MINERU2
    assert spec.supported_engines == {
        VlmEngineType.TRANSFORMERS,
        VlmEngineType.MLX,
        VlmEngineType.API,
        VlmEngineType.API_OPENAI,
        VlmEngineType.API_LMSTUDIO,
    }

    transformers_config = spec.get_engine_config(VlmEngineType.TRANSFORMERS)
    assert transformers_config.torch_dtype == "bfloat16"
    assert transformers_config.min_engine_version == "4.56.0"
    assert (
        transformers_config.extra_config["transformers_model_type"]
        == TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT
    )
    assert (
        transformers_config.extra_config["transformers_prompt_style"]
        == TransformersPromptStyle.CHAT
    )

    mlx_config = spec.get_engine_config(VlmEngineType.MLX)
    assert mlx_config.repo_id == "carlesonielfa/MinerU2.5-Pro-2604-1.2B-mlx-bf16"
    assert mlx_config.extra_config["mlx_tied_word_embeddings"] is True

    assert spec.get_api_params(VlmEngineType.API_OPENAI) == {
        "model": "opendatalab/MinerU2.5-Pro-2604-1.2B",
        "max_tokens": 4096,
    }
    assert spec.get_api_params(VlmEngineType.API_LMSTUDIO) == {
        "model": "mineru2.5-pro-2604-1.2b",
        "max_tokens": 4096,
    }


def test_parse_mineru2_layout_filters_table_internal_regions() -> None:
    output = "".join(
        [
            "<|box_start|>100 100 900 700<|box_end|>"
            "<|ref_start|>table<|ref_end|><|rotate_up|>",
            "<|box_start|>200 200 300 300<|box_end|>"
            "<|ref_start|>text<|ref_end|><|rotate_up|>",
            "<|box_start|>100 750 800 900<|box_end|>"
            "<|ref_start|>text<|ref_end|><|rotate_right|><|txt_contd_tgt|>",
            "<|box_start|>0 0 0 100<|box_end|><|ref_start|>text<|ref_end|>",
        ]
    )

    regions = parse_mineru2_layout(output)

    assert len(regions) == 2
    assert regions[0] == MinerU2Region(type="table", bbox=(0.1, 0.1, 0.9, 0.7), angle=0)
    assert regions[1] == MinerU2Region(
        type="text",
        bbox=(0.1, 0.75, 0.8, 0.9),
        angle=90,
        merge_prev=True,
    )


def test_prepare_mineru2_crops_uses_type_specific_prompts() -> None:
    regions = [
        MinerU2Region(type="text", bbox=(0.0, 0.0, 0.5, 0.5), angle=90),
        MinerU2Region(type="table", bbox=(0.5, 0.0, 1.0, 0.5)),
        MinerU2Region(type="equation", bbox=(0.0, 0.5, 0.5, 1.0)),
        MinerU2Region(type="image", bbox=(0.5, 0.5, 1.0, 1.0)),
    ]

    crops = prepare_mineru2_crops(Image.new("RGB", (200, 100)), regions)

    assert [crop.region_index for crop in crops] == [0, 1, 2]
    assert [crop.prompt for crop in crops] == [
        "\nText Recognition:",
        "\nTable Recognition:",
        "\nFormula Recognition:",
    ]
    assert crops[0].image.size == (50, 100)


def test_parse_mineru2_builds_structured_document_and_otsl_table() -> None:
    regions = [
        MinerU2Region(type="doc_title", bbox=(0.1, 0.05, 0.9, 0.1), content="Report"),
        MinerU2Region(
            type="paragraph_title",
            bbox=(0.1, 0.15, 0.9, 0.2),
            content="Results",
        ),
        MinerU2Region(
            type="table",
            bbox=(0.1, 0.25, 0.9, 0.6),
            content=(
                "<ched>Name<ched>Value<nl><fcel>Merged<lcel><nl><fcel>Total<fcel>42<nl>"
            ),
        ),
        MinerU2Region(type="image", bbox=(0.1, 0.65, 0.4, 0.9)),
        MinerU2Region(
            type="ref_text", bbox=(0.45, 0.65, 0.9, 0.9), content="Reference"
        ),
    ]

    document = parse_mineru2(
        serialize_mineru2_regions(regions),
        original_page_size=Size(width=600, height=800),
        page_no=3,
        filename="report.pdf",
    )

    assert document.texts[0].label == DocItemLabel.TITLE
    assert document.texts[0].text == "Report"
    assert document.texts[1].label == DocItemLabel.SECTION_HEADER
    assert document.texts[1].text == "Results"
    assert document.texts[-1].label == DocItemLabel.REFERENCE
    assert document.texts[-1].prov[0].page_no == 3
    assert document.texts[-1].prov[0].bbox.l == 270
    assert len(document.pictures) == 1

    table = document.tables[0].data
    assert (table.num_rows, table.num_cols) == (3, 2)
    assert [cell.text for cell in table.table_cells] == [
        "Name",
        "Value",
        "Merged",
        "Total",
        "42",
    ]
    assert table.table_cells[2].col_span == 2


class _MinerU2Engine:
    def __init__(self) -> None:
        self.batches = []

    def predict_batch(self, batch):
        self.batches.append(batch)
        if len(self.batches) == 1:
            return [
                VlmEngineOutput(
                    text=(
                        "<|box_start|>0 0 500 500<|box_end|>"
                        "<|ref_start|>text<|ref_end|><|rotate_up|>"
                        "<|box_start|>500 0 1000 500<|box_end|>"
                        "<|ref_start|>table<|ref_end|><|rotate_up|>"
                        "<|box_start|>0 500 500 1000<|box_end|>"
                        "<|ref_start|>image<|ref_end|><|rotate_up|>"
                    ),
                    metadata={"num_tokens": 12, "generation_time": 0.2},
                )
            ]
        return [
            VlmEngineOutput(
                text="Body text", metadata={"num_tokens": 2, "generation_time": 0.1}
            ),
            VlmEngineOutput(
                text="<fcel>A<nl>",
                metadata={"num_tokens": 3, "generation_time": 0.1},
            ),
        ]

    def cleanup(self) -> None:
        return None


def test_vlm_convert_model_runs_mineru2_two_step_batches() -> None:
    model = VlmConvertModel.__new__(VlmConvertModel)
    model.enabled = True
    model.engine = _MinerU2Engine()
    model.options = VlmConvertOptions.from_preset(
        "mineru2_pro", engine_options=AutoInlineVlmEngineOptions()
    )

    image = Image.new("RGB", (200, 300), "white")
    page = Page(page_no=1)
    page._image_cache = {model.options.scale: image}
    page._default_image_scale = model.options.scale

    assert list(model(SimpleNamespace(timings={}), [page])) == [page]
    assert len(model.engine.batches) == 2
    assert model.engine.batches[0][0].prompt == MINERU2_LAYOUT_PROMPT
    assert model.engine.batches[0][0].image.size == (1036, 1036)
    assert [engine_input.prompt for engine_input in model.engine.batches[1]] == [
        "\nText Recognition:",
        "\nTable Recognition:",
    ]

    assert page.predictions.vlm_response is not None
    result = json.loads(page.predictions.vlm_response.text)
    assert [region["content"] for region in result] == [
        "Body text",
        "<fcel>A<nl>",
        None,
    ]
    assert page.predictions.vlm_response.num_tokens == 17


def test_mlx_tied_word_embeddings_uses_embedding_projection(
    monkeypatch, tmp_path
) -> None:
    repo_dir = tmp_path / "org--model"
    repo_dir.mkdir()
    language_model = SimpleNamespace(
        args=SimpleNamespace(tie_word_embeddings=False),
        lm_head=object(),
    )
    loaded_model = SimpleNamespace(language_model=language_model)
    load_calls = []

    def fake_load(path, *, strict=True):
        load_calls.append((path, strict))
        return loaded_model, object()

    fake_mlx_vlm = ModuleType("mlx_vlm")
    fake_mlx_vlm.load = fake_load  # type: ignore[attr-defined]
    fake_mlx_utils = ModuleType("mlx_vlm.utils")
    fake_mlx_utils.load_config = lambda path: {"model_type": "qwen2_vl"}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", fake_mlx_utils)

    engine = MlxVlmEngine.__new__(MlxVlmEngine)
    engine.artifacts_path = tmp_path
    engine.model_config = EngineModelConfig(
        repo_id="org/model",
        extra_config={"mlx_tied_word_embeddings": True},
    )

    engine._load_model_for_repo("org/model")

    assert load_calls == [(repo_dir, False)]
    assert language_model.args.tie_word_embeddings is True
    assert "lm_head" not in vars(language_model)
