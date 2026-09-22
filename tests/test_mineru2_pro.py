# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the MinerU2-Pro two-step VLM preset."""

import json
import sys
from pathlib import PurePath
from types import ModuleType, SimpleNamespace

import pytest
import torch
from docling_core.types.doc import DocItemLabel, Size
from PIL import Image

from docling.datamodel.base_models import (
    ConversionStatus,
    FailureCategory,
    Page,
    PagePredictions,
    VlmPrediction,
)
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
from docling.models.inference_engines.vlm.transformers_engine import (
    _FrequencyPresencePenaltyLogitsProcessor,
)
from docling.models.stages.vlm_convert.vlm_convert_model import VlmConvertModel
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.utils.mineru_utils import (
    MINERU2_EQUATION_GENERATION_CONFIG,
    MINERU2_LAYOUT_GENERATION_CONFIG,
    MINERU2_LAYOUT_PROMPT,
    MINERU2_TABLE_GENERATION_CONFIG,
    MINERU2_TEXT_GENERATION_CONFIG,
    MinerU2Region,
    parse_mineru2,
    parse_mineru2_layout,
    prepare_mineru2_crops,
    serialize_mineru2_transcript,
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
    assert spec.supported_engines == {VlmEngineType.TRANSFORMERS}

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


def test_mineru2_transcript_preserves_native_outputs() -> None:
    layout = "\n<|box_start|>0 0 1 1<|box_end|>\n"
    recognition = [(7, " leading\n中文 <tag> trailing ")]

    transcript = json.loads(serialize_mineru2_transcript(layout, recognition))

    assert transcript["layout"] == layout
    assert transcript["recognition"] == [{"region_index": 7, "text": recognition[0][1]}]


def test_parse_mineru2_builds_structured_document_and_otsl_table() -> None:
    layout = "".join(
        [
            "<|box_start|>100 50 900 100<|box_end|><|ref_start|>doc_title<|ref_end|>",
            "<|box_start|>100 150 900 200<|box_end|><|ref_start|>paragraph_title<|ref_end|>",
            "<|box_start|>100 250 900 600<|box_end|><|ref_start|>table<|ref_end|>",
            "<|box_start|>100 650 400 900<|box_end|><|ref_start|>image<|ref_end|>",
            "<|box_start|>450 650 900 900<|box_end|><|ref_start|>ref_text<|ref_end|>",
        ]
    )
    transcript = serialize_mineru2_transcript(
        layout,
        [
            (0, "Report"),
            (1, "Results"),
            (
                2,
                "<ched>Name<ched>Value<nl><fcel>Merged<lcel><nl>"
                "<fcel>Total<fcel>42<nl>",
            ),
            (4, "Reference"),
        ],
    )

    document = parse_mineru2(
        transcript,
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


def test_parse_mineru2_merges_text_continuations() -> None:
    layout = "".join(
        [
            "<|box_start|>0 0 1000 200<|box_end|><|ref_start|>text<|ref_end|><|txt_contd_tgt|>",
            "<|box_start|>0 200 1000 400<|box_end|><|ref_start|>text<|ref_end|>",
            "<|box_start|>0 400 1000 600<|box_end|><|ref_start|>text<|ref_end|><|txt_contd_tgt|>",
            "<|box_start|>0 600 1000 800<|box_end|><|ref_start|>text<|ref_end|><|txt_contd_tgt|>",
        ]
    )
    document = parse_mineru2(
        serialize_mineru2_transcript(
            layout,
            [(0, "Fallback"), (1, "Hello"), (2, "world"), (3, "中文")],
        ),
        original_page_size=Size(width=100, height=100),
        page_no=1,
    )

    assert [item.text for item in document.texts] == ["Fallback", "Hello world中文"]
    assert len(document.texts[1].prov) == 3
    assert [provenance.bbox.t for provenance in document.texts[1].prov] == [
        20,
        40,
        60,
    ]


def test_parse_mineru2_index_is_text() -> None:
    layout = "<|box_start|>0 0 1000 1000<|box_end|><|ref_start|>index<|ref_end|>"

    document = parse_mineru2(
        serialize_mineru2_transcript(layout, [(0, "Contents")]),
        original_page_size=Size(width=100, height=100),
        page_no=1,
    )

    assert [(item.label, item.text) for item in document.texts] == [
        (DocItemLabel.TEXT, "Contents")
    ]


class _MinerU2Engine:
    def __init__(self) -> None:
        self.batches = []
        self.layout_outputs = [
            (
                "<|box_start|>0 0 500 500<|box_end|>"
                "<|ref_start|>text<|ref_end|><|rotate_up|>"
                "<|box_start|>500 0 1000 500<|box_end|>"
                "<|ref_start|>table<|ref_end|><|rotate_up|>"
                "<|box_start|>0 500 500 1000<|box_end|>"
                "<|ref_start|>equation<|ref_end|><|rotate_up|>"
                "<|box_start|>500 500 1000 1000<|box_end|>"
                "<|ref_start|>text<|ref_end|><|rotate_up|>"
            ),
            (
                "<|box_start|>0 0 1000 1000<|box_end|>"
                "<|ref_start|>text<|ref_end|><|rotate_up|>"
            ),
        ]

    def predict_batch(self, batch):
        self.batches.append(batch)
        if batch[0].prompt == MINERU2_LAYOUT_PROMPT:
            return [
                VlmEngineOutput(
                    text=self.layout_outputs[index],
                    metadata={"num_tokens": 12, "generation_time": 0.2},
                )
                if index == 0
                else VlmEngineOutput(
                    text=self.layout_outputs[index],
                    metadata={"num_tokens": 8, "generation_time": 0.2},
                )
                for index, _input in enumerate(batch)
            ]
        return [
            VlmEngineOutput(
                text=(
                    f"Text {engine_input.image.getpixel((0, 0))[0]}"
                    if engine_input.prompt == "\nText Recognition:"
                    else f"<fcel>T{engine_input.image.getpixel((0, 0))[0]}<nl>"
                    if engine_input.prompt == "\nTable Recognition:"
                    else f"E{engine_input.image.getpixel((0, 0))[0]}"
                ),
                metadata={"num_tokens": 1, "generation_time": 0.1},
            )
            for engine_input in batch
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

    pages = [Page(page_no=1), Page(page_no=2)]
    for page, red in zip(pages, [10, 20]):
        page._image_cache = {
            model.options.scale: Image.new("RGB", (200, 300), (red, 0, 0))
        }
        page._default_image_scale = model.options.scale

    assert list(model(SimpleNamespace(timings={}), pages)) == pages
    assert len(model.engine.batches) == 5
    assert [engine_input.prompt for engine_input in model.engine.batches[0]] == [
        MINERU2_LAYOUT_PROMPT,
        MINERU2_LAYOUT_PROMPT,
    ]
    assert model.engine.batches[0][0].image.size == (1036, 1036)
    assert [engine_input.prompt for engine_input in model.engine.batches[1]] == [
        "\nText Recognition:",
        "\nText Recognition:",
    ]
    assert model.engine.batches[2][0].prompt == "\nText Recognition:"
    assert model.engine.batches[3][0].prompt == "\nTable Recognition:"
    assert model.engine.batches[4][0].prompt == "\nFormula Recognition:"
    assert all(len(batch) <= len(pages) for batch in model.engine.batches[1:])
    assert model.engine.batches[0][0].extra_generation_config == (
        MINERU2_LAYOUT_GENERATION_CONFIG
    )
    assert model.engine.batches[1][0].extra_generation_config == (
        MINERU2_TEXT_GENERATION_CONFIG
    )
    assert model.engine.batches[3][0].extra_generation_config == (
        MINERU2_TABLE_GENERATION_CONFIG
    )
    assert model.engine.batches[4][0].extra_generation_config == (
        MINERU2_EQUATION_GENERATION_CONFIG
    )

    first_response = pages[0].predictions.vlm_response
    second_response = pages[1].predictions.vlm_response
    assert first_response is not None
    assert second_response is not None
    first_transcript = json.loads(first_response.text)
    second_transcript = json.loads(second_response.text)
    assert set(first_transcript) == {"version", "layout", "recognition"}
    assert first_transcript["version"] == 1
    assert first_transcript["layout"] == model.engine.layout_outputs[0]
    assert second_transcript["layout"] == model.engine.layout_outputs[1]
    assert first_transcript["recognition"] == [
        {"region_index": 0, "text": "Text 10"},
        {"region_index": 1, "text": "<fcel>T10<nl>"},
        {"region_index": 2, "text": "E10"},
        {"region_index": 3, "text": "Text 10"},
    ]
    assert second_transcript["recognition"] == [{"region_index": 0, "text": "Text 20"}]
    assert first_response.num_tokens == 16
    assert second_response.num_tokens == 9


def test_transformers_applies_mineru_recognition_penalties() -> None:
    processor = _FrequencyPresencePenaltyLogitsProcessor(
        prompt_length=1, presence_penalty=1.0, frequency_penalty=0.05
    )
    scores = processor(
        torch.tensor([[3, 7, 7, 9]]),
        torch.zeros((1, 12)),
    )

    assert scores[0, 3] == 0
    assert scores[0, 7] == pytest.approx(-1.1)
    assert scores[0, 9] == pytest.approx(-1.05)


def test_mineru_finalization_reports_malformed_nonempty_layout() -> None:
    pipeline = VlmPipeline.__new__(VlmPipeline)
    pipeline.pipeline_options = SimpleNamespace(
        vlm_options=VlmConvertOptions.from_preset(
            "mineru2_pro", engine_options=AutoInlineVlmEngineOptions()
        ),
        generate_page_images=False,
        generate_picture_images=False,
        images_scale=1.0,
    )
    pipeline.force_backend_text = False
    pages = [
        Page(
            page_no=4,
            size=Size(width=100, height=100),
            predictions=PagePredictions(
                vlm_response=VlmPrediction(
                    text=serialize_mineru2_transcript("not layout", [])
                )
            ),
        ),
        Page(
            page_no=5,
            size=Size(width=100, height=100),
            predictions=PagePredictions(
                vlm_response=VlmPrediction(text=serialize_mineru2_transcript("", []))
            ),
        ),
    ]
    conv_res = SimpleNamespace(
        input=SimpleNamespace(file=PurePath("test.pdf")),
        errors=[],
        pages=pages,
        status=ConversionStatus.STARTED,
    )

    documents = [pipeline._finalize_page_document(conv_res, page) for page in pages]

    assert not documents[0].texts
    assert not documents[1].texts
    assert [(error.page_no, error.category) for error in conv_res.errors] == [
        (4, FailureCategory.INFERENCE_FAILURE)
    ]
    assert pipeline._determine_status(conv_res) == ConversionStatus.PARTIAL_SUCCESS

    with pytest.raises(ValueError, match="malformed transcript envelope"):
        parse_mineru2(
            '{"version":2,"layout":"","recognition":[]}',
            original_page_size=Size(width=100, height=100),
            page_no=6,
        )


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
