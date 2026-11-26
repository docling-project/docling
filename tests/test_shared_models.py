# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import gc
import weakref
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionConfig,
    LlamaConfig,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.chart_extraction_options import ChartExtractionVlmEngineOptions
from docling.datamodel.pipeline_options import (
    GraniteVisionTableStructureOptions,
    PictureDescriptionVlmEngineOptions,
)
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.models.inference_engines.vlm.auto_inline_engine import AutoInlineVlmEngine
from docling.models.inference_engines.vlm.transformers_engine import (
    TransformersVlmEngine,
)
from docling.models.stages.chart_extraction.granite_vision import (
    ChartExtractionVlmEngineModel,
)
from docling.models.stages.picture_description.picture_description_vlm_engine_model import (
    PictureDescriptionVlmEngineModel,
)
from docling.models.stages.table_structure.table_structure_model_granite_vision import (
    GraniteVisionTableStructureModel,
)

pytestmark = pytest.mark.ml_vlm

_CPU = AcceleratorOptions(device="cpu")


@pytest.fixture
def artifacts_path(tmp_path: Path) -> Path:
    """A tiny random VLM in the folder where the Granite Vision stages look for it."""
    model_dir = tmp_path / GraniteVisionTableStructureModel._model_repo_folder
    model = LlavaForConditionalGeneration(
        LlavaConfig(
            vision_config=CLIPVisionConfig(
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=8,
                patch_size=4,
            ),
            text_config=LlamaConfig(
                vocab_size=5,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            ),
            image_token_index=4,
        )
    )
    model.save_pretrained(model_dir)

    vocab = {"<unk>": 0, "<pad>": 1, "<s>": 2, "</s>": 3, "<image>": 4}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    LlavaProcessor(
        image_processor=CLIPImageProcessor(
            size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8}
        ),
        tokenizer=tokenizer,
        patch_size=4,
    ).save_pretrained(model_dir)
    return tmp_path


def _table_stage(artifacts_path: Path) -> GraniteVisionTableStructureModel:
    return GraniteVisionTableStructureModel(
        enabled=True,
        artifacts_path=artifacts_path,
        options=GraniteVisionTableStructureOptions(),
        accelerator_options=_CPU,
    )


def _chart_stage(
    artifacts_path: Path, options: ChartExtractionVlmEngineOptions
) -> ChartExtractionVlmEngineModel:
    return ChartExtractionVlmEngineModel(
        enabled=True,
        artifacts_path=artifacts_path,
        options=options,
        accelerator_options=_CPU,
    )


def _picture_stage(
    artifacts_path: Path, chart_options: ChartExtractionVlmEngineOptions
) -> PictureDescriptionVlmEngineModel:
    return PictureDescriptionVlmEngineModel(
        enabled=True,
        enable_remote_services=False,
        artifacts_path=artifacts_path,
        options=PictureDescriptionVlmEngineOptions(
            model_spec=chart_options.model_spec,
            engine_options=chart_options.engine_options,
        ),
        accelerator_options=_CPU,
    )


def _engine_model(
    stage: ChartExtractionVlmEngineModel | PictureDescriptionVlmEngineModel,
) -> PreTrainedModel:
    """The model of a stage that runs the auto-inline engine on Transformers."""
    assert isinstance(stage.engine, AutoInlineVlmEngine)
    assert isinstance(stage.engine.actual_engine, TransformersVlmEngine)
    assert stage.engine.actual_engine.vlm_model is not None
    return stage.engine.actual_engine.vlm_model


def test_stages_share_one_model(artifacts_path: Path) -> None:
    chart_options = ChartExtractionVlmEngineOptions.from_preset("granite_vision_v4")
    table = _table_stage(artifacts_path)
    chart = _chart_stage(artifacts_path, chart_options)
    picture = _picture_stage(artifacts_path, chart_options)

    assert table._model is _engine_model(chart)
    assert table._model is _engine_model(picture)


def test_model_lives_until_its_last_user_is_deleted(artifacts_path: Path) -> None:
    chart_options = ChartExtractionVlmEngineOptions.from_preset("granite_vision_v4")
    chart = _chart_stage(artifacts_path, chart_options)
    picture = _picture_stage(artifacts_path, chart_options)
    model_ref = weakref.ref(_engine_model(chart))

    del chart
    gc.collect()
    assert model_ref() is _engine_model(picture)

    del picture
    gc.collect()
    assert model_ref() is None


def test_different_dtype_is_not_shared(artifacts_path: Path) -> None:
    table = _table_stage(artifacts_path)
    chart = _chart_stage(
        artifacts_path,
        ChartExtractionVlmEngineOptions.from_preset(
            "granite_vision_v4",
            engine_options=TransformersVlmEngineOptions(torch_dtype="float32"),
        ),
    )

    assert isinstance(chart.engine, TransformersVlmEngine)
    chart_model = chart.engine.vlm_model
    assert chart_model is not None
    assert chart_model is not table._model
    assert chart_model.dtype == torch.float32
    assert table._model.dtype == torch.bfloat16
