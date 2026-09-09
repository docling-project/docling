# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import GenerationConfig

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import VlmStopReason
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.datamodel.vlm_model_specs import (
    GRANITE_VISION_4_1_TRANSFORMERS,
    NU_EXTRACT_2B_TRANSFORMERS,
)
from docling.models.extraction import transformers_extraction_model as module
from docling.models.extraction.transformers_extraction_model import (
    TransformersExtractionModel,
)


class _LoadedModel:
    def eval(self) -> None:
        pass


def test_transformers_engine_options_control_model_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    processor_loader = Mock(
        return_value=SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))
    )
    model_loader = Mock(return_value=_LoadedModel())
    monkeypatch.setattr(module.AutoProcessor, "from_pretrained", processor_loader)
    monkeypatch.setattr(
        module.AutoModelForImageTextToText, "from_pretrained", model_loader
    )
    monkeypatch.setattr(module, "decide_device", lambda *_args, **_kwargs: "cpu")
    options = GRANITE_VISION_4_1_TRANSFORMERS.model_copy(
        update={
            "engine_options": TransformersVlmEngineOptions(
                device=AcceleratorDevice.CPU,
                torch_dtype="float16",
                trust_remote_code=False,
                compile_model=False,
            )
        }
    )

    TransformersExtractionModel(
        enabled=True,
        artifacts_path=tmp_path,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.MPS),
        vlm_options=options,
    )

    assert processor_loader.call_args.kwargs["trust_remote_code"] is False
    assert model_loader.call_args.kwargs["device_map"] == "cpu"
    assert model_loader.call_args.kwargs["dtype"] == "float16"
    assert model_loader.call_args.kwargs["trust_remote_code"] is False


@pytest.mark.parametrize(
    ("generated_tokens", "expected_reason"),
    [
        ([3, 4, 5], VlmStopReason.LENGTH),
        ([3, 99, 0], VlmStopReason.END_OF_SEQUENCE),
    ],
)
def test_transformers_reports_generation_stop_reason(
    generated_tokens: list[int], expected_reason: VlmStopReason
) -> None:
    model = TransformersExtractionModel.__new__(TransformersExtractionModel)
    model.model_spec = NU_EXTRACT_2B_TRANSFORMERS.model_spec.model_copy(
        update={"max_new_tokens": 3}
    )
    model.engine_options = TransformersVlmEngineOptions(use_kv_cache=False)
    model.max_new_tokens = 3
    model.temperature = 0.0
    model.generation_config = GenerationConfig(eos_token_id=99, pad_token_id=0)
    model.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(eos_token_id=99, pad_token_id=0),
        batch_decode=lambda *_args, **_kwargs: ["{}"],
    )
    model.vlm_model = SimpleNamespace(
        generate=lambda **_kwargs: torch.tensor([[1, 2, *generated_tokens]])
    )

    prediction = next(
        iter(model._generate_and_decode({"input_ids": torch.tensor([[1, 2]])}))
    )

    assert prediction.num_tokens == 3
    assert prediction.stop_reason == expected_reason
