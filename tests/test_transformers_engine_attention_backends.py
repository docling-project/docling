# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The attention backends the Transformers engine allows during generation.

On CUDA the engine keeps PyTorch from choosing the cuDNN attention backend,
whose graph cache grows with every decode step and is never freed
(pytorch/pytorch#202607). Other devices are left unchanged.
"""

from __future__ import annotations

import sys
from contextlib import AbstractContextManager
from typing import Any

import pytest
import torch
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options_vlm_model import TransformersPromptStyle
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.models.inference_engines.vlm import transformers_engine
from docling.models.inference_engines.vlm.base import VlmEngineInput
from docling.models.inference_engines.vlm.transformers_engine import (
    TransformersVlmEngine,
    _generation_attention_backends,
)


def _enabled_backends() -> dict[str, bool]:
    return {
        "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
    }


def test_cuda_generation_excludes_cudnn_attention() -> None:
    before = _enabled_backends()

    with _generation_attention_backends("cuda:0"):
        inside = _enabled_backends()

    assert inside == {"cudnn": False, "flash": True, "efficient": True, "math": True}
    assert _enabled_backends() == before


@pytest.mark.parametrize("device", ["cpu", "mps", "xpu", None])
def test_other_devices_keep_the_attention_backends(device: str | None) -> None:
    before = _enabled_backends()

    with _generation_attention_backends(device):
        assert _enabled_backends() == before


def test_torch_without_sdpa_kernel_keeps_the_attention_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # torch < 2.3 has no torch.nn.attention.sdpa_kernel.
    monkeypatch.setitem(sys.modules, "torch.nn.attention", None)
    before = _enabled_backends()

    with _generation_attention_backends("cuda"):
        assert _enabled_backends() == before


class _Processor:
    """Processor and tokenizer in one, with a single token."""

    pad_token = "_"
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "x" * len(ids)

    def batch_decode(self, sequences, **kwargs) -> list[str]:
        return [self.decode(seq) for seq in sequences]

    def __call__(self, text, images, **kwargs):
        return {"input_ids": torch.ones((len(text), 1), dtype=torch.long)}


class _RecordingModel:
    """Records the attention backends that are enabled while generate() runs."""

    def __init__(self) -> None:
        self.backends_during_generate: dict[str, bool] | None = None

    def generate(self, input_ids, generation_config, stopping_criteria=None):
        self.backends_during_generate = _enabled_backends()
        return torch.cat([input_ids, torch.ones_like(input_ids)], dim=-1)


def test_predict_batch_generates_within_the_attention_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices: list[str | None] = []

    def as_on_cuda(device: str | None) -> AbstractContextManager[Any]:
        # The engine runs on the CPU here; record its device and apply the CUDA choice.
        devices.append(device)
        return _generation_attention_backends("cuda:0")

    monkeypatch.setattr(
        transformers_engine, "_generation_attention_backends", as_on_cuda
    )
    engine = TransformersVlmEngine(
        options=TransformersVlmEngineOptions(),
        accelerator_options=AcceleratorOptions(device="cpu"),
        artifacts_path=None,
    )
    engine._initialized = True
    engine.device = "cpu"
    engine.processor = _Processor()
    model = _RecordingModel()
    engine.vlm_model = model  # type: ignore[assignment]

    engine.predict_batch(
        [
            VlmEngineInput(
                image=Image.new("RGB", (8, 8), "white"),
                prompt="Prompt",
                max_new_tokens=4,
                extra_generation_config={
                    "transformers_prompt_style": TransformersPromptStyle.RAW
                },
            )
        ]
    )

    assert devices == ["cpu"]
    assert model.backends_during_generate is not None
    assert model.backends_during_generate["cudnn"] is False
    assert torch.backends.cuda.cudnn_sdp_enabled() is True
