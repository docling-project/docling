# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The cuDNN attention switch used around Transformers generation.

On CUDA, generation keeps PyTorch from choosing the cuDNN attention backend, whose
graph cache grows with every decode step and is never freed
(pytorch/pytorch#202607). The flag is process-wide, so overlapping generations on
several threads share one reference-counted switch. Other devices are unchanged.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager

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
)
from docling.utils.accelerator_utils import cudnn_attention_disabled


def _enabled_backends() -> dict[str, bool]:
    return {
        "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
    }


@pytest.fixture
def flash_disabled() -> Iterator[None]:
    torch.backends.cuda.enable_flash_sdp(False)
    try:
        yield
    finally:
        torch.backends.cuda.enable_flash_sdp(True)


def test_cuda_generation_disables_only_cudnn_attention() -> None:
    before = _enabled_backends()

    with cudnn_attention_disabled("cuda:0"):
        inside = _enabled_backends()

    assert inside == {**before, "cudnn": False}
    assert _enabled_backends() == before


def test_other_backend_choices_are_kept(flash_disabled: None) -> None:
    with cudnn_attention_disabled("cuda"):
        assert torch.backends.cuda.flash_sdp_enabled() is False

    assert torch.backends.cuda.flash_sdp_enabled() is False


def test_overlapping_uses_restore_after_the_last_one() -> None:
    first = cudnn_attention_disabled("cuda:0")
    second = cudnn_attention_disabled("cuda:1")

    first.__enter__()
    second.__enter__()
    first.__exit__(None, None, None)
    still_disabled = torch.backends.cuda.cudnn_sdp_enabled()
    second.__exit__(None, None, None)

    assert still_disabled is False
    assert torch.backends.cuda.cudnn_sdp_enabled() is True


def test_overlapping_threads_keep_cudnn_off_until_both_finish() -> None:
    first_inside, second_inside, first_done = (threading.Event() for _ in range(3))
    seen_while_second_runs: list[bool] = []

    def first() -> None:
        with cudnn_attention_disabled("cuda:0"):
            first_inside.set()
            second_inside.wait(timeout=5)
        first_done.set()

    def second() -> None:
        first_inside.wait(timeout=5)
        with cudnn_attention_disabled("cuda:0"):
            second_inside.set()
            first_done.wait(timeout=5)
            seen_while_second_runs.append(torch.backends.cuda.cudnn_sdp_enabled())

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert seen_while_second_runs == [False]
    assert torch.backends.cuda.cudnn_sdp_enabled() is True


def test_previous_disabled_state_is_restored() -> None:
    torch.backends.cuda.enable_cudnn_sdp(False)
    try:
        with cudnn_attention_disabled("cuda:0"):
            pass
        assert torch.backends.cuda.cudnn_sdp_enabled() is False
    finally:
        torch.backends.cuda.enable_cudnn_sdp(True)


def test_flag_is_restored_when_generation_raises() -> None:
    with pytest.raises(RuntimeError), cudnn_attention_disabled("cuda:0"):
        raise RuntimeError("generation failed")

    assert torch.backends.cuda.cudnn_sdp_enabled() is True


@pytest.mark.parametrize("device", ["cpu", "mps", "xpu", None])
def test_other_devices_keep_the_attention_backends(device: str | None) -> None:
    before = _enabled_backends()

    with cudnn_attention_disabled(device):
        assert _enabled_backends() == before


def test_torch_without_cudnn_attention_keeps_the_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Older torch builds have no cuDNN attention backend.
    monkeypatch.setitem(sys.modules, "torch.backends.cuda", None)
    before = _enabled_backends()

    with cudnn_attention_disabled("cuda"):
        monkeypatch.undo()
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
    """Records whether cuDNN attention is enabled while generate() runs."""

    def __init__(self) -> None:
        self.cudnn_during_generate: bool | None = None

    def generate(self, input_ids, generation_config, stopping_criteria=None):
        self.cudnn_during_generate = torch.backends.cuda.cudnn_sdp_enabled()
        return torch.cat([input_ids, torch.ones_like(input_ids)], dim=-1)


def test_engine_generates_within_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    devices: list[str | None] = []

    @contextmanager
    def as_on_cuda(device: str | None) -> Iterator[None]:
        # The engine runs on the CPU here; record its device and apply the CUDA choice.
        devices.append(device)
        with cudnn_attention_disabled("cuda:0"):
            yield

    monkeypatch.setattr(transformers_engine, "cudnn_attention_disabled", as_on_cuda)
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
    assert model.cudnn_during_generate is False
    assert torch.backends.cuda.cudnn_sdp_enabled() is True
