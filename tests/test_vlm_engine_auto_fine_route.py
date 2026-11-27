# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""fine_route="auto": the model's density router chooses the connector path.

The Transformers and MLX engines ask the router of the model for each page and
call the processor with the path it chooses. The vLLM engine cannot ask the
router, so it must not pass "auto" to the processor, which reads it as true.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options_vlm_model import TransformersPromptStyle
from docling.datamodel.vlm_engine_options import (
    MlxVlmEngineOptions,
    TransformersVlmEngineOptions,
)
from docling.models.inference_engines.vlm.base import VlmEngineInput
from docling.models.inference_engines.vlm.mlx_engine import MlxVlmEngine
from docling.models.inference_engines.vlm.transformers_engine import (
    TransformersVlmEngine,
)
from docling.models.inference_engines.vlm.vllm_engine import VllmVlmEngine

_DENSE = (0, 0, 0)
_SPARSE = (255, 255, 255)


def _page(index: int, dense: bool) -> Image.Image:
    """A page that the fake router reads: red = page index, green = dense."""
    return Image.new("RGB", (8, 8), (index, 255 if dense else 0, 0))


class _RoutingProcessor:
    """Processor and tokenizer in one; records the fine_route of each call."""

    pad_token = "_"
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, text, images, **kwargs):
        self.calls.append({"pages": len(images), **kwargs})
        pixels = torch.tensor(
            [[float(v) for v in image.getpixel((0, 0))[:2]] for image in images]
        )
        inputs = {
            "input_ids": torch.ones((len(images), 1), dtype=torch.long),
            "pixel_values": pixels,
            "pixel_attention_mask": torch.ones((len(images), 1), dtype=torch.bool),
        }
        if kwargs.get("fine_route"):
            inputs["tile_fine_mask"] = torch.ones(len(images), dtype=torch.bool)
        return inputs

    def batch_decode(self, sequences, **kwargs) -> list[str]:
        # Token 100 + n: page n on the coarse path; 200 + n: on the fine path.
        return [
            ("fine" if int(seq[0]) >= 200 else "coarse") + str(int(seq[0]) % 100)
            for seq in sequences
        ]


class _RoutingModel:
    """A model whose router sends the dense (green) pages to the fine path."""

    def __init__(self, *, has_router: bool = True) -> None:
        self.router_calls = 0
        self.model = SimpleNamespace(
            density_router=object() if has_router else None,
            predict_fine_route=self._predict_fine_route,
        )

    def _predict_fine_route(self, pixel_values, pixel_attention_mask):
        self.router_calls += 1
        return pixel_values[:, 1] > 0

    def generate(
        self,
        input_ids,
        pixel_values,
        pixel_attention_mask,
        generation_config,
        tile_fine_mask=None,
        **kwargs,
    ):
        base = 100 if tile_fine_mask is None else 200
        reply = (pixel_values[:, 0].long() + base)[:, None]
        return torch.cat([input_ids, reply], dim=-1)


def _transformers_engine(model: _RoutingModel) -> TransformersVlmEngine:
    engine = TransformersVlmEngine(
        options=TransformersVlmEngineOptions(),
        accelerator_options=AcceleratorOptions(device="cpu"),
        artifacts_path=None,
    )
    engine._initialized = True
    engine.device = "cpu"
    engine.processor = _RoutingProcessor()
    engine.vlm_model = model  # type: ignore[assignment]
    return engine


def _inputs(dense: list[bool], fine_route: Any = "auto") -> list[VlmEngineInput]:
    processor_kwargs = {} if fine_route is None else {"fine_route": fine_route}
    return [
        VlmEngineInput(
            image=_page(index, is_dense),
            prompt="Prompt",
            extra_generation_config={
                "transformers_prompt_style": TransformersPromptStyle.RAW,
                "extra_processor_kwargs": {"max_patches": 4, **processor_kwargs},
            },
        )
        for index, is_dense in enumerate(dense)
    ]


def test_transformers_auto_keeps_the_coarse_inputs_when_no_page_is_dense() -> None:
    model = _RoutingModel()
    engine = _transformers_engine(model)

    outputs = engine.predict_batch(_inputs([False, False]))

    assert [output.text for output in outputs] == ["coarse0", "coarse1"]
    assert [output.metadata["fine_route"] for output in outputs] == [False, False]
    # One processor call: the inputs of the router are the inputs of generate.
    assert engine.processor.calls == [{"pages": 2, "max_patches": 4} | _TENSORS]
    assert model.router_calls == 1


def test_transformers_auto_uses_the_fine_path_when_all_pages_are_dense() -> None:
    model = _RoutingModel()
    engine = _transformers_engine(model)

    outputs = engine.predict_batch(_inputs([True, True]))

    assert [output.text for output in outputs] == ["fine0", "fine1"]
    assert [output.metadata["fine_route"] for output in outputs] == [True, True]
    assert [call.get("fine_route") for call in engine.processor.calls] == [None, True]
    assert all(call["max_patches"] == 4 for call in engine.processor.calls)


def test_transformers_auto_splits_a_mixed_batch_and_keeps_the_order() -> None:
    model = _RoutingModel()
    engine = _transformers_engine(model)

    outputs = engine.predict_batch(_inputs([True, False, True, False, False]))

    assert [output.text for output in outputs] == [
        "fine0",
        "coarse1",
        "fine2",
        "coarse3",
        "coarse4",
    ]
    assert [output.metadata["fine_route"] for output in outputs] == [
        True,
        False,
        True,
        False,
        False,
    ]
    # The router call on the whole batch, then one batch for each path.
    assert [
        (call["pages"], call.get("fine_route")) for call in engine.processor.calls
    ] == [(5, None), (3, False), (2, True)]
    assert model.router_calls == 1


def test_transformers_auto_is_coarse_for_a_model_without_a_router() -> None:
    model = _RoutingModel(has_router=False)
    engine = _transformers_engine(model)

    outputs = engine.predict_batch(_inputs([True, True]))

    assert [output.text for output in outputs] == ["coarse0", "coarse1"]
    assert model.router_calls == 0
    # "auto" never reaches the processor, which would read it as true.
    assert all("fine_route" not in call for call in engine.processor.calls)


@pytest.mark.parametrize(
    ("fine_route", "expected"), [(True, "fine"), (False, "coarse"), (None, "coarse")]
)
def test_transformers_fixed_fine_route_does_not_ask_the_router(
    fine_route: bool | None, expected: str
) -> None:
    model = _RoutingModel()
    engine = _transformers_engine(model)

    outputs = engine.predict_batch(_inputs([True, False], fine_route=fine_route))

    assert [output.text for output in outputs] == [f"{expected}0", f"{expected}1"]
    assert model.router_calls == 0
    assert len(engine.processor.calls) == 1
    assert engine.processor.calls[0].get("fine_route") == fine_route
    assert ("fine_route" in outputs[0].metadata) == (fine_route is not None)


_TENSORS = {"return_tensors": "pt", "padding": True}


class _MlxRoutingModel:
    def __init__(self, *, has_router: bool = True) -> None:
        self.has_router = has_router
        self.router_calls = 0

    def __contains__(self, name: str) -> bool:
        return name == "density_router" and self.has_router

    def predict_fine_route(self, pixel_values, pixel_attention_mask):
        self.router_calls += 1
        assert pixel_attention_mask.shape == (1, 1)
        return np.asarray(pixel_values)[:, 1] > 0


def _mlx_engine(model: _MlxRoutingModel, captured: list[dict[str, Any]]) -> Any:
    engine = MlxVlmEngine(options=MlxVlmEngineOptions(), artifacts_path=None)
    engine._initialized = True
    engine.vlm_model = model
    engine.config = object()
    engine.apply_chat_template = lambda *args, **kwargs: "<image>prompt"
    processor_calls: list[dict[str, Any]] = []

    def processor(images, text, **kwargs):
        processor_calls.append({"text": text, **kwargs})
        (image,) = images
        return {
            "pixel_values": np.array([[float(v) for v in image.getpixel((0, 0))[:2]]]),
            "pixel_attention_mask": np.ones((1, 1), dtype=bool),
        }

    def stream_generate(model, processor, prompt, images, **kwargs):
        captured.append(kwargs)
        yield SimpleNamespace(text="fine" if kwargs.get("fine_route") else "coarse")

    engine.processor = processor
    engine.processor_calls = processor_calls
    engine.stream_generate = stream_generate
    return engine


@pytest.fixture
def fake_mlx():
    """Expose a fake mlx.core so the engine runs without Apple Silicon."""
    core = SimpleNamespace(array=lambda value: value)
    with patch.dict(sys.modules, {"mlx": SimpleNamespace(core=core), "mlx.core": core}):
        yield core


def test_mlx_auto_routes_each_page(fake_mlx) -> None:
    model = _MlxRoutingModel()
    captured: list[dict[str, Any]] = []
    engine = _mlx_engine(model, captured)

    outputs = engine.predict_batch(_inputs([True, False, True]))

    assert [output.text for output in outputs] == ["fine", "coarse", "fine"]
    assert [output.metadata["fine_route"] for output in outputs] == [True, False, True]
    assert [kwargs["fine_route"] for kwargs in captured] == [True, False, True]
    assert model.router_calls == 3
    # The router reads the page as the default path tiles it.
    assert engine.processor_calls == [{"text": "<image>prompt", "max_patches": 4}] * 3


def test_mlx_auto_is_coarse_for_a_model_without_a_router() -> None:
    model = _MlxRoutingModel(has_router=False)
    captured: list[dict[str, Any]] = []
    engine = _mlx_engine(model, captured)

    outputs = engine.predict_batch(_inputs([True]))

    assert [output.text for output in outputs] == ["coarse"]
    assert captured[0]["fine_route"] is False
    assert model.router_calls == 0
    assert engine.processor_calls == []


@pytest.mark.parametrize("fine_route", [True, False])
def test_mlx_fixed_fine_route_does_not_ask_the_router(fine_route: bool) -> None:
    model = _MlxRoutingModel()
    captured: list[dict[str, Any]] = []
    engine = _mlx_engine(model, captured)

    engine.predict_batch(_inputs([True, False], fine_route=fine_route))

    assert [kwargs["fine_route"] for kwargs in captured] == [fine_route, fine_route]
    assert model.router_calls == 0


class _RecordingVllm:
    def __init__(self) -> None:
        self.llm_inputs: list[dict[str, Any]] = []

    def generate(self, llm_inputs, sampling_params=None):
        self.llm_inputs = llm_inputs
        completion = SimpleNamespace(
            text="reply", token_ids=[1], finish_reason="stop", stop_reason=None
        )
        return [SimpleNamespace(outputs=[completion]) for _ in llm_inputs]


def _run_vllm(processor_kwargs: dict[str, Any], count: int = 2):
    engine = VllmVlmEngine.__new__(VllmVlmEngine)
    engine._initialized = True
    engine.llm = _RecordingVllm()
    engine.processor = SimpleNamespace()
    engine.sampling_params = SimpleNamespace()
    engine.model_config = SimpleNamespace(repo_id="org/model")
    inputs = [
        VlmEngineInput(
            image=Image.new("RGB", (8, 8), "white"),
            prompt="p",
            extra_generation_config={"extra_processor_kwargs": processor_kwargs},
        )
        for _ in range(count)
    ]
    module = "docling.models.inference_engines.vlm.vllm_engine"
    fake_vllm = SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace())
    with (
        patch.dict(sys.modules, {"vllm": fake_vllm}),
        patch(module + ".preprocess_image_batch", return_value=["img"] * count),
        patch(module + ".format_prompt_for_vlm", return_value="prompt"),
    ):
        outputs = engine.predict_batch(inputs)
    return engine, outputs


def test_vllm_does_not_pass_auto_to_the_processor(caplog) -> None:
    """vLLM's processor reads "auto" as true: every page would be fine."""
    engine, outputs = _run_vllm({"fine_route": "auto", "max_patches": 4})

    assert [output.text for output in outputs] == ["reply", "reply"]
    assert [item["mm_processor_kwargs"] for item in engine.llm.llm_inputs] == [
        {"max_patches": 4}
    ] * 2
    assert caplog.text.count("fine_route='auto' is not available") == 1

    engine, _ = _run_vllm({"fine_route": "auto"})
    assert all("mm_processor_kwargs" not in item for item in engine.llm.llm_inputs)


@pytest.mark.parametrize("fine_route", [True, False])
def test_vllm_still_passes_a_fixed_fine_route(fine_route: bool) -> None:
    engine, _ = _run_vllm({"fine_route": fine_route})

    assert [item["mm_processor_kwargs"] for item in engine.llm.llm_inputs] == [
        {"fine_route": fine_route}
    ] * 2
