# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The stop reason that the vLLM engine reports for each request result."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from docling.datamodel.base_models import VlmStopReason
from docling.models.inference_engines.vlm.base import VlmEngineInput
from docling.models.inference_engines.vlm.vllm_engine import VllmVlmEngine


class _FakeSamplingParams:
    """Stand-in for vllm.SamplingParams, which is an optional dependency."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeVllm:
    def __init__(self, requests):
        self._requests = requests

    def generate(self, llm_inputs, sampling_params=None):
        return self._requests


@pytest.fixture
def fake_vllm():
    """Expose a fake vllm module so the engine runs without a GPU install."""
    module = SimpleNamespace(SamplingParams=_FakeSamplingParams)
    with patch.dict(sys.modules, {"vllm": module}):
        yield module


def _completion(text, token_ids, finish_reason, stop_reason=None):
    """Build a completion with the fields that vLLM CompletionOutput carries."""
    return SimpleNamespace(
        text=text,
        token_ids=token_ids,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
    )


def _engine(requests):
    engine = VllmVlmEngine.__new__(VllmVlmEngine)
    engine._initialized = True
    engine.llm = _FakeVllm(requests)
    engine.processor = SimpleNamespace()
    engine.sampling_params = SimpleNamespace()
    engine.model_config = SimpleNamespace(repo_id="org/model")
    return engine


def _run(engine, count=1):
    inputs = [
        VlmEngineInput(image=Image.new("RGB", (8, 8), "white"), prompt="p")
        for _ in range(count)
    ]
    module = "docling.models.inference_engines.vlm.vllm_engine"
    with (
        patch(module + ".preprocess_image_batch", return_value=["img"] * count),
        patch(module + ".format_prompt_for_vlm", return_value="prompt"),
    ):
        return engine.predict_batch(inputs)


def test_a_request_without_output_is_reported_as_an_inference_error(fake_vllm) -> None:
    """vLLM reports a failed request as an empty output list, not as a stop."""
    engine = _engine([SimpleNamespace(outputs=[])])

    outputs = _run(engine)

    assert outputs[0].text == ""
    assert outputs[0].stop_reason == VlmStopReason.INFERENCE_ERROR.value
    assert outputs[0].metadata["num_tokens"] == 0


def test_a_sequence_that_hit_its_end_token_is_a_completed_call(fake_vllm) -> None:
    """vLLM sets the finish reason to stop when the model emits EOS."""
    engine = _engine([SimpleNamespace(outputs=[_completion("a", [1, 2], "stop")])])

    outputs = _run(engine)

    assert outputs[0].text == "a"
    assert outputs[0].stop_reason == VlmStopReason.END_OF_SEQUENCE.value
    assert outputs[0].metadata["num_tokens"] == 2


def test_a_sequence_that_ran_out_of_tokens_stays_a_truncation(fake_vllm) -> None:
    """A length stop means the output is incomplete and must stay flagged."""
    engine = _engine([SimpleNamespace(outputs=[_completion("a", [1], "length")])])

    outputs = _run(engine)

    assert outputs[0].stop_reason == VlmStopReason.LENGTH.value


def test_an_unknown_finish_reason_is_not_read_as_a_completion(fake_vllm) -> None:
    """An unrecognised reason must never be reported as a finished sequence."""
    engine = _engine([SimpleNamespace(outputs=[_completion("a", [1], "made_up")])])

    outputs = _run(engine)

    assert outputs[0].stop_reason == VlmStopReason.UNSPECIFIED.value


@pytest.mark.parametrize("finish_reason", ["abort", "error"])
def test_an_unfinished_request_is_not_counted_as_a_page(
    fake_vllm, finish_reason
) -> None:
    """vLLM documents abort, and newer versions use error, for a request that
    did not finish, so a page that carries partial text must not be a success."""
    engine = _engine([SimpleNamespace(outputs=[_completion("a", [1], finish_reason)])])

    outputs = _run(engine)

    assert outputs[0].stop_reason == VlmStopReason.INFERENCE_ERROR.value


def test_a_stop_string_end_on_its_own_is_a_completed_call(fake_vllm) -> None:
    """A stop string end comes back as finish_reason stop with the string set."""
    completion = _completion("a", [1], "stop", stop_reason="</doctag>")

    engine = _engine([SimpleNamespace(outputs=[completion])])

    outputs = _run(engine)

    assert outputs[0].stop_reason == VlmStopReason.END_OF_SEQUENCE.value


def test_a_custom_stop_string_does_not_hide_the_finish_reason(fake_vllm) -> None:
    """stop_reason holds the matched stop string, so finish_reason leads."""
    completion = _completion("a", [1], "length", stop_reason="###")

    engine = _engine([SimpleNamespace(outputs=[completion])])

    outputs = _run(engine)

    assert outputs[0].stop_reason == VlmStopReason.LENGTH.value


def test_the_batch_size_metadata_is_kept(fake_vllm) -> None:
    """The engine already reported the batch size and must keep reporting it."""
    engine = _engine(
        [
            SimpleNamespace(outputs=[_completion("a", [1], "stop")]),
            SimpleNamespace(outputs=[_completion("b", [1], "stop")]),
        ]
    )

    outputs = _run(engine, count=2)

    assert [o.metadata["batch_size"] for o in outputs] == [2, 2]
