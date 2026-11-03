# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The stop reason that the legacy vLLM page model reports for each result."""

from types import SimpleNamespace

from docling.datamodel.base_models import VlmStopReason
from docling.models.vlm_pipeline_models.vllm_model import _prediction_from_vllm_request


def _completion(text, token_ids, finish_reason, stop_reason=None):
    """Build a completion with the fields that vLLM CompletionOutput carries."""
    return SimpleNamespace(
        text=text,
        token_ids=token_ids,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
    )


def _predict(request, track_generated_tokens=True):
    return _prediction_from_vllm_request(
        request,
        generation_time=1.0,
        input_prompt="p",
        track_generated_tokens=track_generated_tokens,
        decode_response=lambda text: text,
    )


def test_a_normal_end_of_sequence_is_not_reported_as_a_truncation() -> None:
    """vLLM leaves the stop reason field empty when the model ends on its own."""
    request = SimpleNamespace(outputs=[_completion("a", [1, 2], "stop")])

    prediction = _predict(request)

    assert prediction.stop_reason == VlmStopReason.END_OF_SEQUENCE
    assert prediction.num_tokens == 2
    assert prediction.text == "a"


def test_a_sequence_that_ran_out_of_tokens_stays_a_truncation() -> None:
    request = SimpleNamespace(outputs=[_completion("a", [1], "length")])

    prediction = _predict(request)

    assert prediction.stop_reason == VlmStopReason.LENGTH


def test_a_request_without_a_completion_is_an_inference_error() -> None:
    """An empty output list must not raise an index error inside the batch."""
    request = SimpleNamespace(outputs=[])

    prediction = _predict(request)

    assert prediction.text == ""
    assert prediction.num_tokens == 0
    assert prediction.stop_reason == VlmStopReason.INFERENCE_ERROR
    assert prediction.error_message


def test_the_generated_tokens_are_dropped_when_they_are_not_tracked() -> None:
    request = SimpleNamespace(outputs=[_completion("a", [1, 2], "stop")])

    prediction = _predict(request, track_generated_tokens=False)

    assert prediction.generated_tokens == []
    assert prediction.num_tokens == 2
