# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Remote VLM calls driven against a real OpenAI-compatible server.

Every remote VLM and picture-description path funnels through
``docling.utils.api_image_request``, which uses ``requests``. An httpx-only
mock cannot see that traffic at all, so these run against a real socket.

The emphasis is on whether the *options* a caller configures actually govern
the outgoing call -- the same question as the ``serialize_as_any`` regression,
one layer further out -- plus the streaming paths, which only a genuinely
chunked response exercises.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from PIL import Image
from pydantic import AnyUrl

from docling.datamodel.base_models import VlmStopReason
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType
from docling.models.inference_engines.vlm.api_openai_compatible_engine import (
    ApiVlmEngine,
)
from docling.models.inference_engines.vlm.base import VlmEngineInput
from docling.models.stages.vlm_convert.vlm_convert_model import (
    _prediction_from_engine_output,
)
from docling.models.utils.generation_utils import GenerationStopper
from docling.models.vlm_pipeline_models.api_vlm_model import ApiVlmModel
from docling.utils.api_image_request import (
    api_image_request,
    api_image_request_streaming,
)
from tests.fakes.http_service import FakeService
from tests.fakes.openai_compatible import FakeOpenAiApi


@pytest.fixture
def api() -> Iterator[FakeOpenAiApi]:
    service = FakeService()
    fake = FakeOpenAiApi()
    service.include(fake.router)
    service.start()
    fake.service = service
    try:
        yield fake
    finally:
        service.stop()


@pytest.fixture
def endpoint(api: FakeOpenAiApi) -> AnyUrl:
    return AnyUrl(f"{api.service.base_url}/v1/chat/completions")


@pytest.fixture
def image() -> Image.Image:
    return Image.new("RGB", (8, 8), color="white")


def _sent_body(api: FakeOpenAiApi) -> dict:
    requests = api.service.requests_for("POST", r"/v1/chat/completions")
    assert requests, "no call reached the API"
    return requests[-1].json()


# -- what the caller configures must govern the call ---------------------


def test_generation_params_are_merged_into_the_request_body(api, endpoint, image):
    api_image_request(
        image,
        "describe",
        endpoint,
        model="custom-vlm",
        temperature=0.1,
        max_tokens=256,
    )

    body = _sent_body(api)
    assert body["model"] == "custom-vlm"
    assert body["temperature"] == 0.1
    assert body["max_tokens"] == 256


def test_custom_headers_are_sent(api, endpoint, image):
    api_image_request(
        image,
        "describe",
        endpoint,
        headers={"Authorization": "Bearer sk-not-real", "X-Tenant": "acme"},
        model="custom-vlm",
    )

    sent = api.service.requests_for("POST", r"/v1/chat/completions")[-1].headers
    assert sent["authorization"] == "Bearer sk-not-real"
    assert sent["x-tenant"] == "acme"


def test_the_prompt_and_image_are_both_sent_in_the_message(api, endpoint, image):
    api_image_request(image, "what is in this picture?", endpoint, model="m")

    content = _sent_body(api)["messages"][0]["content"]
    kinds = {part["type"] for part in content}
    assert kinds == {"image_url", "text"}
    text_part = next(p for p in content if p["type"] == "text")
    assert text_part["text"] == "what is in this picture?"
    image_part = next(p for p in content if p["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_timeout_is_applied_and_reported_as_an_inference_error(api, endpoint, image):
    """A read timeout is not raised, but it is not an empty completion either.

    The caller gets an empty result marked INFERENCE_ERROR with the reason, so it
    can be told apart from a model that genuinely produced nothing (#4009).
    """
    api.delay_seconds = 1.0

    started = time.monotonic()
    result = api_image_request(image, "describe", endpoint, timeout=0.15, model="m")
    elapsed = time.monotonic() - started

    assert result.text == ""
    assert result.stop_reason == VlmStopReason.INFERENCE_ERROR
    assert result.error is not None and "timed out" in result.error
    # It gave up on the configured timeout instead of waiting for the response.
    assert elapsed < 0.9


# -- responses -----------------------------------------------------------


def test_completion_text_and_token_count_are_returned(api, endpoint, image):
    api.completion = "A white square."
    api.prompt_tokens, api.completion_tokens = 11, 7

    result = api_image_request(image, "describe", endpoint, model="m")

    assert result.text == "A white square."
    assert result.num_tokens == 18
    assert result.stop_reason == VlmStopReason.END_OF_SEQUENCE


def test_a_missing_usage_block_leaves_the_token_count_unset(api, endpoint, image):
    api.report_usage = False

    result = api_image_request(image, "describe", endpoint, model="m")

    assert result.text
    assert result.num_tokens is None


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_api_errors_do_not_raise_but_are_marked_as_inference_errors(
    api, endpoint, image, status
):
    """A failed call is not an exception, but it is not empty output either:
    the result carries INFERENCE_ERROR and the provider's status (#4009)."""
    api.fail_status = status

    result = api_image_request(image, "describe", endpoint, model="m")

    assert result.text == ""
    assert result.stop_reason == VlmStopReason.INFERENCE_ERROR
    assert result.error is not None and result.error.startswith(f"HTTP {status}")


# -- streaming -----------------------------------------------------------


def test_streamed_deltas_are_reassembled_in_order(api, endpoint, image):
    api.stream_chunks = ["Hel", "lo ", "world"]

    result = api_image_request_streaming(image, "describe", endpoint, model="m")

    assert result.text == "Hello world"
    assert _sent_body(api)["stream"] is True


def test_streaming_reports_usage_from_the_final_chunk(api, endpoint, image):
    api.stream_chunks = ["a", "b"]
    api.prompt_tokens, api.completion_tokens = 3, 4

    result = api_image_request_streaming(image, "describe", endpoint, model="m")

    assert result.num_tokens == 7


def test_non_data_lines_in_the_stream_are_ignored(api, endpoint, image):
    """Proxies inject comments and keep-alives; they must not corrupt output."""
    api.stream_chunks = ["ok"]
    api.stream_preamble = [": keep-alive comment", "event: ping"]

    result = api_image_request_streaming(image, "describe", endpoint, model="m")

    assert result.text == "ok"


def test_malformed_json_chunks_are_skipped(api, endpoint, image):
    api.stream_chunks = ["good"]
    api.stream_preamble = ["data: {not valid json"]

    result = api_image_request_streaming(image, "describe", endpoint, model="m")

    assert result.text == "good"


def test_streamed_logprobs_are_collected_in_order(api, endpoint, image):
    api.stream_chunks = ["Hel", "lo"]
    api.stream_logprobs = [-0.25, -1.5]

    result = api_image_request_streaming(
        image, "describe", endpoint, model="m", logprobs=True
    )

    assert result.logprobs is not None
    assert [(t.token, t.logprob) for t in result.logprobs.content] == [
        ("Hel", -0.25),
        ("lo", -1.5),
    ]


@pytest.mark.parametrize("status", [400, 500])
def test_streaming_api_errors_do_not_raise_but_are_reported(
    api, endpoint, image, status
):
    """Like the non-streaming call, a rejected streaming call must not raise:
    one failing page would otherwise abort the whole document."""
    api.fail_status = status

    result = api_image_request_streaming(image, "describe", endpoint, model="m")

    assert result.text == ""
    assert result.error is not None and result.error.startswith(f"HTTP {status}")


def test_streaming_timeout_is_reported_as_an_error(api, endpoint, image):
    api.delay_seconds = 1.0

    result = api_image_request_streaming(
        image, "describe", endpoint, timeout=0.15, model="m"
    )

    assert result.text == ""
    assert result.error is not None and "timed out" in result.error


def test_a_stream_that_stalls_midway_is_reported_and_its_partial_text_dropped(
    api, endpoint, image
):
    """Text received before the stream stalled is incomplete, so it is not
    returned as if it were the page's output."""
    api.stream_chunks = ["first ", "second ", "third"]
    api.stream_stall_after = 2
    api.delay_seconds = 1.0

    result = api_image_request_streaming(
        image, "describe", endpoint, timeout=0.15, model="m"
    )

    assert result.text == ""
    assert result.error is not None


# -- the engine layer ----------------------------------------------------


def _engine(api: FakeOpenAiApi, **overrides) -> ApiVlmEngine:
    options = ApiVlmEngineOptions(
        engine_type=VlmEngineType.API_OPENAI,
        url=AnyUrl(f"{api.service.base_url}/v1/chat/completions"),
        **overrides,
    )
    return ApiVlmEngine(enable_remote_services=True, options=options)


def test_engine_options_drive_the_outgoing_request(api, image):
    """params and headers set on the engine options must reach the server."""
    engine = _engine(
        api,
        params={"model": "custom-vlm", "temperature": 0.25},
        headers={"X-Tenant": "acme"},
    )

    outputs = engine.predict_batch([VlmEngineInput(image=image, prompt="describe")])

    assert outputs[0].text
    body = _sent_body(api)
    assert body["model"] == "custom-vlm"
    assert body["temperature"] == 0.25
    sent = api.service.requests_for("POST", r"/v1/chat/completions")[-1].headers
    assert sent["x-tenant"] == "acme"


def test_engine_keeps_logprobs_when_a_stopper_aborts_the_stream(api, image):
    """A custom stopper switches the engine to streaming; generated tokens
    must still reach the prediction, up to the point where it stopped."""

    class _StopOnLoop(GenerationStopper):
        def should_stop(self, s: str) -> bool:
            return "loop" in s

    api.stream_chunks = ["ok ", "loop", " not read"]
    api.stream_logprobs = [-0.1, -0.2, -0.3]
    engine = _engine(api, params={"model": "m", "logprobs": True})

    outputs = engine.predict_batch(
        [
            VlmEngineInput(
                image=image,
                prompt="describe",
                extra_generation_config={"custom_stopping_criteria": [_StopOnLoop]},
            )
        ]
    )

    assert _sent_body(api)["stream"] is True
    prediction = _prediction_from_engine_output(outputs[0])
    assert prediction.text == "ok loop"
    assert [(t.text, t.logprob) for t in prediction.generated_tokens] == [
        ("ok ", -0.1),
        ("loop", -0.2),
    ]


def test_engine_reports_a_failed_stream_as_an_inference_error(api, image):
    """With a stopper the engine streams; a failed stream must reach the
    prediction as INFERENCE_ERROR with its reason, as a failed non-streamed
    call does, so the pipeline marks only that page as failed."""

    class _NeverStop(GenerationStopper):
        def should_stop(self, s: str) -> bool:
            return False

    api.fail_status = 400
    engine = _engine(api, params={"model": "m"})

    outputs = engine.predict_batch(
        [
            VlmEngineInput(
                image=image,
                prompt="describe",
                extra_generation_config={"custom_stopping_criteria": [_NeverStop]},
            )
        ]
    )

    assert _sent_body(api)["stream"] is True
    prediction = _prediction_from_engine_output(outputs[0])
    assert prediction.text == ""
    assert prediction.stop_reason == VlmStopReason.INFERENCE_ERROR
    assert prediction.error_message is not None
    assert prediction.error_message.startswith("HTTP 400")


def test_api_vlm_model_reports_a_failed_stream_as_an_inference_error(
    api, endpoint, image
):
    """The VlmPipeline's ApiVlmModel also streams when given a stopper; a failed
    stream must reach its prediction as INFERENCE_ERROR with the reason."""

    class _NeverStop(GenerationStopper):
        def should_stop(self, s: str) -> bool:
            return False

    api.fail_status = 400
    options = ApiVlmOptions(
        url=endpoint,
        prompt="describe",
        response_format=ResponseFormat.PLAINTEXT,
        params={"model": "m"},
        custom_stopping_criteria=[_NeverStop()],
    )
    model = ApiVlmModel(enabled=True, enable_remote_services=True, vlm_options=options)

    (prediction,) = model.process_images([image], "describe")

    assert _sent_body(api)["stream"] is True
    assert prediction.text == ""
    assert prediction.stop_reason == VlmStopReason.INFERENCE_ERROR
    assert prediction.error_message is not None
    assert prediction.error_message.startswith("HTTP 400")


def test_engine_requires_remote_services_to_be_enabled(api):
    options = ApiVlmEngineOptions(
        engine_type=VlmEngineType.API_OPENAI,
        url=AnyUrl(f"{api.service.base_url}/v1/chat/completions"),
    )

    with pytest.raises(Exception, match="remote"):
        ApiVlmEngine(enable_remote_services=False, options=options)


def test_engine_batches_every_input(api, image):
    engine = _engine(api, params={"model": "m"})

    outputs = engine.predict_batch(
        [
            VlmEngineInput(image=image, prompt="first"),
            VlmEngineInput(image=image, prompt="second"),
        ]
    )

    assert len(outputs) == 2
    assert len(api.service.requests_for("POST", r"/v1/chat/completions")) == 2
