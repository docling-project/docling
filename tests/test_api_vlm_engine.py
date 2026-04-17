from typing import Any

import pytest
from PIL import Image

from docling.datamodel.stage_model_specs import EngineModelConfig
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.models.inference_engines.vlm.api_openai_compatible_engine import (
    ApiVlmEngine,
)
from docling.models.inference_engines.vlm.base import VlmEngineInput, VlmEngineType

_API_URL = "http://localhost:11434/v1/chat/completions"


@pytest.fixture
def captured_api_call(monkeypatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def _fake_api_image_request(**kwargs):
        captured.update(kwargs)
        return "ok", 1, "stop"

    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.api_openai_compatible_engine.api_image_request",
        _fake_api_image_request,
    )
    return captured


def _make_input(**overrides) -> VlmEngineInput:
    params: dict[str, Any] = {
        "image": Image.new("RGB", (8, 8), "white"),
        "prompt": "Prompt",
    }
    params.update(overrides)
    return VlmEngineInput(**params)


def test_api_vlm_engine_uses_request_generation_settings_over_model_defaults(
    captured_api_call,
) -> None:
    engine = ApiVlmEngine(
        enable_remote_services=True,
        options=ApiVlmEngineOptions(
            engine_type=VlmEngineType.API_OPENAI,
            url=_API_URL,
        ),
        model_config=EngineModelConfig(
            extra_config={
                "api_params": {
                    "model": "test-model",
                    "max_tokens": 4096,
                    "temperature": 0.0,
                }
            }
        ),
    )

    outputs = engine.predict_batch(
        [
            _make_input(
                temperature=0.4,
                max_new_tokens=128,
                stop_strings=["</doctag>"],
            )
        ]
    )

    assert [output.text for output in outputs] == ["ok"]
    assert captured_api_call["model"] == "test-model"
    assert captured_api_call["temperature"] == 0.4
    assert captured_api_call["max_tokens"] == 128
    assert captured_api_call["stop"] == ["</doctag>"]


def test_api_vlm_engine_allows_explicit_user_params_to_override_request_settings(
    captured_api_call,
) -> None:
    engine = ApiVlmEngine(
        enable_remote_services=True,
        options=ApiVlmEngineOptions(
            engine_type=VlmEngineType.API_OPENAI,
            url=_API_URL,
            params={
                "model": "override-model",
                "temperature": 0.8,
                "max_completion_tokens": 256,
            },
        ),
        model_config=EngineModelConfig(
            extra_config={"api_params": {"model": "default-model", "max_tokens": 4096}}
        ),
    )

    outputs = engine.predict_batch(
        [_make_input(temperature=0.4, max_new_tokens=128)]
    )

    assert [output.text for output in outputs] == ["ok"]
    assert captured_api_call["model"] == "override-model"
    assert captured_api_call["temperature"] == 0.8
    assert captured_api_call["max_completion_tokens"] == 256
    assert "max_tokens" not in captured_api_call


def test_api_vlm_engine_user_stop_overrides_request_stop_strings(
    captured_api_call,
) -> None:
    """User-provided ``stop`` wins over per-request ``stop_strings`` (#3321)."""
    engine = ApiVlmEngine(
        enable_remote_services=True,
        options=ApiVlmEngineOptions(
            engine_type=VlmEngineType.API_OPENAI,
            url=_API_URL,
            params={"model": "m", "stop": ["USER_STOP"]},
        ),
    )

    engine.predict_batch([_make_input(stop_strings=["MODEL_STOP"])])

    assert captured_api_call["stop"] == ["USER_STOP"]


def test_api_vlm_engine_preserves_user_params_exclusivity(
    captured_api_call,
) -> None:
    """Vendor-specific user params (e.g. watsonx ``model_id``) are not mixed
    with model-spec defaults, so no conflicting ``model`` key leaks through."""
    engine = ApiVlmEngine(
        enable_remote_services=True,
        options=ApiVlmEngineOptions(
            engine_type=VlmEngineType.API,
            url=_API_URL,
            params={"model_id": "vendor-model", "project_id": "proj"},
        ),
        model_config=EngineModelConfig(
            extra_config={"api_params": {"model": "default-model"}}
        ),
    )

    engine.predict_batch([_make_input()])

    assert captured_api_call["model_id"] == "vendor-model"
    assert captured_api_call["project_id"] == "proj"
    assert "model" not in captured_api_call
