from PIL import Image

from docling.datamodel.stage_model_specs import EngineModelConfig
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.models.inference_engines.vlm.api_openai_compatible_engine import (
    ApiVlmEngine,
)
from docling.models.inference_engines.vlm.base import VlmEngineInput, VlmEngineType


def test_api_vlm_engine_uses_request_generation_settings_over_model_defaults(
    monkeypatch,
) -> None:
    captured = {}

    def _fake_api_image_request(**kwargs):
        captured.update(kwargs)
        return "ok", 1, "stop"

    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.api_openai_compatible_engine.api_image_request",
        _fake_api_image_request,
    )

    engine = ApiVlmEngine(
        enable_remote_services=True,
        options=ApiVlmEngineOptions(
            engine_type=VlmEngineType.API_OPENAI,
            url="http://localhost:11434/v1/chat/completions",
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
            VlmEngineInput(
                image=Image.new("RGB", (8, 8), "white"),
                prompt="Prompt",
                temperature=0.4,
                max_new_tokens=128,
                stop_strings=["</doctag>"],
            )
        ]
    )

    assert [output.text for output in outputs] == ["ok"]
    assert captured["model"] == "test-model"
    assert captured["temperature"] == 0.4
    assert captured["max_tokens"] == 128
    assert captured["stop"] == ["</doctag>"]


def test_api_vlm_engine_allows_explicit_user_params_to_override_request_settings(
    monkeypatch,
) -> None:
    captured = {}

    def _fake_api_image_request(**kwargs):
        captured.update(kwargs)
        return "ok", 1, "stop"

    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.api_openai_compatible_engine.api_image_request",
        _fake_api_image_request,
    )

    engine = ApiVlmEngine(
        enable_remote_services=True,
        options=ApiVlmEngineOptions(
            engine_type=VlmEngineType.API_OPENAI,
            url="http://localhost:11434/v1/chat/completions",
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
        [
            VlmEngineInput(
                image=Image.new("RGB", (8, 8), "white"),
                prompt="Prompt",
                temperature=0.4,
                max_new_tokens=128,
            )
        ]
    )

    assert [output.text for output in outputs] == ["ok"]
    assert captured["model"] == "override-model"
    assert captured["temperature"] == 0.8
    assert captured["max_completion_tokens"] == 256
    assert "max_tokens" not in captured
