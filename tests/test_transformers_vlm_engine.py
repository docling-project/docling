import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options_vlm_model import TransformersModelType
from docling.datamodel.stage_model_specs import EngineModelConfig
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.models.inference_engines.vlm.base import VlmEngineInput
from docling.models.inference_engines.vlm.transformers_engine import (
    TransformersVlmEngine,
)
from docling.models.inference_engines.vlm.transformers_runtime_adapters import (
    FalconOCRTransformersAdapter,
    falcon_ocr_build_prompt,
)

pytestmark = pytest.mark.skipif(
    os.getenv("CI"),
    reason="Skipping VLM unit tests in CI",
)


def test_transformers_engine_honors_explicit_attn_implementation(
    monkeypatch,
    tmp_path: Path,
):
    captured: dict[str, object] = {}

    class FakeModel:
        def eval(self):
            return self

    def fake_processor_from_pretrained(*args, **kwargs):
        return SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))

    def fake_model_from_pretrained(*args, **kwargs):
        captured["model_kwargs"] = kwargs
        return FakeModel()

    def fake_auto_config_from_pretrained(*args, **kwargs):
        captured["config_kwargs"] = kwargs
        return SimpleNamespace(model_type="falcon_ocr")

    def fake_generation_config_from_pretrained(*args, **kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.transformers_engine.resolve_model_artifacts_path",
        lambda **kwargs: tmp_path,
    )
    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.transformers_engine.AutoProcessor.from_pretrained",
        fake_processor_from_pretrained,
    )
    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.transformers_engine.AutoModelForCausalLM.from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.transformers_runtime_adapters.AutoConfig.from_pretrained",
        fake_auto_config_from_pretrained,
    )
    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.transformers_engine.GenerationConfig.from_pretrained",
        fake_generation_config_from_pretrained,
    )

    TransformersVlmEngine(
        options=TransformersVlmEngineOptions(
            compile_model=False,
            trust_remote_code=True,
        ),
        accelerator_options=AcceleratorOptions(device="cpu"),
        artifacts_path=tmp_path,
        model_config=EngineModelConfig(
            repo_id="tiiuae/Falcon-OCR",
            extra_config={
                "transformers_model_type": TransformersModelType.AUTOMODEL_CAUSALLM,
                "attn_implementation": "eager",
                "transformers_runtime_adapter": FalconOCRTransformersAdapter,
            },
        ),
    )

    assert captured["model_kwargs"]["attn_implementation"] == "eager"
    assert captured["config_kwargs"]["attn_implementation"] == "eager"
    assert captured["model_kwargs"]["config"]._attn_implementation == "eager"
    assert captured["model_kwargs"]["config"]._attn_implementation_internal == "eager"


def test_transformers_engine_accepts_legacy_private_attn_implementation_key(
    monkeypatch,
    tmp_path: Path,
):
    captured: dict[str, object] = {}

    class FakeModel:
        def eval(self):
            return self

    def fake_processor_from_pretrained(*args, **kwargs):
        return SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))

    def fake_model_from_pretrained(*args, **kwargs):
        captured["model_kwargs"] = kwargs
        return FakeModel()

    def fake_auto_config_from_pretrained(*args, **kwargs):
        captured["config_kwargs"] = kwargs
        return SimpleNamespace(model_type="falcon_ocr")

    def fake_generation_config_from_pretrained(*args, **kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.transformers_engine.resolve_model_artifacts_path",
        lambda **kwargs: tmp_path,
    )
    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.transformers_engine.AutoProcessor.from_pretrained",
        fake_processor_from_pretrained,
    )
    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.transformers_engine.AutoModelForCausalLM.from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.transformers_runtime_adapters.AutoConfig.from_pretrained",
        fake_auto_config_from_pretrained,
    )
    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.transformers_engine.GenerationConfig.from_pretrained",
        fake_generation_config_from_pretrained,
    )

    TransformersVlmEngine(
        options=TransformersVlmEngineOptions(
            compile_model=False,
            trust_remote_code=True,
        ),
        accelerator_options=AcceleratorOptions(device="cpu"),
        artifacts_path=tmp_path,
        model_config=EngineModelConfig(
            repo_id="tiiuae/Falcon-OCR",
            extra_config={
                "transformers_model_type": TransformersModelType.AUTOMODEL_CAUSALLM,
                "_attn_implementation": "eager",
                "transformers_runtime_adapter": FalconOCRTransformersAdapter,
            },
        ),
    )

    assert captured["model_kwargs"]["attn_implementation"] == "eager"
    assert captured["config_kwargs"]["attn_implementation"] == "eager"
    assert captured["model_kwargs"]["config"]._attn_implementation == "eager"
    assert captured["model_kwargs"]["config"]._attn_implementation_internal == "eager"


def test_transformers_engine_falls_back_without_generation_config_file(
    monkeypatch,
    tmp_path: Path,
):
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(model_name="falcon")

        def eval(self):
            return self

    def fake_processor_from_pretrained(*args, **kwargs):
        return SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))

    def fake_model_from_pretrained(*args, **kwargs):
        model = FakeModel()
        captured["model"] = model
        captured["model_kwargs"] = kwargs
        return model

    def fake_auto_config_from_pretrained(*args, **kwargs):
        captured["config_kwargs"] = kwargs
        return SimpleNamespace(model_type="falcon_ocr")

    def fake_generation_config_from_pretrained(*args, **kwargs):
        raise OSError("missing file named generation_config.json")

    def fake_generation_config_from_model_config(model_config):
        captured["fallback_model_config"] = model_config
        return SimpleNamespace(source="fallback")

    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.transformers_engine.resolve_model_artifacts_path",
        lambda **kwargs: tmp_path,
    )
    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.transformers_engine.AutoProcessor.from_pretrained",
        fake_processor_from_pretrained,
    )
    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.transformers_engine.AutoModelForCausalLM.from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.transformers_runtime_adapters.AutoConfig.from_pretrained",
        fake_auto_config_from_pretrained,
    )
    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.transformers_engine.GenerationConfig.from_pretrained",
        fake_generation_config_from_pretrained,
    )
    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.transformers_engine.GenerationConfig.from_model_config",
        fake_generation_config_from_model_config,
    )

    engine = TransformersVlmEngine(
        options=TransformersVlmEngineOptions(
            compile_model=False,
            trust_remote_code=True,
        ),
        accelerator_options=AcceleratorOptions(device="cpu"),
        artifacts_path=tmp_path,
        model_config=EngineModelConfig(
            repo_id="tiiuae/Falcon-OCR",
            extra_config={
                "transformers_model_type": TransformersModelType.AUTOMODEL_CAUSALLM,
                "attn_implementation": "eager",
                "transformers_runtime_adapter": FalconOCRTransformersAdapter,
            },
        ),
    )

    assert engine.generation_config.source == "fallback"
    assert captured["fallback_model_config"] is captured["model"].config
    assert captured["config_kwargs"]["attn_implementation"] == "eager"
    assert captured["model_kwargs"]["config"]._attn_implementation == "eager"


def test_transformers_engine_uses_runtime_adapter_private_batch_generate() -> None:
    captured: dict[str, object] = {}

    class FakeFalconModel:
        def _generate_batch(self, image_prompt_pairs, **kwargs):
            captured["image_prompt_pairs"] = image_prompt_pairs
            captured["generation_kwargs"] = kwargs
            return ["page-1", "page-2"]

    engine = TransformersVlmEngine(
        options=TransformersVlmEngineOptions(
            compile_model=False,
            trust_remote_code=True,
        ),
        accelerator_options=AcceleratorOptions(device="cpu"),
        artifacts_path=None,
    )
    engine._initialized = True
    engine.device = "cpu"
    engine.vlm_model = FakeFalconModel()
    engine.model_config = EngineModelConfig(
        repo_id="tiiuae/Falcon-OCR",
        extra_config={"transformers_runtime_adapter": FalconOCRTransformersAdapter},
    )

    inputs = [
        VlmEngineInput(
            image=Image.new("RGB", (8, 8), color="white"),
            prompt=falcon_ocr_build_prompt(""),
            temperature=0.2,
            max_new_tokens=123,
            extra_generation_config={
                "top_k": 7,
                "min_dimension": 80,
                "max_dimension": 900,
                "seed": 99,
            },
        ),
        VlmEngineInput(
            image=Image.new("RGB", (4, 4), color="white"),
            prompt=falcon_ocr_build_prompt("Keep formulas"),
            temperature=0.2,
            max_new_tokens=123,
            extra_generation_config={
                "top_k": 7,
                "min_dimension": 80,
                "max_dimension": 900,
                "seed": 99,
            },
        ),
    ]

    outputs = engine.predict_batch(inputs)

    assert [output.text for output in outputs] == ["page-1", "page-2"]
    assert all(
        output.metadata["transformers_runtime_adapter"] == "falcon_ocr"
        for output in outputs
    )
    assert captured["image_prompt_pairs"] == [
        (
            inputs[0].image,
            "<|image|>Extract the text content from this image.\n<|OCR_PLAIN|>",
        ),
        (
            inputs[1].image,
            "<|image|>Keep formulas\n<|OCR_PLAIN|>",
        ),
    ]
    assert captured["generation_kwargs"] == {
        "max_new_tokens": 123,
        "temperature": 0.2,
        "top_k": 7,
        "min_dimension": 80,
        "max_dimension": 900,
        "seed": 99,
    }


def test_transformers_engine_uses_runtime_adapter_public_generate_fallback() -> None:
    captured: dict[str, object] = {}

    class FakeFalconModel:
        def generate(
            self,
            images,
            *,
            category,
            max_new_tokens,
            temperature,
            top_k,
            min_dimension,
            max_dimension,
            compile,
            seed,
        ):
            captured["images"] = images
            captured["generation_kwargs"] = {
                "category": category,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "min_dimension": min_dimension,
                "max_dimension": max_dimension,
                "compile": compile,
                "seed": seed,
            }
            return ["page-a", "page-b"]

    engine = TransformersVlmEngine(
        options=TransformersVlmEngineOptions(
            compile_model=False,
            trust_remote_code=True,
        ),
        accelerator_options=AcceleratorOptions(device="cpu"),
        artifacts_path=None,
    )
    engine._initialized = True
    engine.device = "cpu"
    engine.vlm_model = FakeFalconModel()
    engine.model_config = EngineModelConfig(
        repo_id="tiiuae/Falcon-OCR",
        extra_config={"transformers_runtime_adapter": FalconOCRTransformersAdapter},
    )

    inputs = [
        VlmEngineInput(
            image=Image.new("RGB", (8, 8), color="white"),
            prompt=falcon_ocr_build_prompt(""),
            temperature=0.2,
            max_new_tokens=123,
            extra_generation_config={
                "top_k": 7,
                "min_dimension": 80,
                "max_dimension": 900,
                "seed": 99,
            },
        ),
        VlmEngineInput(
            image=Image.new("RGB", (4, 4), color="white"),
            prompt=falcon_ocr_build_prompt(
                "Extract the formula content from this image."
            ),
            temperature=0.2,
            max_new_tokens=123,
            extra_generation_config={
                "top_k": 7,
                "min_dimension": 80,
                "max_dimension": 900,
                "seed": 99,
            },
        ),
    ]

    outputs = engine.predict_batch(inputs)

    assert [output.text for output in outputs] == ["page-a", "page-b"]
    assert all(
        output.metadata["transformers_runtime_adapter"] == "falcon_ocr"
        for output in outputs
    )
    assert captured["images"] == [inputs[0].image, inputs[1].image]
    assert captured["generation_kwargs"] == {
        "category": ["plain", "formula"],
        "max_new_tokens": 123,
        "temperature": 0.2,
        "top_k": 7,
        "min_dimension": 80,
        "max_dimension": 900,
        "compile": False,
        "seed": 99,
    }
