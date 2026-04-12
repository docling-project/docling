from pathlib import Path
from types import SimpleNamespace

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options_vlm_model import TransformersModelType
from docling.datamodel.stage_model_specs import EngineModelConfig
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.models.inference_engines.vlm.transformers_engine import (
    TransformersVlmEngine,
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
            },
        ),
    )

    assert captured["model_kwargs"]["_attn_implementation"] == "eager"
