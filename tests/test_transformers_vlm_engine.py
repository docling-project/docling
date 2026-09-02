# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions


class _DummyProcessor:
    def __init__(self) -> None:
        self.tokenizer = type("Tokenizer", (), {"model_max_length": 1024})()


class _DummyModel:
    def eval(self) -> None:
        return None


def test_transformers_vlm_engine_uses_repo_id_for_processor(
    monkeypatch, tmp_path
) -> None:
    processor_calls: list[object] = []
    model_calls: list[object] = []

    fake_torch = ModuleType("torch")
    fake_torch.__version__ = "2.9.0"
    fake_torch.compile = lambda model: model

    fake_transformers = ModuleType("transformers")

    def fake_processor_from_pretrained(source, *args, **kwargs):
        processor_calls.append(source)
        if isinstance(source, Path):
            raise AttributeError("'dict' object has no attribute 'model_type'")
        return _DummyProcessor()

    def fake_model_from_pretrained(source, *args, **kwargs):
        model_calls.append(source)
        return _DummyModel()

    fake_transformers.AutoModel = type(
        "AutoModel",
        (),
        {"from_pretrained": staticmethod(fake_model_from_pretrained)},
    )
    fake_transformers.AutoModelForCausalLM = type(
        "AutoModelForCausalLM",
        (),
        {"from_pretrained": staticmethod(fake_model_from_pretrained)},
    )
    fake_transformers.AutoModelForImageTextToText = type(
        "AutoModelForImageTextToText",
        (),
        {"from_pretrained": staticmethod(fake_model_from_pretrained)},
    )
    fake_transformers.AutoProcessor = type(
        "AutoProcessor",
        (),
        {"from_pretrained": staticmethod(fake_processor_from_pretrained)},
    )
    fake_transformers.BitsAndBytesConfig = type("BitsAndBytesConfig", (), {})
    fake_transformers.GenerationConfig = type(
        "GenerationConfig",
        (),
        {"from_pretrained": staticmethod(lambda *args, **kwargs: object())},
    )
    fake_transformers.PreTrainedModel = object
    fake_transformers.StoppingCriteria = object
    fake_transformers.StoppingCriteriaList = list
    fake_transformers.StopStringCriteria = object

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    engine_module = importlib.import_module(
        "docling.models.inference_engines.vlm.transformers_engine"
    )
    monkeypatch.setattr(
        engine_module.importlib.metadata, "version", lambda name: "5.0.0"
    )
    monkeypatch.setattr(
        engine_module,
        "resolve_model_artifacts_path",
        lambda **kwargs: tmp_path / "docling-project--CodeFormulaV2",
    )
    monkeypatch.setattr(engine_module, "decide_device", lambda *args, **kwargs: "cpu")

    engine = object.__new__(engine_module.TransformersVlmEngine)
    engine.options = TransformersVlmEngineOptions(
        device="cpu",
        compile_model=False,
        quantized=False,
        trust_remote_code=False,
    )
    engine.accelerator_options = AcceleratorOptions(device="cpu")
    engine.model_config = SimpleNamespace(
        repo_id="docling-project/CodeFormulaV2",
        revision="main",
        extra_config={},
        torch_dtype=None,
    )
    engine.artifacts_path = tmp_path
    engine.device = "cpu"
    engine._initialized = False

    engine_module.TransformersVlmEngine._load_model_for_repo(
        engine,
        "docling-project/CodeFormulaV2",
    )

    assert processor_calls == ["docling-project/CodeFormulaV2"]
    assert model_calls == [tmp_path / "docling-project--CodeFormulaV2"]
