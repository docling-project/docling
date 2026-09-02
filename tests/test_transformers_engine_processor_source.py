# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Transformers VLM engine regression tests."""

import sys
import types

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options_vlm_model import TransformersModelType
from docling.datamodel.stage_model_specs import EngineModelConfig
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions


def test_transformers_engine_loads_processor_from_repo_id(monkeypatch, tmp_path):
    fake_torch = types.SimpleNamespace(
        __version__="2.9.0",
        compile=lambda model: model,
        bfloat16="bfloat16",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    import docling.models.inference_engines.vlm.transformers_engine as tf_engine

    captured: dict[str, object] = {}

    class FakeProcessor:
        tokenizer = None

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            captured["model_source"] = args[0]
            captured["model_kwargs"] = kwargs
            return cls()

        def eval(self):
            return None

    monkeypatch.setattr(
        tf_engine,
        "resolve_model_artifacts_path",
        lambda **kwargs: tmp_path / "artifacts",
    )
    monkeypatch.setattr(tf_engine, "decide_device", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(
        tf_engine.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: captured.update(
            processor_source=args[0], processor_kwargs=kwargs
        )
        or FakeProcessor(),
    )
    monkeypatch.setattr(tf_engine, "AutoModelForImageTextToText", FakeModel)
    monkeypatch.setattr(
        tf_engine.GenerationConfig,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    tf_engine.TransformersVlmEngine(
        options=TransformersVlmEngineOptions(compile_model=False),
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
        artifacts_path=tmp_path,
        model_config=EngineModelConfig(
            repo_id="docling-project/CodeFormulaV2",
            revision="main",
            extra_config={
                "transformers_model_type": TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
                "torch_dtype": "bfloat16",
            },
        ),
    )

    assert captured["processor_source"] == "docling-project/CodeFormulaV2"
    assert captured["model_source"] == tmp_path / "artifacts"
