# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import importlib

import pytest


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("docling.models.inference_engines.vlm.api_openai_compatible_engine", "ApiVlmEngine"),
        ("docling.models.inference_engines.vlm.auto_inline_engine", "AutoInlineVlmEngine"),
        ("docling.models.inference_engines.vlm.mlx_engine", "MlxVlmEngine"),
        ("docling.models.inference_engines.vlm.transformers_engine", "TransformersVlmEngine"),
        ("docling.models.inference_engines.vlm.vllm_engine", "VllmVlmEngine"),
    ],
)
def test_engine_cleanup_handles_unavailable_module_logger(
    monkeypatch, module_name, class_name
):
    module = importlib.import_module(module_name)
    engine_cls = getattr(module, class_name)
    engine = object.__new__(engine_cls)

    if class_name == "ApiVlmEngine":
        pass
    elif class_name == "AutoInlineVlmEngine":
        engine.actual_engine = None
    elif class_name == "MlxVlmEngine":
        engine.vlm_model = None
        engine.processor = None
    elif class_name == "TransformersVlmEngine":
        engine.vlm_model = None
        engine.processor = None
        engine.device = None
    else:
        engine.llm = None
        engine.processor = None

    monkeypatch.setattr(module, "_log", None)
    engine.cleanup()


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("docling.models.stages.vlm_convert.vlm_convert_model", "VlmConvertModel"),
        (
            "docling.models.stages.picture_description.picture_description_vlm_engine_model",
            "PictureDescriptionVlmEngineModel",
        ),
    ],
)
def test_model_destructor_handles_unavailable_module_logger(
    monkeypatch, module_name, class_name
):
    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name)

    class FailingEngine:
        def cleanup(self):
            raise RuntimeError("cleanup failed")

    model = object.__new__(model_cls)
    model.engine = FailingEngine()
    monkeypatch.setattr(module, "_log", None)

    model_cls.__del__(model)

    if class_name == "VlmConvertModel":
        del model.engine
    else:
        model.engine = None
