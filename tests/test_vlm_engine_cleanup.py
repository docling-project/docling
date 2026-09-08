# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import importlib

import pytest


@pytest.mark.parametrize(
    ("module_name", "class_name", "attrs"),
    [
        (
            "docling.models.inference_engines.vlm.vllm_engine",
            "VllmVlmEngine",
            {"llm": None, "processor": None},
        ),
        (
            "docling.models.inference_engines.vlm.auto_inline_engine",
            "AutoInlineVlmEngine",
            {"actual_engine": None},
        ),
        (
            "docling.models.inference_engines.vlm.api_openai_compatible_engine",
            "ApiVlmEngine",
            {},
        ),
        (
            "docling.models.inference_engines.vlm.mlx_engine",
            "MlxVlmEngine",
            {"vlm_model": None, "processor": None},
        ),
        (
            "docling.models.inference_engines.vlm.transformers_engine",
            "TransformersVlmEngine",
            {"vlm_model": None, "processor": None, "device": None},
        ),
    ],
)
def test_cleanup_handles_unavailable_module_logger(
    monkeypatch, module_name, class_name, attrs
):
    module = importlib.import_module(module_name)
    engine_cls = getattr(module, class_name)
    engine = object.__new__(engine_cls)

    for name, value in attrs.items():
        setattr(engine, name, value)

    monkeypatch.setattr(module, "_log", None)

    engine.cleanup()


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        (
            "docling.models.stages.vlm_convert.vlm_convert_model",
            "VlmConvertModel",
        ),
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
