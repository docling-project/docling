# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pytest

pytestmark = pytest.mark.cross_platform


def _setup_env(monkeypatch):
    monkeypatch.setenv("DOCLING_PERF_PAGE_BATCH_SIZE", "12")
    monkeypatch.setenv("DOCLING_DEBUG_VISUALIZE_RAW_LAYOUT", "True")
    monkeypatch.setenv("DOCLING_ARTIFACTS_PATH", "/path/to/artifacts")
    monkeypatch.setenv("DOCLING_INFERENCE_COMPILE_TORCH_MODELS", "True")


def test_settings(monkeypatch):
    _setup_env(monkeypatch)

    import importlib

    import docling.datamodel.settings as m

    # Reinitialize settings module
    importlib.reload(m)

    # Check top level setting
    assert str(m.settings.artifacts_path) == "/path/to/artifacts"

    # Check nested set via environment variables
    assert m.settings.perf.page_batch_size == 12
    assert m.settings.debug.visualize_raw_layout is True
    assert m.settings.inference.compile_torch_models is True

    # Check nested defaults
    assert m.settings.perf.doc_batch_size == 1
    assert m.settings.debug.visualize_ocr is False


def test_compile_model_defaults_from_settings(monkeypatch):
    monkeypatch.setenv("DOCLING_INFERENCE_COMPILE_TORCH_MODELS", "True")

    import importlib

    import docling.datamodel.settings as settings_module
    from docling.datamodel.image_classification_engine_options import (
        TransformersImageClassificationEngineOptions,
    )
    from docling.datamodel.object_detection_engine_options import (
        TransformersObjectDetectionEngineOptions,
    )
    from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions

    importlib.reload(settings_module)

    assert TransformersObjectDetectionEngineOptions().compile_model is True
    assert TransformersImageClassificationEngineOptions().compile_model is True
    assert TransformersVlmEngineOptions().compile_model is True


def test_scoped_settings_restores_state():
    import importlib

    import docling.datamodel.settings as settings_module

    settings_module = importlib.reload(settings_module)

    original = settings_module.settings.model_copy(deep=True)

    with pytest.raises(RuntimeError):
        with settings_module.scoped(
            perf=settings_module.BatchConcurrencySettings(page_batch_size=99),
            debug=settings_module.DebugSettings(profile_pipeline_timings=True),
            inference=settings_module.InferenceSettings(compile_torch_models=False),
        ):
            assert settings_module.settings.perf.page_batch_size == 99
            assert settings_module.settings.debug.profile_pipeline_timings is True
            assert settings_module.settings.inference.compile_torch_models is False
            raise RuntimeError("boom")

    assert settings_module.settings.model_dump(mode="json") == original.model_dump(
        mode="json"
    )

    fresh = settings_module.defaults()
    assert fresh is not settings_module.settings
    assert fresh.model_dump(mode="json") == settings_module.settings.model_dump(
        mode="json"
    )


def test_page_range_clamped_to_maxsize():
    import sys

    from pydantic import ValidationError

    from docling.datamodel.settings import DocumentLimits

    # (1, sys.maxsize) after a JavaScript JSON round-trip becomes (1, 2**63)
    limits = DocumentLimits(page_range=(1, 2**63))
    assert limits.page_range == (1, sys.maxsize)

    limits = DocumentLimits(page_range=(2**63, 2**64))
    assert limits.page_range == (sys.maxsize, sys.maxsize)

    limits = DocumentLimits(page_range=(3, 7))
    assert limits.page_range == (3, 7)

    with pytest.raises(ValidationError):
        DocumentLimits(page_range=(0, 2**63))
    with pytest.raises(ValidationError):
        DocumentLimits(page_range=(5, 4))
