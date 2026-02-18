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
