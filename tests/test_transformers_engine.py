import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.stage_model_specs import EngineModelConfig
from docling.exceptions import DoclingModelDownloadError
from docling.models.inference_engines.vlm.transformers_engine import (
    TransformersVlmEngine,
    TransformersVlmEngineOptions,
)


def test_transformers_vlm_engine_download_failure(monkeypatch, caplog):
    """
    Verify that TransformersVlmEngine handles a failed download correctly within resolve_model_folder
    - Exception bubbles up
    - Error is logged
    """
    with monkeypatch.context() as m:
        m.setattr(
            "docling.models.inference_engines.vlm.transformers_engine.HuggingFaceModelDownloadMixin.download_models",
            lambda *_, **__: (_ for _ in ()).throw(
                DoclingModelDownloadError("Mock download failure")
            ),
        )
        opts = TransformersVlmEngineOptions()
        accel = AcceleratorOptions()
        conf = EngineModelConfig()
        conf.repo_id = "dummy-repo"
        with pytest.raises(DoclingModelDownloadError):
            TransformersVlmEngine(
                options=opts,
                accelerator_options=accel,
                model_config=conf,
                artifacts_path=None,
            )

        assert any(
            "failed to download" in record.message.lower() for record in caplog.records
        ), "Expected a warning/error about the download failure in the logs."
