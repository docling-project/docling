import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.stage_model_specs import EngineModelConfig
from docling.exceptions import DoclingModelDownloadError
from docling.models.inference_engines.vlm.vllm_engine import (
    VllmVlmEngine,
    VllmVlmEngineOptions,
)


def test_vllm_engine_model_download_failure(monkeypatch, caplog):
    """
    Verify that VllmVlmEngine handles a failed download correctly within resolve_model_folder
    - Exception bubbles up
    - Error is logged
    """
    with monkeypatch.context() as m:
        m.setattr(
            "docling.models.utils.hf_model_download.HuggingFaceModelDownloadMixin.download_models",
            lambda *_, **__: (_ for _ in ()).throw(
                DoclingModelDownloadError("Mock download failure")
            ),
        )

        opts = VllmVlmEngineOptions()
        accel = AcceleratorOptions()
        conf = EngineModelConfig()
        conf.repo_id = "dummy-repo"
        with pytest.raises(DoclingModelDownloadError):
            VllmVlmEngine(
                options=opts,
                accelerator_options=accel,
                model_config=conf,
                artifacts_path=None,
            )

        assert any(
            "failed to download" in record.message.lower() for record in caplog.records
        ), "Expected a warning/error about the download failure in the logs."
