import pytest
from docling.exceptions import DoclingModelDownloadError

from docling.models.inference_engines.common.hf_vision_base import HfVisionModelMixin
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.stage_model_specs import EngineModelConfig


def test_hf_vision_base_model_download_failure(monkeypatch, caplog):
    """
    Verify that HfVisionModelMixin handles a failed download correctly within _resolve_model_folder
    - Exception bubbles up
    - Error is logged
    """

    with monkeypatch.context() as m:
        m.setattr(
            "docling.models.inference_engines.common.hf_vision_base.HuggingFaceModelDownloadMixin.download_models",
            lambda *_, **__: (_ for _ in ()).throw(
                DoclingModelDownloadError("Mock download failure")
            ),
        )
        accel = AcceleratorOptions()
        conf = EngineModelConfig()
        conf.repo_id = "docling"
        with pytest.raises(DoclingModelDownloadError):
            mixin = HfVisionModelMixin()
            mixin._init_hf_vision_model(
                model_config=conf,
                accelerator_options=accel,
                artifacts_path=None,
                model_family_name="dummy-model",
            )
            mixin._resolve_model_folder("docling/docling-project", "1.0")

        assert any(
            "failed to download" in record.message.lower() for record in caplog.records
        ), "Expected a warning/error about the download failure in the logs."
