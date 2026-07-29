import pytest

from docling.datamodel import accelerator_options
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.exceptions import DoclingModelDownloadError
from docling.models.vlm_pipeline_models.hf_transformers_model import (
    HuggingFaceTransformersVlmModel,
    InlineVlmOptions,
)


def test_hf_transformers_model_download_failure(monkeypatch, caplog):
    """
    Verify that HuggingFaceTransformersVlmModel handles a failed download correctly.
    - Exception bubbles up
    - Error is logged
    """
    with monkeypatch.context() as m:
        m.setattr(
            "docling.models.vlm_pipeline_models.hf_transformers_model.HuggingFaceTransformersVlmModel.download_models",
            lambda *_, **__: (_ for _ in ()).throw(
                DoclingModelDownloadError("Mock download failure")
            ),
        )
        opts = InlineVlmOptions(
            prompt="test-prompt",
            repo_id="dummy-repo",
            inference_framework="transformers",
            response_format="markdown",
        )
        with pytest.raises(DoclingModelDownloadError):
            HuggingFaceTransformersVlmModel(
                enabled=True,
                artifacts_path=None,
                accelerator_options=AcceleratorOptions(),
                vlm_options=opts,
            )
        assert any(
            "failed to download" in record.message.lower() for record in caplog.records
        ), "Expected a warning/error about the download failure in the logs."
