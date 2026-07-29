import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.exceptions import DoclingModelDownloadError
from docling.models import inference_engines
from docling.models.vlm_pipeline_models.mlx_model import (
    HuggingFaceMlxModel,
    InlineVlmOptions,
)


def test_mlx_model_download_failure(monkeypatch, caplog):
    """
    Verify that HuggingFaceMlxModel handles a failed download correctly
    - Exception bubbles up
    - Error is logged
    """
    with monkeypatch.context() as m:
        m.setattr(
            "docling.models.vlm_pipeline_models.mlx_model.HuggingFaceMlxModel.download_models",
            lambda *_, **__: (_ for _ in ()).throw(
                DoclingModelDownloadError("Mock download failure")
            ),
        )
        opts = InlineVlmOptions(
            prompt="test-prompt",
            repo_id="dummy-repo",
            inference_framework="mlx",
            response_format="markdown",
        )
        with pytest.raises(DoclingModelDownloadError):
            HuggingFaceMlxModel(
                enabled=True,
                artifacts_path=None,
                accelerator_options=AcceleratorOptions(),
                vlm_options=opts,
            )

        assert any(
            "failed to download" in record.message.lower() for record in caplog.records
        ), "Expected a warning/error about the download failure in the logs."
