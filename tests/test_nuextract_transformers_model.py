import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.exceptions import DoclingModelDownloadError
from docling.models.extraction.nuextract_transformers_model import (
    InlineVlmOptions,
    NuExtractTransformersModel,
)


def test_nuextract_transformers_model_download_failure(monkeypatch, caplog):
    """
    Verify that NuExtractTransformersModel handles a failed download correctly within initialization
    - Exception bubbles up
    - Error is logged
    """

    with monkeypatch.context() as m:
        m.setattr(
            "docling.models.extraction.nuextract_transformers_model.download_hf_model",
            lambda *_, **__: (_ for _ in ()).throw(
                DoclingModelDownloadError("Mock download failure")
            ),
        )
        opts = InlineVlmOptions()
        accel = AcceleratorOptions()
        with pytest.raises(DoclingModelDownloadError):
            NuExtractTransformersModel(
                artifacts_path=None,
                enabled=True,
                accelerator_options=accel,
                vlm_options=opts,
            )
        assert any(
            "failed to download" in record.message.lower() for record in caplog.records
        ), "Expected a warning/error about the download failure in the logs."
