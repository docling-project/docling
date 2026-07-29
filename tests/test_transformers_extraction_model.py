import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.exceptions import DoclingModelDownloadError
from docling.models.extraction.transformers_extraction_model import (
    InlineVlmOptions,
    TransformersExtractionModel,
)


def test_transformers_extraction_model_download_failure(monkeypatch, caplog):
    """
    Verify that TransformersExtractionModel handles a failed download correctly within initialization
    - Exception bubbles up
    - Error is logged
    """
    with monkeypatch.context() as m:
        m.setattr(
            "docling.models.extraction.transformers_extraction_model.download_hf_model",
            lambda *_, **__: (_ for _ in ()).throw(
                DoclingModelDownloadError("Mock Failure")
            ),
        )
        opts = InlineVlmOptions()
        accel = AcceleratorOptions()
        with pytest.raises(DoclingModelDownloadError):
            TransformersExtractionModel(
                enabled=True,
                vlm_options=opts,
                accelerator_options=accel,
                artifacts_path=None,
            )
        assert any(
            "failed to download" in record.message.lower() for record in caplog.records
        ), "Expected a warning/error about the download failure in the logs."
