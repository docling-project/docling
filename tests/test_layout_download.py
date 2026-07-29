import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.exceptions import DoclingModelDownloadError
from docling.models.stages.layout.layout_model import LayoutModel, LayoutOptions


def test_layout_model_download_failure(monkeypatch, caplog):
    """
    Verify that LayoutModel handles a failed download correctly within initialization
    - Exception bubbles up
    - Error is logged
    """

    with monkeypatch.context() as m:
        m.setattr(
            "docling.models.stages.layout.layout_model.download_hf_model",
            lambda *_, **__: (_ for _ in ()).throw(
                DoclingModelDownloadError("Mock download failure")
            ),
        )
        opts = LayoutOptions()
        accel = AcceleratorOptions()
        with pytest.raises(DoclingModelDownloadError):
            LayoutModel(artifacts_path=None, options=opts, accelerator_options=accel)

        assert any(
            "failed to download" in record.message.lower() for record in caplog.records
        ), "Expected a warning/error about the download failure in the logs."
