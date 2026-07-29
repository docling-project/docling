import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.exceptions import DoclingModelDownloadError
from docling.models.stages.chart_extraction.granite_vision import (
    ChartExtractionModelGraniteVision,
    ChartExtractionModelGraniteVisionV4,
    ChartExtractionModelOptions,
)


def test_chart_extraction_model_granite_vision_download_failure(monkeypatch, caplog):
    """
    Verify that ChartExtractionModelGraniteVision through _BaseChartExtractionModelGraniteVision
    handles a failed download correctly within initialization
    - Exception bubbles up
    - Error is logged
    """

    with monkeypatch.context() as m:
        m.setattr(
            "docling.models.stages.chart_extraction.granite_vision.download_hf_model",
            lambda *_, **__: (_ for _ in ()).throw(
                DoclingModelDownloadError("Mock download failure")
            ),
        )
        opts = ChartExtractionModelOptions()
        accel = AcceleratorOptions()
        with pytest.raises(DoclingModelDownloadError):
            ChartExtractionModelGraniteVision(
                enabled=True,
                artifacts_path=None,
                options=opts,
                accelerator_options=accel,
            )
        assert any(
            "failed to download" in record.message.lower() for record in caplog.records
        ), "Expected a warning/error about the download failure in the logs."


def test_chart_extraction_model_granite_vision_v4_download_failure(monkeypatch, caplog):
    """
    Verify that ChartExtractionModelGraniteVisionV4 through _BaseChartExtractionModelGraniteVision
    handles a failed download correctly within initialization
    - Exception bubbles up
    - Error is logged
    """

    with monkeypatch.context() as m:
        m.setattr(
            "docling.models.stages.chart_extraction.granite_vision.download_hf_model",
            lambda *_, **__: (_ for _ in ()).throw(
                DoclingModelDownloadError("Mock download failure")
            ),
        )
        opts = ChartExtractionModelOptions()
        accel = AcceleratorOptions()
        with pytest.raises(DoclingModelDownloadError):
            ChartExtractionModelGraniteVisionV4(
                enabled=True,
                artifacts_path=None,
                options=opts,
                accelerator_options=accel,
            )
        assert any(
            "failed to download" in record.message.lower() for record in caplog.records
        ), "Expected a warning/error about the download failure in the logs."
