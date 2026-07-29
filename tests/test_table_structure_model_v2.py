import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.exceptions import DoclingModelDownloadError
from docling.models.stages.table_structure.table_structure_model_v2 import (
    TableStructureModelV2,
    TableStructureV2Options,
)


def test_table_structure_model_v2_download_failure(monkeypatch, caplog):
    """
    Verify that TableStructureModelV2 handles a failed download correctly within initialization
    - Exception bubbles up
    - Error is logged
    """

    with monkeypatch.context() as m:
        m.setattr(
            "docling.models.stages.table_structure.table_structure_model_v2.download_hf_model",
            lambda *_, **__: (_ for _ in ()).throw(
                DoclingModelDownloadError("Mock download failure")
            ),
        )
        opts = TableStructureV2Options()
        accel = AcceleratorOptions()
        with pytest.raises(DoclingModelDownloadError):
            TableStructureModelV2(
                enabled=True,
                artifacts_path=None,
                options=opts,
                accelerator_options=accel,
            )
        assert any(
            "failed to download" in record.message.lower() for record in caplog.records
        ), "Expected a warning/error about the download failure in the logs."
