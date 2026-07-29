import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.exceptions import DoclingModelDownloadError
from docling.models.stages.code_formula.code_formula_model import (
    CodeFormulaModel,
    CodeFormulaModelOptions,
)


def test_table_structure_model_download_failure(monkeypatch, caplog):
    """
    Verify that CodeFormulaModel handles a failed download correctly within initialization
    - Exception bubbles up
    - Error is logged
    """

    with monkeypatch.context() as m:
        m.setattr(
            "docling.models.stages.code_formula.code_formula_model.download_hf_model",
            lambda *_, **__: (_ for _ in ()).throw(
                DoclingModelDownloadError("Mock download failure")
            ),
        )
        opts = CodeFormulaModelOptions()
        accel = AcceleratorOptions()
        with pytest.raises(DoclingModelDownloadError):
            CodeFormulaModel(
                artifacts_path=None,
                enabled=True,
                accelerator_options=accel,
                options=opts,
            )
        assert any(
            "failed to download" in record.message.lower() for record in caplog.records
        ), "Expected a warning/error about the download failure in the logs."
