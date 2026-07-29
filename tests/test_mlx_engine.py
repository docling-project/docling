import pytest

from docling.exceptions import DoclingModelDownloadError
from docling.models.inference_engines.vlm.mlx_engine import (
    MlxVlmEngine,
    MlxVlmEngineOptions,
)


def test_mlx_engine_model_download_failure(monkeypatch, caplog):
    """
    Verify that MLXVlmEngine handles a failed download correctly within _resolve_model_folder
    - Exception bubbles up
    - Error is logged
    """

    with monkeypatch.context() as m:
        m.setattr(
            "docling.models.inference_engines.vlm.mlx_engine.HuggingFaceModelDownloadMixin.download_models",
            lambda *_, **__: (_ for _ in ()).throw(
                DoclingModelDownloadError("Mock download failure")
            ),
        )
        opts = MlxVlmEngineOptions()
        with pytest.raises(DoclingModelDownloadError):
            engine = MlxVlmEngine(
                options=opts,
                artifacts_path=None,
            )
            engine._load_model_for_repo("dummy-repo", "main")
        assert any(
            "failed to download" in record.message.lower() for record in caplog.records
        ), "Expected a warning/error about the download failure in the logs."
