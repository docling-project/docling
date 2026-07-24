from io import BytesIO
from pathlib import Path

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

pytestmark = pytest.mark.ml_ocr


def _capture_params(
    monkeypatch: pytest.MonkeyPatch, options: RapidOcrOptions, artifacts_path: Path
) -> dict[str, object]:
    """Build a RapidOcrModel with real rapidocr resolution but faked inference
    and downloading, returning the params dict handed to RapidOCR."""
    import rapidocr

    captured: dict[str, object] = {}

    class FakeRapidOCR:
        def __init__(self, *, params):
            captured["params"] = params

    monkeypatch.setattr(rapidocr, "RapidOCR", FakeRapidOCR)
    monkeypatch.setattr(
        "docling.models.stages.ocr.rapid_ocr_model.download_url_with_progress",
        lambda url, *, progress: BytesIO(b"dummy content"),
    )

    RapidOcrModel(
        enabled=True,
        artifacts_path=artifacts_path,
        options=options,
        accelerator_options=AcceleratorOptions(device="cpu", num_threads=4),
    )
    return captured["params"]


@pytest.mark.parametrize(
    ("backend", "engine_key"),
    [
        ("onnxruntime", "EngineConfig.onnxruntime.intra_op_num_threads"),
        ("openvino", "EngineConfig.openvino.inference_num_threads"),
        ("paddle", "EngineConfig.paddle.cpu_math_library_num_threads"),
    ],
)
def test_rapidocr_num_threads_propagated_per_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
    engine_key: str,
):
    params = _capture_params(monkeypatch, RapidOcrOptions(backend=backend), tmp_path)
    # num_threads must reach the engine actually in use, not only ONNXRuntime.
    assert params[engine_key] == 4


@pytest.mark.parametrize("backend", ["paddle", "torch"])
def test_rapidocr_gpu_device_uses_cuda_ep_cfg_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
):
    params = _capture_params(monkeypatch, RapidOcrOptions(backend=backend), tmp_path)
    # The GPU device id must use the engine's real key; the legacy top-level
    # `gpu_id` key is not read by RapidOCR (see #3049 for the torch fix).
    assert f"EngineConfig.{backend}.cuda_ep_cfg.device_id" in params
    assert f"EngineConfig.{backend}.gpu_id" not in params


def test_rapidocr_pins_explicit_model_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    params = _capture_params(
        monkeypatch, RapidOcrOptions(backend="onnxruntime"), tmp_path
    )
    # Paths are always pinned now, so rapidocr never lazy-resolves models.
    assert params["Det.model_path"] is not None
    assert params["Rec.model_path"] is not None
    assert "Det.lang_type" not in params
    assert "Rec.lang_type" not in params
