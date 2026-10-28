# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO
from pathlib import Path

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

pytestmark = pytest.mark.ml_ocr


def _capture_params(
    monkeypatch: pytest.MonkeyPatch,
    options: RapidOcrOptions,
    artifacts_path: Path,
    resolved_device: str = "cpu",
) -> dict[str, object]:
    """Build a RapidOcrModel with real rapidocr resolution but faked inference
    and downloading, returning the params dict handed to RapidOCR.

    artifacts_path is strictly offline, so the checkpoints are prefetched first.
    """
    import rapidocr

    captured: dict[str, object] = {}

    class FakeRapidOCR:
        def __init__(self, *, params):
            captured["params"] = params

    monkeypatch.setattr(rapidocr, "RapidOCR", FakeRapidOCR)
    monkeypatch.setattr(
        "docling.models.stages.ocr.rapid_ocr_model.decide_device",
        lambda device: resolved_device,
    )
    monkeypatch.setattr(
        "docling.models.stages.ocr.rapid_ocr_model.download_url_with_progress",
        lambda url, *, progress: BytesIO(b"dummy content"),
    )
    RapidOcrModel.download_models(
        backend=options.backend,
        lang=options.lang[0],
        local_dir=artifacts_path / RapidOcrModel._model_repo_folder,
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


@pytest.mark.parametrize("backend", ["onnxruntime", "paddle", "torch"])
def test_rapidocr_gpu_device_uses_cuda_ep_cfg_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
):
    params = _capture_params(
        monkeypatch,
        RapidOcrOptions(backend=backend),
        tmp_path,
        resolved_device="cuda:2",
    )
    # The GPU device id must use the engine's real key; the legacy top-level
    # `gpu_id` key is not read by RapidOCR (see #3049 for the torch fix).
    assert f"EngineConfig.{backend}.cuda_ep_cfg.device_id" in params
    assert params[f"EngineConfig.{backend}.cuda_ep_cfg.device_id"] == 2
    assert params[f"EngineConfig.{backend}.use_cuda"] is True
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


def test_rapidocr_onnxruntime_skips_the_cudnn_algorithm_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    params = _capture_params(
        monkeypatch,
        RapidOcrOptions(backend="onnxruntime"),
        tmp_path,
        resolved_device="cuda:0",
    )
    # RapidOCR defaults to EXHAUSTIVE, which re-searches convolution algorithms for
    # every new input shape. Recognition feeds one text-line crop at a time, so the
    # shape changes constantly and the search costs more than the inference it picks
    # kernels for -- enough to make the GPU slower than the CPU.
    assert (
        params["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search"]
        == "DEFAULT"
    )


def test_rapidocr_onnxruntime_engine_receives_the_dml_choice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    params = _capture_params(
        monkeypatch, RapidOcrOptions(backend="onnxruntime"), tmp_path
    )
    # DirectML is selected from the engine config too, not from the per-model entries.
    assert params["EngineConfig.onnxruntime.use_dml"] == params["Det.use_dml"]
