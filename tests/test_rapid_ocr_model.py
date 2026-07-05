import sys
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

pytestmark = pytest.mark.ml_ocr


@pytest.mark.parametrize(
    ("backend", "det_name", "cls_name", "rec_name", "rec_keys_name"),
    [
        (
            "onnxruntime",
            "PP-OCRv6_det_small.onnx",
            "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
            "PP-OCRv6_rec_small.onnx",
            None,
        ),
        (
            "torch",
            "ch_PP-OCRv4_det_mobile.pth",
            "ch_ptocr_mobile_v2.0_cls_mobile.pth",
            "ch_PP-OCRv4_rec_mobile.pth",
            "paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile/ppocr_keys_v1.txt",
        ),
    ],
)
def test_rapidocr_default_models_use_current_default_assets(
    backend: str,
    det_name: str,
    cls_name: str,
    rec_name: str,
    rec_keys_name: str | None,
):
    model_paths = RapidOcrModel._default_models[backend]

    assert "/v3.9.0/" in model_paths["det_model_path"]["url"]
    assert model_paths["det_model_path"]["path"].endswith(det_name)
    assert model_paths["cls_model_path"]["path"].endswith(cls_name)
    assert model_paths["rec_model_path"]["path"].endswith(rec_name)
    if rec_keys_name is None:
        assert model_paths["rec_keys_path"]["path"] is None
        assert model_paths["rec_keys_path"]["url"] is None
    else:
        assert model_paths["rec_keys_path"]["path"].endswith(rec_keys_name)
    assert model_paths["font_path"]["path"] == "resources/fonts/FZYTK.TTF"

    for detail in model_paths.values():
        if detail["path"] is None or detail["url"] is None:
            continue
        assert "_infer" not in detail["path"]
        assert "_infer" not in detail["url"]


@pytest.mark.parametrize(
    ("backend", "det_name", "cls_name", "rec_name", "rec_keys_name"),
    [
        (
            "onnxruntime",
            "PP-OCRv6_det_small.onnx",
            "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
            "PP-OCRv6_rec_small.onnx",
            None,
        ),
        (
            "torch",
            "ch_PP-OCRv4_det_mobile.pth",
            "ch_ptocr_mobile_v2.0_cls_mobile.pth",
            "ch_PP-OCRv4_rec_mobile.pth",
            "ppocr_keys_v1.txt",
        ),
    ],
)
def test_rapidocr_model_initialization_uses_default_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
    det_name: str,
    cls_name: str,
    rec_name: str,
    rec_keys_name: str | None,
):
    captured: dict[str, object] = {}

    class FakeEngineType(str, Enum):
        ONNXRUNTIME = "onnxruntime"
        OPENVINO = "openvino"
        PADDLE = "paddle"
        TORCH = "torch"

    class FakeRapidOCR:
        def __init__(self, params):
            captured["params"] = params

    monkeypatch.setitem(
        sys.modules,
        "rapidocr",
        SimpleNamespace(EngineType=FakeEngineType, RapidOCR=FakeRapidOCR),
    )

    model_root = tmp_path / RapidOcrModel._model_repo_folder
    for detail in RapidOcrModel._default_models[backend].values():
        if detail["path"] is None:
            continue
        file_path = model_root / detail["path"]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"")

    RapidOcrModel(
        enabled=True,
        artifacts_path=tmp_path,
        options=RapidOcrOptions(backend=backend),
        accelerator_options=AcceleratorOptions(device="cpu", num_threads=1),
    )

    params = captured["params"]
    assert Path(params["Det.model_path"]).name == det_name
    assert Path(params["Cls.model_path"]).name == cls_name
    assert Path(params["Rec.model_path"]).name == rec_name
    if rec_keys_name is None:
        assert params["Rec.rec_keys_path"] is None
    else:
        assert Path(params["Rec.rec_keys_path"]).name == rec_keys_name
    assert Path(params["Global.font_path"]).name == "FZYTK.TTF"


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
    backend: str,
    engine_key: str,
):
    captured: dict[str, object] = {}

    class FakeEngineType(str, Enum):
        ONNXRUNTIME = "onnxruntime"
        OPENVINO = "openvino"
        PADDLE = "paddle"
        TORCH = "torch"

    class FakeRapidOCR:
        def __init__(self, params):
            captured["params"] = params

    monkeypatch.setitem(
        sys.modules,
        "rapidocr",
        SimpleNamespace(EngineType=FakeEngineType, RapidOCR=FakeRapidOCR),
    )

    RapidOcrModel(
        enabled=True,
        artifacts_path=None,
        options=RapidOcrOptions(backend=backend),
        accelerator_options=AcceleratorOptions(device="cpu", num_threads=4),
    )

    # num_threads must reach the engine actually in use, not only ONNXRuntime.
    assert captured["params"][engine_key] == 4


@pytest.mark.parametrize("backend", ["paddle", "torch"])
def test_rapidocr_gpu_device_uses_cuda_ep_cfg_key(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
):
    captured: dict[str, object] = {}

    class FakeEngineType(str, Enum):
        ONNXRUNTIME = "onnxruntime"
        OPENVINO = "openvino"
        PADDLE = "paddle"
        TORCH = "torch"

    class FakeRapidOCR:
        def __init__(self, params):
            captured["params"] = params

    monkeypatch.setitem(
        sys.modules,
        "rapidocr",
        SimpleNamespace(EngineType=FakeEngineType, RapidOCR=FakeRapidOCR),
    )

    RapidOcrModel(
        enabled=True,
        artifacts_path=None,
        options=RapidOcrOptions(backend=backend),
        accelerator_options=AcceleratorOptions(device="cpu"),
    )

    params = captured["params"]
    # The GPU device id must use the engine's real key; the legacy top-level
    # `gpu_id` key is not read by RapidOCR (see #3049 for the torch fix).
    assert f"EngineConfig.{backend}.cuda_ep_cfg.device_id" in params
    assert f"EngineConfig.{backend}.gpu_id" not in params


def test_rapidocr_torch_without_artifacts_uses_ppocrv4_defaults(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    class FakeEngineType(str, Enum):
        ONNXRUNTIME = "onnxruntime"
        OPENVINO = "openvino"
        PADDLE = "paddle"
        TORCH = "torch"

    class FakeRapidOCR:
        def __init__(self, params):
            captured["params"] = params

    class FakeModelType(str, Enum):
        MOBILE = "mobile"
        SERVER = "server"

    class FakeOCRVersion(str, Enum):
        PPOCRV4 = "PP-OCRv4"
        PPOCRV5 = "PP-OCRv5"

    monkeypatch.setitem(
        sys.modules,
        "rapidocr",
        SimpleNamespace(EngineType=FakeEngineType, RapidOCR=FakeRapidOCR),
    )
    monkeypatch.setitem(
        sys.modules,
        "rapidocr.utils.typings",
        SimpleNamespace(ModelType=FakeModelType, OCRVersion=FakeOCRVersion),
    )

    RapidOcrModel(
        enabled=True,
        artifacts_path=None,
        options=RapidOcrOptions(backend="torch"),
        accelerator_options=AcceleratorOptions(device="cpu", num_threads=1),
    )

    params = captured["params"]
    assert params["Det.ocr_version"] == FakeOCRVersion.PPOCRV4
    assert params["Det.model_type"] == FakeModelType.MOBILE
    assert params["Cls.ocr_version"] == FakeOCRVersion.PPOCRV4
    assert params["Cls.model_type"] == FakeModelType.MOBILE
    assert params["Rec.ocr_version"] == FakeOCRVersion.PPOCRV4
    assert params["Rec.model_type"] == FakeModelType.MOBILE
