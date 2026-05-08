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
    ("backend", "det_name", "cls_name", "rec_name"),
    [
        (
            "onnxruntime",
            "ch_PP-OCRv4_det_mobile.onnx",
            "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
            "ch_PP-OCRv4_rec_mobile.onnx",
        ),
        (
            "torch",
            "ch_PP-OCRv4_det_mobile.pth",
            "ch_ptocr_mobile_v2.0_cls_mobile.pth",
            "ch_PP-OCRv4_rec_mobile.pth",
        ),
    ],
)
def test_rapidocr_default_models_use_3_8_mobile_assets(
    backend: str,
    det_name: str,
    cls_name: str,
    rec_name: str,
):
    model_paths = RapidOcrModel._default_models[backend]

    assert "/v3.8.0/" in model_paths["det_model_path"]["url"]
    assert model_paths["det_model_path"]["path"].endswith(det_name)
    assert model_paths["cls_model_path"]["path"].endswith(cls_name)
    assert model_paths["rec_model_path"]["path"].endswith(rec_name)
    assert model_paths["rec_keys_path"]["path"].endswith(
        "paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile/ppocr_keys_v1.txt"
    )
    assert model_paths["font_path"]["path"] == "resources/fonts/FZYTK.TTF"

    for detail in model_paths.values():
        assert "_infer" not in detail["path"]
        assert "_infer" not in detail["url"]


@pytest.mark.parametrize(
    ("backend", "det_name", "cls_name", "rec_name"),
    [
        (
            "onnxruntime",
            "ch_PP-OCRv4_det_mobile.onnx",
            "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
            "ch_PP-OCRv4_rec_mobile.onnx",
        ),
        (
            "torch",
            "ch_PP-OCRv4_det_mobile.pth",
            "ch_ptocr_mobile_v2.0_cls_mobile.pth",
            "ch_PP-OCRv4_rec_mobile.pth",
        ),
    ],
)
def test_rapidocr_model_initialization_uses_mobile_default_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
    det_name: str,
    cls_name: str,
    rec_name: str,
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
    assert Path(params["Rec.rec_keys_path"]).name == "ppocr_keys_v1.txt"
    assert Path(params["Global.font_path"]).name == "FZYTK.TTF"
