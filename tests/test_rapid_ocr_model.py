import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from rapidocr import EngineType as RapidEngineType
from rapidocr.utils.typings import (
    LangCls,
    LangDet,
    LangRec,
    ModelType,
    OCRVersion,
    TaskType,
)

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel


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


def _install_fake_rapidocr_package(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, object]
) -> None:
    class FakeRapidOCR:
        def __init__(self, params: dict[str, object]) -> None:
            for key, value in params.items():
                _, _, option = key.partition(".")
                if option in {
                    "engine_type",
                    "lang_type",
                    "model_type",
                    "ocr_version",
                    "task_type",
                } and not isinstance(value, Enum):
                    raise TypeError(f"The value of {key} must be Enum Type.")

            captured["params"] = params

    fake_module = ModuleType("rapidocr")
    fake_module.__path__ = []
    fake_module.EngineType = RapidEngineType
    fake_module.RapidOCR = FakeRapidOCR

    fake_utils_module = ModuleType("rapidocr.utils")
    fake_typings_module = ModuleType("rapidocr.utils.typings")
    fake_typings_module.LangCls = LangCls
    fake_typings_module.LangDet = LangDet
    fake_typings_module.LangRec = LangRec
    fake_typings_module.ModelType = ModelType
    fake_typings_module.OCRVersion = OCRVersion
    fake_typings_module.TaskType = TaskType
    fake_utils_module.typings = fake_typings_module
    fake_module.utils = fake_utils_module

    monkeypatch.setitem(sys.modules, "rapidocr", fake_module)
    monkeypatch.setitem(sys.modules, "rapidocr.utils", fake_utils_module)
    monkeypatch.setitem(sys.modules, "rapidocr.utils.typings", fake_typings_module)


@pytest.mark.parametrize(
    ("param_key", "param_value", "expected_value"),
    [
        ("Det.lang_type", "ch", LangDet.CH),
        ("Cls.lang_type", "ch", LangCls.CH),
        ("Rec.lang_type", "en", LangRec.EN),
        ("Det.model_type", "server", ModelType.SERVER),
        ("Cls.model_type", "mobile", ModelType.MOBILE),
        ("Rec.model_type", "server", ModelType.SERVER),
        ("Det.ocr_version", "PP-OCRv5", OCRVersion.PPOCRV5),
        ("Cls.ocr_version", "PP-OCRv4", OCRVersion.PPOCRV4),
        ("Rec.ocr_version", "PP-OCRv5", OCRVersion.PPOCRV5),
        ("Det.engine_type", "onnxruntime", RapidEngineType.ONNXRUNTIME),
        ("Cls.engine_type", "torch", RapidEngineType.TORCH),
        ("Rec.task_type", "rec", TaskType.REC),
    ],
)
def test_rapidocr_param_normalization_converts_known_enum_strings(
    param_key: str,
    param_value: str,
    expected_value: Enum,
) -> None:
    normalized = RapidOcrModel._normalize_rapidocr_params(
        {
            param_key: param_value,
            "Global.text_score": 0.75,
        }
    )

    assert normalized[param_key] is expected_value
    assert normalized["Global.text_score"] == 0.75


def test_rapidocr_param_normalization_preserves_enum_instances_and_other_keys() -> None:
    normalized = RapidOcrModel._normalize_rapidocr_params(
        {
            "Det.lang_type": LangDet.EN,
            "Cls.model_type": "mobile",
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.task_type": TaskType.REC,
            "Global.text_score": 0.9,
            "Rec.rec_batch_num": 8,
        }
    )

    assert normalized["Det.lang_type"] is LangDet.EN
    assert normalized["Cls.model_type"] is ModelType.MOBILE
    assert normalized["Rec.ocr_version"] is OCRVersion.PPOCRV5
    assert normalized["Rec.task_type"] is TaskType.REC
    assert normalized["Global.text_score"] == 0.9
    assert normalized["Rec.rec_batch_num"] == 8


def test_rapidocr_model_initialization_normalizes_string_enum_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_rapidocr_package(monkeypatch, captured)

    model_root = tmp_path / RapidOcrModel._model_repo_folder
    for detail in RapidOcrModel._default_models["onnxruntime"].values():
        file_path = model_root / detail["path"]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"")

    RapidOcrModel(
        enabled=True,
        artifacts_path=tmp_path,
        options=RapidOcrOptions(
            backend="onnxruntime",
            rapidocr_params={
                "Det.lang_type": "ch",
                "Cls.model_type": "mobile",
                "Rec.ocr_version": "PP-OCRv5",
                "Rec.task_type": "rec",
                "Rec.engine_type": "onnxruntime",
            },
        ),
        accelerator_options=AcceleratorOptions(device="cpu", num_threads=1),
    )

    params = captured["params"]
    assert params["Det.lang_type"] is LangDet.CH
    assert params["Cls.model_type"] is ModelType.MOBILE
    assert params["Rec.ocr_version"] is OCRVersion.PPOCRV5
    assert params["Rec.task_type"] is TaskType.REC
    assert params["Rec.engine_type"] is RapidEngineType.ONNXRUNTIME
