import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
from docling.utils.model_downloader import download_models

pytestmark = pytest.mark.ml_ocr


def _install_fake_rapidocr(
    monkeypatch, captured_params: list[dict[str, object]]
) -> None:
    class FakeRapidOCR:
        def __init__(self, *, params: dict[str, object]) -> None:
            captured_params.append(params)

    fake_module = ModuleType("rapidocr")
    fake_module.EngineType = SimpleNamespace(
        ONNXRUNTIME="onnxruntime",
        OPENVINO="openvino",
        PADDLE="paddle",
        TORCH="torch",
    )
    fake_module.RapidOCR = FakeRapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr", fake_module)


def test_rapidocr_uses_english_default_assets(monkeypatch, tmp_path: Path) -> None:
    captured_params: list[dict[str, object]] = []
    _install_fake_rapidocr(monkeypatch, captured_params)

    RapidOcrModel(
        enabled=True,
        artifacts_path=tmp_path,
        options=RapidOcrOptions(lang=["en"], backend="onnxruntime"),
        accelerator_options=AcceleratorOptions(),
    )

    assert len(captured_params) == 1
    params = captured_params[0]
    assert params["Det.model_path"] == (
        tmp_path / "RapidOcr" / "onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx"
    )
    assert params["Rec.model_path"] == (
        tmp_path / "RapidOcr" / "onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx"
    )
    assert params["Rec.rec_keys_path"] is None


def test_rapidocr_defaults_to_chinese_mobile_assets(
    monkeypatch, tmp_path: Path
) -> None:
    captured_params: list[dict[str, object]] = []
    _install_fake_rapidocr(monkeypatch, captured_params)

    RapidOcrModel(
        enabled=True,
        artifacts_path=tmp_path,
        options=RapidOcrOptions(backend="torch"),
        accelerator_options=AcceleratorOptions(),
    )

    assert len(captured_params) == 1
    params = captured_params[0]
    assert params["Det.model_path"] == (
        tmp_path / "RapidOcr" / "torch/PP-OCRv4/det/ch_PP-OCRv4_det_mobile.pth"
    )
    assert params["Rec.model_path"] == (
        tmp_path / "RapidOcr" / "torch/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile.pth"
    )
    assert params["Rec.rec_keys_path"] == (
        tmp_path
        / "RapidOcr"
        / "paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile/ppocr_keys_v1.txt"
    )


def test_download_models_uses_default_onnx_paths(monkeypatch, tmp_path: Path) -> None:
    downloaded_urls: list[str] = []

    def fake_download_url_with_progress(url: str, *, progress: bool) -> BytesIO:
        del progress
        downloaded_urls.append(url)
        return BytesIO(b"dummy content")

    monkeypatch.setattr(
        "docling.models.stages.ocr.rapid_ocr_model.download_url_with_progress",
        fake_download_url_with_progress,
    )

    RapidOcrModel.download_models(
        local_dir=tmp_path,
        backend="onnxruntime",
        lang="english",
        force=True,
    )

    assert any("PP-OCRv6_det_small.onnx" in url for url in downloaded_urls)
    assert any("PP-OCRv6_rec_small.onnx" in url for url in downloaded_urls)
    assert (tmp_path / "onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx").exists()
    assert (tmp_path / "onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx").exists()


def test_model_downloader_fetches_both_rapidocr_language_sets(
    monkeypatch, tmp_path: Path
) -> None:
    captured_calls: list[dict[str, object]] = []

    def fake_download_models(**kwargs: object) -> None:
        captured_calls.append(kwargs)

    monkeypatch.setattr(RapidOcrModel, "download_models", fake_download_models)
    download_models(
        output_dir=tmp_path,
        with_layout=False,
        with_tableformer=False,
        with_tableformer_v2=False,
        with_code_formula=False,
        with_picture_classifier=False,
        with_smolvlm=False,
        with_granitedocling=False,
        with_granitedocling_mlx=False,
        with_smoldocling=False,
        with_smoldocling_mlx=False,
        with_granite_vision=False,
        with_granite_chart_extraction=False,
        with_granite_chart_extraction_v4=False,
        with_rapidocr=True,
        with_easyocr=False,
    )

    assert len(captured_calls) == 4
    assert {(call["backend"], call["lang"]) for call in captured_calls} == {
        ("torch", "chinese"),
        ("torch", "english"),
        ("onnxruntime", "chinese"),
        ("onnxruntime", "english"),
    }


def test_rapidocr_uses_latin_default_assets(monkeypatch, tmp_path: Path) -> None:
    captured_params: list[dict[str, object]] = []
    _install_fake_rapidocr(monkeypatch, captured_params)

    RapidOcrModel(
        enabled=True,
        artifacts_path=tmp_path,
        options=RapidOcrOptions(lang=["de", "fr"], backend="onnxruntime"),
        accelerator_options=AcceleratorOptions(),
    )

    assert len(captured_params) == 1
    params = captured_params[0]
    assert params["Rec.model_path"] == (
        tmp_path / "RapidOcr" / "onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx"
    )
    assert params["Rec.rec_keys_path"] is None


def test_resolve_language_aliases_and_groups(caplog) -> None:
    from docling.models.stages.ocr.rapid_ocr_model import _resolve_rapidocr_language

    assert _resolve_rapidocr_language(["eng"]) == "english"
    assert _resolve_rapidocr_language(["en-US"]) == "english"
    assert _resolve_rapidocr_language(["deu"]) == "latin"
    assert _resolve_rapidocr_language(["latin"]) == "latin"
    # english + another Latin-script language -> latin covers both
    assert _resolve_rapidocr_language(["en", "de"]) == "latin"
    assert _resolve_rapidocr_language(["zh"]) == "chinese"
    assert _resolve_rapidocr_language(None) == "chinese"


def test_resolve_language_warns_on_silent_fallback(caplog) -> None:
    import logging

    from docling.models.stages.ocr.rapid_ocr_model import _resolve_rapidocr_language

    with caplog.at_level(logging.WARNING):
        resolved = _resolve_rapidocr_language(["klingon"])
    assert resolved == "chinese"
    assert any(
        "no bundled model set" in record.getMessage() for record in caplog.records
    )
    assert any(
        "drops inter-word spaces" in record.getMessage() for record in caplog.records
    )


def test_rapidocr_passes_lang_type_without_artifacts(monkeypatch) -> None:
    captured_params: list[dict[str, object]] = []
    _install_fake_rapidocr(monkeypatch, captured_params)

    fake_typings = ModuleType("rapidocr.utils.typings")
    fake_typings.LangDet = SimpleNamespace(EN="en", CH="ch", MULTI="multi")
    fake_typings.LangRec = SimpleNamespace(EN="en", LATIN="latin", CH="ch")
    fake_utils = ModuleType("rapidocr.utils")
    fake_utils.typings = fake_typings
    monkeypatch.setitem(sys.modules, "rapidocr.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "rapidocr.utils.typings", fake_typings)

    RapidOcrModel(
        enabled=True,
        artifacts_path=None,
        options=RapidOcrOptions(lang=["en"], backend="onnxruntime"),
        accelerator_options=AcceleratorOptions(),
    )

    assert len(captured_params) == 1
    params = captured_params[0]
    assert params["Rec.lang_type"] == "en"
    assert params["Det.lang_type"] == "en"
    # No pinned paths: rapidocr resolves the models itself.
    assert params["Rec.model_path"] is None
