import logging
from io import BytesIO
from pathlib import Path

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.datamodel.settings import settings
from docling.models.stages.ocr.rapid_ocr_model import (
    RapidOcrModel,
    _resolve_rapidocr,
)
from docling.utils.model_downloader import download_models

pytestmark = pytest.mark.ml_ocr


def _install_fakes(monkeypatch, captured_params: list[dict[str, object]]) -> list[str]:
    """Fake only inference + downloading; keep rapidocr's real model registry.

    Returns the list that will collect every downloaded URL.
    """
    import rapidocr

    class FakeRapidOCR:
        def __init__(self, *, params: dict[str, object]) -> None:
            captured_params.append(params)

    monkeypatch.setattr(rapidocr, "RapidOCR", FakeRapidOCR)

    downloaded_urls: list[str] = []

    def fake_download_url_with_progress(url: str, *, progress: bool) -> BytesIO:
        del progress
        downloaded_urls.append(url)
        return BytesIO(b"dummy content")

    monkeypatch.setattr(
        "docling.models.stages.ocr.rapid_ocr_model.download_url_with_progress",
        fake_download_url_with_progress,
    )
    return downloaded_urls


def _build(monkeypatch, options: RapidOcrOptions, artifacts_path: Path | None):
    captured_params: list[dict[str, object]] = []
    downloaded = _install_fakes(monkeypatch, captured_params)
    RapidOcrModel(
        enabled=True,
        artifacts_path=artifacts_path,
        options=options,
        accelerator_options=AcceleratorOptions(),
    )
    assert len(captured_params) == 1
    return captured_params[0], downloaded


# --- resolution -------------------------------------------------------------


def test_resolve_defaults_to_ppocrv6_chinese() -> None:
    from rapidocr.utils.typings import OCRVersion

    assert _resolve_rapidocr([], "onnxruntime") == (OCRVersion.PPOCRV6, "ch")
    assert _resolve_rapidocr(["chinese"], "onnxruntime") == (
        OCRVersion.PPOCRV6,
        "ch",
    )
    assert _resolve_rapidocr(["zh"], "onnxruntime") == (OCRVersion.PPOCRV6, "ch")


def test_resolve_english_and_latin_use_ppocrv6() -> None:
    from rapidocr.utils.typings import OCRVersion

    assert _resolve_rapidocr(["english"], "onnxruntime") == (
        OCRVersion.PPOCRV6,
        "en",
    )
    assert _resolve_rapidocr(["en"], "torch") == (OCRVersion.PPOCRV6, "en")
    assert _resolve_rapidocr(["de"], "onnxruntime") == (OCRVersion.PPOCRV6, "de")
    assert _resolve_rapidocr(["fr"], "onnxruntime") == (OCRVersion.PPOCRV6, "fr")


def test_resolve_script_families_route_by_backend() -> None:
    from rapidocr.utils.typings import OCRVersion

    # onnxruntime/openvino/paddle -> PP-OCRv5
    assert _resolve_rapidocr(["th"], "onnxruntime") == (OCRVersion.PPOCRV5, "th")
    assert _resolve_rapidocr(["cyrillic"], "onnxruntime") == (
        OCRVersion.PPOCRV5,
        "cyrillic",
    )
    # torch -> PP-OCRv4
    assert _resolve_rapidocr(["arabic"], "torch") == (OCRVersion.PPOCRV4, "arabic")


def test_resolve_raises_on_unsupported_language() -> None:
    with pytest.raises(ValueError):
        _resolve_rapidocr(["klingon"], "onnxruntime")
    # Thai is a PP-OCRv5 language, not served by the torch PP-OCRv4 backbone.
    with pytest.raises(ValueError):
        _resolve_rapidocr(["th"], "torch")


def test_resolve_multiple_languages_warns_and_uses_first(caplog) -> None:
    from rapidocr.utils.typings import OCRVersion

    with caplog.at_level(logging.WARNING):
        resolved = _resolve_rapidocr(["de", "fr"], "onnxruntime")
    assert resolved == (OCRVersion.PPOCRV6, "de")
    assert any("single language" in r.getMessage() for r in caplog.records)


# --- model selection / pinned paths -----------------------------------------


def test_rapidocr_default_onnx_uses_ppocrv6(monkeypatch, tmp_path: Path) -> None:
    params, _ = _build(
        monkeypatch,
        RapidOcrOptions(lang=["en"], backend="onnxruntime"),
        tmp_path,
    )
    assert Path(params["Det.model_path"]).name == "PP-OCRv6_det_small.onnx"
    assert Path(params["Rec.model_path"]).name == "PP-OCRv6_rec_small.onnx"
    # onnx v6 embeds its charset -> no separate keys file.
    assert params["Rec.rec_keys_path"] is None
    # everything lands under the docling artifacts folder.
    assert str(params["Rec.model_path"]).startswith(str(tmp_path / "RapidOcr"))


def test_rapidocr_default_torch_uses_ppocrv6(monkeypatch, tmp_path: Path) -> None:
    params, _ = _build(
        monkeypatch,
        RapidOcrOptions(backend="torch"),  # default lang -> chinese -> ch -> v6
        tmp_path,
    )
    assert Path(params["Det.model_path"]).name == "PP-OCRv6_det_small.pth"
    assert Path(params["Rec.model_path"]).name == "PP-OCRv6_rec_small.pth"
    # torch rec ships a dict_url, so the keys file is resolved and downloaded.
    assert params["Rec.rec_keys_path"] is not None
    assert Path(params["Rec.rec_keys_path"]).exists()


def test_rapidocr_latin_language_uses_ppocrv6(monkeypatch, tmp_path: Path) -> None:
    params, _ = _build(
        monkeypatch,
        RapidOcrOptions(lang=["de", "fr"], backend="onnxruntime"),
        tmp_path,
    )
    assert Path(params["Rec.model_path"]).name == "PP-OCRv6_rec_small.onnx"
    assert params["Rec.rec_keys_path"] is None


def test_rapidocr_thai_uses_ppocrv5(monkeypatch, tmp_path: Path) -> None:
    params, _ = _build(
        monkeypatch,
        RapidOcrOptions(lang=["th"], backend="onnxruntime"),
        tmp_path,
    )
    assert Path(params["Det.model_path"]).name == "ch_PP-OCRv5_det_mobile.onnx"
    assert Path(params["Rec.model_path"]).name == "th_PP-OCRv5_rec_mobile.onnx"


def test_rapidocr_arabic_torch_uses_ppocrv4(monkeypatch, tmp_path: Path) -> None:
    params, _ = _build(
        monkeypatch,
        RapidOcrOptions(lang=["arabic"], backend="torch"),
        tmp_path,
    )
    assert Path(params["Rec.model_path"]).name == "arabic_PP-OCRv4_rec_mobile.pth"
    # v4 rec ships a character dictionary.
    assert params["Rec.rec_keys_path"] is not None


def test_rapidocr_unsupported_language_raises(monkeypatch, tmp_path: Path) -> None:
    captured_params: list[dict[str, object]] = []
    _install_fakes(monkeypatch, captured_params)
    with pytest.raises(ValueError):
        RapidOcrModel(
            enabled=True,
            artifacts_path=tmp_path,
            options=RapidOcrOptions(lang=["klingon"], backend="onnxruntime"),
            accelerator_options=AcceleratorOptions(),
        )


def test_rapidocr_no_artifacts_downloads_to_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    params, _ = _build(
        monkeypatch,
        RapidOcrOptions(lang=["en"], backend="onnxruntime"),
        None,
    )
    expected = tmp_path / "models" / "RapidOcr"
    assert str(params["Rec.model_path"]).startswith(str(expected))


def test_rapidocr_explicit_paths_skip_resolution(monkeypatch, tmp_path: Path) -> None:
    det = tmp_path / "custom_det.onnx"
    rec = tmp_path / "custom_rec.onnx"
    det.write_bytes(b"x")
    rec.write_bytes(b"x")
    params, downloaded = _build(
        monkeypatch,
        RapidOcrOptions(
            lang=["en"],
            backend="onnxruntime",
            det_model_path=str(det),
            rec_model_path=str(rec),
        ),
        tmp_path,
    )
    assert params["Det.model_path"] == str(det)
    assert params["Rec.model_path"] == str(rec)
    # user pinned det+rec -> no auto-download happens.
    assert downloaded == []


# --- download_models / prefetch ---------------------------------------------


def test_download_models_downloads_ppocrv6(monkeypatch, tmp_path: Path) -> None:
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
        force=True,
    )

    assert any("PP-OCRv6_det_small.onnx" in url for url in downloaded_urls)
    assert any("PP-OCRv6_rec_small.onnx" in url for url in downloaded_urls)
    assert (tmp_path / "PP-OCRv6_det_small.onnx").exists()
    assert (tmp_path / "PP-OCRv6_rec_small.onnx").exists()


def test_model_downloader_fetches_rapidocr_per_backend(
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

    assert len(captured_calls) == 2
    assert {call["backend"] for call in captured_calls} == {"torch", "onnxruntime"}
    # No per-language loop anymore.
    assert all("lang" not in call for call in captured_calls)
