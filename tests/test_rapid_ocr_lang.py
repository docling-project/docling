# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.datamodel.settings import settings
from docling.exceptions import OcrLanguageNotSupportedError
from docling.models.stages.ocr.ppocr_languages import ppocr_supported_tags
from docling.models.stages.ocr.rapid_ocr_model import (
    RapidOcrModel,
    _parse_rapidocr_model_spec,
    _rapidocr_vocabulary,
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


def _seed(artifacts_path: Path, backend: str, lang: str) -> None:
    """Prefetch one `(backend, lang)` set into artifacts_path, the way a user would."""
    RapidOcrModel.download_models(
        backend=backend,
        lang=lang,
        local_dir=artifacts_path / RapidOcrModel._model_repo_folder,
    )


def _build(
    monkeypatch,
    options: RapidOcrOptions,
    artifacts_path: Path | None,
    *,
    seed: tuple[str, str] | None = None,
):
    captured_params: list[dict[str, object]] = []
    downloaded = _install_fakes(monkeypatch, captured_params)
    if seed is not None:
        assert artifacts_path is not None
        _seed(artifacts_path, *seed)
        # Prefetching is the setup step; only what the model itself fetches is under test.
        downloaded.clear()
    RapidOcrModel(
        enabled=True,
        artifacts_path=artifacts_path,
        options=options,
        accelerator_options=AcceleratorOptions(),
    )
    assert len(captured_params) == 1
    return captured_params[0], downloaded


# --- resolution -------------------------------------------------------------


def _resolved(lang: str, backend: str):
    """The (version, registry code) pair the assertions below care about."""
    spec = _resolve_rapidocr(lang, backend)
    return spec.ppocr_version, spec.rapidocr_code


def test_resolve_populates_the_whole_spec() -> None:
    from rapidocr.utils.typings import OCRVersion

    spec = _resolve_rapidocr("zh", "onnxruntime")
    assert spec.backend == "onnxruntime"
    # The user's spelling is preserved verbatim, the registry code is normalized.
    assert spec.user_lang == "zh"
    assert spec.rapidocr_code == "ch"
    assert spec.ppocr_version == OCRVersion.PPOCRV6


def test_resolve_defaults_to_ppocrv6_chinese() -> None:
    from rapidocr.utils.typings import OCRVersion

    assert _resolved("zh-Hans", "onnxruntime") == (OCRVersion.PPOCRV6, "ch")
    assert _resolved("zh", "onnxruntime") == (OCRVersion.PPOCRV6, "ch")


def test_resolve_english_and_latin_use_ppocrv6() -> None:
    from rapidocr.utils.typings import OCRVersion

    assert _resolved("en", "onnxruntime") == (OCRVersion.PPOCRV6, "en")
    assert _resolved("en", "torch") == (OCRVersion.PPOCRV6, "en")
    assert _resolved("de", "onnxruntime") == (OCRVersion.PPOCRV6, "de")
    assert _resolved("fr", "onnxruntime") == (OCRVersion.PPOCRV6, "fr")


def test_resolve_script_families_route_by_backend() -> None:
    from rapidocr.utils.typings import OCRVersion

    # onnxruntime/openvino/paddle -> PP-OCRv5
    assert _resolved("th", "onnxruntime") == (OCRVersion.PPOCRV5, "th")
    assert _resolved("native:cyrillic", "onnxruntime") == (
        OCRVersion.PPOCRV5,
        "cyrillic",
    )
    # torch -> PP-OCRv4
    assert _resolved("native:arabic", "torch") == (OCRVersion.PPOCRV4, "arabic")
    # Devanagari picks the backbone its backend can reach.
    assert _resolved("hi", "onnxruntime") == (OCRVersion.PPOCRV5, "devanagari")
    assert _resolved("hi", "torch") == (OCRVersion.PPOCRV4, "devanagari")


def test_resolve_rejects_a_malformed_tag() -> None:
    with pytest.raises(ValueError, match="BCP-47"):
        _resolve_rapidocr("klingon", "onnxruntime")


def test_resolve_raises_on_unsupported_language() -> None:
    # Thai is a PP-OCRv5 language, not served by the torch PP-OCRv4 backbone.
    with pytest.raises(OcrLanguageNotSupportedError):
        _resolve_rapidocr("th", "torch")
    # PP-OCR has no Georgian recognizer; its `ka` is Kannada.
    with pytest.raises(OcrLanguageNotSupportedError):
        _resolve_rapidocr("ka-Geor", "onnxruntime")


@pytest.mark.parametrize("backend", ["onnxruntime", "openvino", "paddle", "torch"])
def test_resolve_kannada_falls_back_to_ppocrv4_on_every_backend(backend: str) -> None:
    from rapidocr.utils.typings import OCRVersion

    # PP-OCR serves Kannada only on the v4 backbone, so every backend has to
    # reach past its own v5/v6 set for it -- `ka` is the one code v5 lacks.
    assert _resolved("kn", backend) == (OCRVersion.PPOCRV4, "ka")
    # ...and it is advertised, so the coverage error never names it.
    assert "kn-Knda" in ppocr_supported_tags(_rapidocr_vocabulary(backend))


# --- model selection / pinned paths -----------------------------------------


def test_rapidocr_default_onnx_uses_ppocrv6(monkeypatch, tmp_path: Path) -> None:
    params, downloaded = _build(
        monkeypatch,
        RapidOcrOptions(lang=["en"], backend="onnxruntime"),
        tmp_path,
        seed=("onnxruntime", "en"),
    )
    assert Path(params["Det.model_path"]).name == "PP-OCRv6_det_small.onnx"
    assert Path(params["Rec.model_path"]).name == "PP-OCRv6_rec_small.onnx"
    # onnx v6 embeds its charset -> no separate keys file.
    assert params["Rec.rec_keys_path"] is None
    # everything lands under the docling artifacts folder.
    assert str(params["Rec.model_path"]).startswith(str(tmp_path / "RapidOcr"))
    # artifacts_path means offline: the prefetched files are used as-is.
    assert downloaded == []


def test_rapidocr_default_torch_uses_ppocrv6(monkeypatch, tmp_path: Path) -> None:
    params, downloaded = _build(
        monkeypatch,
        RapidOcrOptions(backend="torch"),  # default lang -> zh-Hans -> ch -> v6
        tmp_path,
        seed=("torch", "zh-Hans"),
    )
    assert Path(params["Det.model_path"]).name == "PP-OCRv6_det_small.pth"
    assert Path(params["Rec.model_path"]).name == "PP-OCRv6_rec_small.pth"
    # torch rec ships a dict_url, so the keys file is resolved alongside the model.
    assert params["Rec.rec_keys_path"] is not None
    assert Path(params["Rec.rec_keys_path"]).exists()
    assert downloaded == []


def test_rapidocr_latin_language_uses_ppocrv6(monkeypatch, tmp_path: Path) -> None:
    params, _ = _build(
        monkeypatch,
        RapidOcrOptions(lang=["de", "fr"], backend="onnxruntime"),
        tmp_path,
        seed=("onnxruntime", "de"),
    )
    assert Path(params["Rec.model_path"]).name == "PP-OCRv6_rec_small.onnx"
    assert params["Rec.rec_keys_path"] is None


def test_rapidocr_thai_uses_ppocrv5(monkeypatch, tmp_path: Path) -> None:
    params, _ = _build(
        monkeypatch,
        RapidOcrOptions(lang=["th"], backend="onnxruntime"),
        tmp_path,
        seed=("onnxruntime", "th"),
    )
    assert Path(params["Det.model_path"]).name == "ch_PP-OCRv5_det_mobile.onnx"
    assert Path(params["Rec.model_path"]).name == "th_PP-OCRv5_rec_mobile.onnx"


def test_rapidocr_arabic_torch_uses_ppocrv4(monkeypatch, tmp_path: Path) -> None:
    params, _ = _build(
        monkeypatch,
        RapidOcrOptions(lang=["native:arabic"], backend="torch"),
        tmp_path,
        seed=("torch", "native:arabic"),
    )
    assert Path(params["Rec.model_path"]).name == "arabic_PP-OCRv4_rec_mobile.pth"
    # v4 rec ships a character dictionary.
    assert params["Rec.rec_keys_path"] is not None


def test_rapidocr_malformed_language_raises_at_options_time() -> None:
    """A typo never reaches the model: the options validator rejects it."""
    with pytest.raises(ValidationError, match="BCP-47"):
        RapidOcrOptions(lang=["klingon"], backend="onnxruntime")


def test_rapidocr_unsupported_language_raises(monkeypatch, tmp_path: Path) -> None:
    captured_params: list[dict[str, object]] = []
    _install_fakes(monkeypatch, captured_params)
    # Georgian must be spelled out: a bare `ka` given to RapidOCR is PP-OCR's own
    # code for Kannada.
    with pytest.raises(OcrLanguageNotSupportedError, match="ka-Geor"):
        RapidOcrModel(
            enabled=True,
            artifacts_path=tmp_path,
            options=RapidOcrOptions(lang=["ka-Geor"], backend="onnxruntime"),
            accelerator_options=AcceleratorOptions(),
        )


def test_rapidocr_no_artifacts_uses_library_params(monkeypatch, tmp_path: Path) -> None:
    from rapidocr.utils.typings import OCRVersion

    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    params, downloaded = _build(
        monkeypatch,
        RapidOcrOptions(lang=["en"], backend="onnxruntime"),
        None,
    )
    # Without artifacts_path docling downloads nothing; RapidOCR serves the
    # checkpoints bundled in its package (and its own cache).
    assert downloaded == []
    assert not (tmp_path / "models" / "RapidOcr").exists()
    # Model paths stay unset; the resolved version/language is forwarded instead.
    assert params["Det.model_path"] is None
    assert params["Rec.model_path"] is None
    assert params["Rec.ocr_version"] == OCRVersion.PPOCRV6
    assert params["Rec.lang_type"] == "en"


def test_rapidocr_pinned_paths_skip_download(monkeypatch, tmp_path: Path) -> None:
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
        None,
    )
    assert params["Det.model_path"] == str(det)
    assert params["Rec.model_path"] == str(rec)
    # Pinned det+rec, no artifacts_path -> nothing downloaded; cls is left to
    # RapidOCR via library params (per-model independence).
    assert downloaded == []
    assert "Det.ocr_version" not in params
    assert "Rec.ocr_version" not in params
    assert "Cls.ocr_version" in params


def test_rapidocr_artifacts_pinned_det_rec_still_requires_cls(
    monkeypatch, tmp_path: Path
) -> None:
    det = tmp_path / "custom_det.onnx"
    rec = tmp_path / "custom_rec.onnx"
    det.write_bytes(b"x")
    rec.write_bytes(b"x")
    options = RapidOcrOptions(
        lang=["en"],
        backend="onnxruntime",
        det_model_path=str(det),
        rec_model_path=str(rec),
    )
    # cls is not pinned, so it must be present in the artifacts folder even though
    # det and rec are (this is the asymmetry that used to be silently skipped).
    _install_fakes(monkeypatch, [])
    with pytest.raises(FileNotFoundError, match="cls"):
        RapidOcrModel(
            enabled=True,
            artifacts_path=tmp_path,
            options=options,
            accelerator_options=AcceleratorOptions(),
        )

    params, downloaded = _build(
        monkeypatch, options, tmp_path, seed=("onnxruntime", "en")
    )
    # Pinned det/rec are kept verbatim...
    assert params["Det.model_path"] == str(det)
    assert params["Rec.model_path"] == str(rec)
    # ...and cls resolves into the prefetched bundle, without any download.
    assert str(params["Cls.model_path"]).startswith(str(tmp_path / "RapidOcr"))
    assert downloaded == []


def test_rapidocr_artifacts_missing_raises_with_prefetch_hint(
    monkeypatch, tmp_path: Path
) -> None:
    _install_fakes(monkeypatch, [])
    with pytest.raises(FileNotFoundError) as excinfo:
        RapidOcrModel(
            enabled=True,
            artifacts_path=tmp_path,
            options=RapidOcrOptions(lang=["th"], backend="onnxruntime"),
            accelerator_options=AcceleratorOptions(),
        )
    message = str(excinfo.value)
    assert "th_PP-OCRv5_rec_mobile.onnx" in message
    # The message must hand the user a command that actually fixes it.
    assert "docling-tools models download rapidocr" in message
    assert "--rapidocr-backend-lang onnxruntime:th-Thai" in message
    assert f"-o {tmp_path}" in message


def test_rapidocr_artifacts_never_downloads(monkeypatch, tmp_path: Path) -> None:
    """A populated artifacts_path must be used without touching the network at all."""
    captured_params: list[dict[str, object]] = []
    _install_fakes(monkeypatch, captured_params)
    _seed(tmp_path, "onnxruntime", "en")

    def explode(url: str, *, progress: bool):
        raise AssertionError(f"unexpected download of {url}")

    monkeypatch.setattr(
        "docling.models.stages.ocr.rapid_ocr_model.download_url_with_progress", explode
    )
    RapidOcrModel(
        enabled=True,
        artifacts_path=tmp_path,
        options=RapidOcrOptions(lang=["en"], backend="onnxruntime"),
        accelerator_options=AcceleratorOptions(),
    )
    assert len(captured_params) == 1


@pytest.mark.parametrize("with_artifacts", [True, False])
def test_rapidocr_missing_pinned_path_raises(
    monkeypatch, tmp_path: Path, with_artifacts: bool
) -> None:
    """A pinned path that does not exist is a config error either way."""
    _install_fakes(monkeypatch, [])
    if with_artifacts:
        _seed(tmp_path, "onnxruntime", "en")
    with pytest.raises(FileNotFoundError, match=r"does_not_exist\.onnx"):
        RapidOcrModel(
            enabled=True,
            artifacts_path=tmp_path if with_artifacts else None,
            options=RapidOcrOptions(
                lang=["en"],
                backend="onnxruntime",
                rec_model_path=str(tmp_path / "does_not_exist.onnx"),
            ),
            accelerator_options=AcceleratorOptions(),
        )


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
    # Both defaults resolve to PP-OCRv6, whose det/rec cover every v6 language.
    assert {call["lang"] for call in captured_calls} == {"zh-Hans"}


def test_model_downloader_rapidocr_models_replaces_default(
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
        rapidocr_models=["onnxruntime:th"],
        with_easyocr=False,
    )

    # Explicit specs replace the default pair rather than extending it.
    assert len(captured_calls) == 1
    assert captured_calls[0]["backend"] == "onnxruntime"
    assert captured_calls[0]["lang"] == "th"


def test_model_downloader_rejects_bad_rapidocr_spec(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires with_rapidocr=True"):
        download_models(
            output_dir=tmp_path, with_rapidocr=False, rapidocr_models=["torch:ch"]
        )


@pytest.mark.parametrize(
    "spec", ["onnxruntime:th", "torch:kn", "paddle:zh-Hans", "openvino:el"]
)
def test_parse_rapidocr_model_spec_accepts_valid_pairs(spec: str) -> None:
    parsed = _parse_rapidocr_model_spec(spec)
    assert f"{parsed.backend}:{parsed.user_lang}" == spec
    # Parsing yields the requested form only; resolution is left to the consumer.
    assert parsed.ppocr_version is None
    assert parsed.rapidocr_code is None


@pytest.mark.parametrize(
    "spec",
    ["torch:th", "torch:el", "onnxruntime:ka-Geor", "bogus:en", "no-colon", "a:b:c"],
)
def test_parse_rapidocr_model_spec_rejects_invalid_pairs(spec: str) -> None:
    with pytest.raises(ValueError):
        _parse_rapidocr_model_spec(spec)
