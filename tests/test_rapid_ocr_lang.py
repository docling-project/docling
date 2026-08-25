# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
from io import BytesIO
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.datamodel.settings import settings
from docling.models.stages.ocr.rapid_ocr_model import (
    RapidOcrModel,
    _parse_rapidocr_model_spec,
    _resolve_rapidocr,
)
from docling.utils.model_downloader import _DEFAULT_RAPIDOCR_MODELS, download_models

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


def _seed(
    artifacts_path: Path,
    backend: str,
    lang: str,
    model_size: Literal["tiny", "small", "medium"] = "small",
) -> None:
    """Prefetch one `(backend, lang, model_size)` set into artifacts_path, as a user would."""
    RapidOcrModel.download_models(
        backend=backend,
        lang=lang,
        local_dir=artifacts_path / RapidOcrModel._model_repo_folder,
        model_size=model_size,
    )


def _build(
    monkeypatch,
    options: RapidOcrOptions,
    artifacts_path: Path | None,
    *,
    seed: tuple[str, str]
    | tuple[str, str, Literal["tiny", "small", "medium"]]
    | None = None,
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
    """The (version, registry token) pair the assertions below care about."""
    spec = _resolve_rapidocr(lang, backend)
    return spec.ppocr_version, spec.rapidocr_lang_token


def test_resolve_populates_the_whole_spec() -> None:
    from rapidocr.utils.typings import OCRVersion

    spec = _resolve_rapidocr("zh", "onnxruntime")
    assert spec.backend == "onnxruntime"
    # The user's token is preserved verbatim, the registry token is normalized.
    assert spec.user_lang == "zh"
    assert spec.rapidocr_lang_token == "ch"
    assert spec.ppocr_version == OCRVersion.PPOCRV6


def test_resolve_defaults_to_ppocrv6_chinese() -> None:
    from rapidocr.utils.typings import OCRVersion

    assert _resolved("chinese", "onnxruntime") == (OCRVersion.PPOCRV6, "ch")
    assert _resolved("zh", "onnxruntime") == (OCRVersion.PPOCRV6, "ch")


def test_resolve_english_and_latin_use_ppocrv6() -> None:
    from rapidocr.utils.typings import OCRVersion

    assert _resolved("english", "onnxruntime") == (OCRVersion.PPOCRV6, "en")
    assert _resolved("en", "torch") == (OCRVersion.PPOCRV6, "en")
    assert _resolved("de", "onnxruntime") == (OCRVersion.PPOCRV6, "de")
    assert _resolved("fr", "onnxruntime") == (OCRVersion.PPOCRV6, "fr")


def test_resolve_script_families_route_by_backend() -> None:
    from rapidocr.utils.typings import OCRVersion

    # onnxruntime/openvino/paddle -> PP-OCRv5
    assert _resolved("th", "onnxruntime") == (OCRVersion.PPOCRV5, "th")
    assert _resolved("cyrillic", "onnxruntime") == (OCRVersion.PPOCRV5, "cyrillic")
    # torch -> PP-OCRv4
    assert _resolved("arabic", "torch") == (OCRVersion.PPOCRV4, "arabic")


def test_resolve_raises_on_unsupported_language() -> None:
    with pytest.raises(ValueError):
        _resolve_rapidocr("klingon", "onnxruntime")
    # Thai is a PP-OCRv5 language, not served by the torch PP-OCRv4 backbone.
    with pytest.raises(ValueError):
        _resolve_rapidocr("th", "torch")


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
        RapidOcrOptions(backend="torch"),  # default lang -> chinese -> ch -> v6
        tmp_path,
        seed=("torch", "chinese"),
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
        RapidOcrOptions(lang=["arabic"], backend="torch"),
        tmp_path,
        seed=("torch", "arabic"),
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
    assert "--rapidocr-backend-lang onnxruntime:th" in message
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
    assert {call["lang"] for call in captured_calls} == {"ch"}


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
    "spec", ["onnxruntime:th", "torch:ka", "paddle:ch", "openvino:el"]
)
def test_parse_rapidocr_model_spec_accepts_valid_pairs(spec: str) -> None:
    parsed = _parse_rapidocr_model_spec(spec)
    assert f"{parsed.backend}:{parsed.user_lang}" == spec
    # Parsing yields the requested form only; resolution is left to the consumer.
    assert parsed.ppocr_version is None
    assert parsed.rapidocr_lang_token is None


@pytest.mark.parametrize(
    "spec", ["torch:th", "torch:el", "onnxruntime:ka", "bogus:en", "no-colon", "a:b:c"]
)
def test_parse_rapidocr_model_spec_rejects_invalid_pairs(spec: str) -> None:
    with pytest.raises(ValueError):
        _parse_rapidocr_model_spec(spec)


# --- model_size ---------------------------------------------------------------


def test_rapidocr_options_model_size_defaults_to_small() -> None:
    assert RapidOcrOptions().model_size == "small"


def test_rapidocr_options_without_model_size_still_deserializes() -> None:
    """Old serialized configs (no `model_size` key) must still construct with the default."""
    old_style = {"lang": ["en"], "backend": "onnxruntime"}
    options = RapidOcrOptions.model_validate(old_style)
    assert options.model_size == "small"
    # A full dump/reload round-trip must also stay stable.
    assert RapidOcrOptions.model_validate(options.model_dump()).model_size == "small"


def test_rapidocr_options_model_size_rejects_invalid_value() -> None:
    with pytest.raises(ValidationError):
        RapidOcrOptions(model_size="large")  # type: ignore


@pytest.mark.parametrize(
    "model_size,det_name,rec_name",
    [
        ("small", "PP-OCRv6_det_small.onnx", "PP-OCRv6_rec_small.onnx"),
        ("tiny", "PP-OCRv6_det_tiny.onnx", "PP-OCRv6_rec_tiny.onnx"),
        ("medium", "PP-OCRv6_det_medium.onnx", "PP-OCRv6_rec_medium.onnx"),
    ],
)
def test_rapidocr_model_size_selects_matching_ppocrv6_assets(
    monkeypatch,
    tmp_path: Path,
    model_size: Literal["tiny", "small", "medium"],
    det_name: str,
    rec_name: str,
) -> None:
    """Each size resolves to its own PP-OCRv6 det/rec checkpoint."""
    params, downloaded = _build(
        monkeypatch,
        RapidOcrOptions(lang=["en"], backend="onnxruntime", model_size=model_size),
        tmp_path,
        seed=("onnxruntime", "en", model_size),
    )
    assert Path(params["Det.model_path"]).name == det_name
    assert Path(params["Rec.model_path"]).name == rec_name
    assert downloaded == []


@pytest.mark.parametrize("model_size", ["tiny", "small", "medium"])
def test_rapidocr_model_size_never_affects_classification(
    monkeypatch, tmp_path: Path, model_size: Literal["tiny", "small", "medium"]
) -> None:
    """The classification checkpoint is always PP-OCRv4 mobile, regardless of size."""
    params, _ = _build(
        monkeypatch,
        RapidOcrOptions(lang=["en"], backend="onnxruntime", model_size=model_size),
        tmp_path,
        seed=("onnxruntime", "en", model_size),
    )
    assert Path(params["Cls.model_path"]).name == "ch_ppocr_mobile_v2.0_cls_mobile.onnx"


def test_rapidocr_model_size_tiny_rejects_japanese(monkeypatch, tmp_path: Path) -> None:
    """`tiny` is a real, registry-enforced restriction: rapidocr itself rejects it."""
    _install_fakes(monkeypatch, [])
    with pytest.raises(ValueError, match=r"[Jj]apan"):
        RapidOcrModel(
            enabled=True,
            artifacts_path=tmp_path,
            options=RapidOcrOptions(
                lang=["japan"], backend="onnxruntime", model_size="tiny"
            ),
            accelerator_options=AcceleratorOptions(),
        )


@pytest.mark.parametrize("model_size", ["tiny", "medium"])
def test_rapidocr_model_size_ignored_for_non_ppocrv6_with_warning(
    monkeypatch,
    tmp_path: Path,
    caplog,
    model_size: Literal["tiny", "small", "medium"],
) -> None:
    """A non-default size on a non-PP-OCRv6 language warns and still works correctly."""
    with caplog.at_level(logging.WARNING):
        params, _ = _build(
            monkeypatch,
            RapidOcrOptions(lang=["th"], backend="onnxruntime", model_size=model_size),
            tmp_path,
            seed=("onnxruntime", "th"),
        )
    # "th" resolves to PP-OCRv5, which has no tiny/small/medium axis at all.
    assert Path(params["Det.model_path"]).name == "ch_PP-OCRv5_det_mobile.onnx"
    assert Path(params["Rec.model_path"]).name == "th_PP-OCRv5_rec_mobile.onnx"
    assert any(
        "model_size" in record.message and "PP-OCRv6" in record.message
        for record in caplog.records
    )


def test_rapidocr_model_size_small_no_warning_for_non_ppocrv6(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    """The default size must never generate a new warning for existing users."""
    with caplog.at_level(logging.WARNING):
        _build(
            monkeypatch,
            RapidOcrOptions(lang=["th"], backend="onnxruntime"),  # model_size="small"
            tmp_path,
            seed=("onnxruntime", "th"),
        )
    assert not any("model_size" in record.message for record in caplog.records)


def test_rapidocr_model_size_applies_after_language_reduction(
    monkeypatch, tmp_path: Path
) -> None:
    """Multi-language input is reduced to one language first; size still applies to it."""
    params, _ = _build(
        monkeypatch,
        RapidOcrOptions(lang=["en", "th"], backend="onnxruntime", model_size="tiny"),
        tmp_path,
        seed=("onnxruntime", "en", "tiny"),
    )
    # "en" (the first language) resolves to PP-OCRv6, so model_size applies to it.
    assert Path(params["Det.model_path"]).name == "PP-OCRv6_det_tiny.onnx"
    assert Path(params["Rec.model_path"]).name == "PP-OCRv6_rec_tiny.onnx"


def test_rapidocr_artifacts_missing_hint_omits_model_size_when_default(
    monkeypatch, tmp_path: Path
) -> None:
    """The remediation hint should not mention --model-size unless it's non-default."""
    _install_fakes(monkeypatch, [])
    with pytest.raises(FileNotFoundError) as excinfo:
        RapidOcrModel(
            enabled=True,
            artifacts_path=tmp_path,
            options=RapidOcrOptions(lang=["en"], backend="onnxruntime"),
            accelerator_options=AcceleratorOptions(),
        )
    assert "--model-size" not in str(excinfo.value)


def test_rapidocr_model_size_mismatch_with_prefetched_assets_raises(
    monkeypatch, tmp_path: Path
) -> None:
    """Only `small` was prefetched; requesting `tiny` must fail, not silently reuse small."""
    _install_fakes(monkeypatch, [])
    _seed(tmp_path, "onnxruntime", "en")  # downloads the default "small" assets
    with pytest.raises(FileNotFoundError) as excinfo:
        RapidOcrModel(
            enabled=True,
            artifacts_path=tmp_path,
            options=RapidOcrOptions(
                lang=["en"], backend="onnxruntime", model_size="tiny"
            ),
            accelerator_options=AcceleratorOptions(),
        )
    message = str(excinfo.value)
    assert "PP-OCRv6_det_tiny.onnx" in message
    # The suggested fix command must actually work, not just re-download small again.
    assert "--model-size tiny" in message
    assert "--rapidocr-backend-lang onnxruntime:en" in message


@pytest.mark.parametrize(
    "model_size,det_name,rec_name",
    [
        ("tiny", "PP-OCRv6_det_tiny.onnx", "PP-OCRv6_rec_tiny.onnx"),
        ("medium", "PP-OCRv6_det_medium.onnx", "PP-OCRv6_rec_medium.onnx"),
    ],
)
def test_download_models_respects_model_size(
    monkeypatch,
    tmp_path: Path,
    model_size: Literal["tiny", "medium"],
    det_name: str,
    rec_name: str,
) -> None:
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
        model_size=model_size,
    )

    assert any(det_name in url for url in downloaded_urls)
    assert any(rec_name in url for url in downloaded_urls)
    assert (tmp_path / det_name).exists()
    assert (tmp_path / rec_name).exists()
    # The classification checkpoint must be unaffected by the requested size.
    assert any("ch_ppocr_mobile_v2.0_cls_mobile.onnx" in url for url in downloaded_urls)


def test_model_downloader_forwards_rapidocr_model_size(
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
        rapidocr_model_size="medium",
        with_easyocr=False,
    )

    # One RapidOcrModel.download_models() call per default (backend, lang) entry --
    # tied to the actual constant, not a magic number, so this doesn't need updating
    # if that default set ever changes; what this test cares about is that every
    # such call receives the requested size.
    assert len(captured_calls) == len(_DEFAULT_RAPIDOCR_MODELS)
    assert all(call["model_size"] == "medium" for call in captured_calls)


def test_model_downloader_rapidocr_model_size_defaults_to_small(
    monkeypatch, tmp_path: Path
) -> None:
    """Existing callers that don't pass rapidocr_model_size must keep getting `small`."""
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

    assert all(call["model_size"] == "small" for call in captured_calls)
