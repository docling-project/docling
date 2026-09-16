# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from typer.testing import CliRunner

from docling.cli.tools import app
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import EasyOcrOptions
from docling.models.stages.ocr import easyocr_model
from docling.models.stages.ocr.easyocr_model import EasyOcrModel
from docling.utils.model_downloader import download_models

pytestmark = pytest.mark.ml_ocr

# Under CI Rich thinks it has a terminal and styles the error panel, landing
# escapes between the border and the wrapped halves of a sentence.
# `TERM=dumb` turns that off, so the panel arrives as plain wrapped text.
runner = CliRunner(env={"TERM": "dumb"})


def _single_line_cli_output(output: str) -> str:
    """The error panel still wraps and draws borders: flatten it to one line."""
    return " ".join(output.replace("│", "").split())


@pytest.mark.parametrize(
    ("tag", "model_name"),
    [
        ("iso:en", "english_g2"),
        ("iso:de", "latin_g2"),
        ("iso:ar", "arabic_g1"),
        ("iso:bn", "bengali_g1"),
        ("iso:hi", "devanagari_g1"),
        ("iso:ru", "cyrillic_g2"),
        ("iso:th", "thai_g1"),
        ("iso:zh-Hant", "zh_tra_g1"),
        ("iso:zh-Hans", "zh_sim_g2"),
        ("iso:ja", "japanese_g2"),
        ("iso:ko", "korean_g2"),
        ("iso:ta", "tamil_g1"),
        ("iso:te", "telugu_g2"),
        ("iso:kn", "kannada_g2"),
    ],
)
def test_prefetch_resolves_bcp47_to_a_checkpoint(tag: str, model_name: str) -> None:
    codes = easyocr_model.resolve_easyocr_codes([tag])

    assert easyocr_model._resolve_easyocr_recognition_models(codes) == [model_name]


def test_resolve_easyocr_languages_maps_to_native_codes() -> None:
    """EasyOCR keeps its own vocabulary internally; only the input changed."""
    assert easyocr_model.resolve_easyocr_codes(
        ["iso:zh-Hant", "iso:sr-Latn", "iso:tg"]
    ) == [
        "ch_tra",
        "rs_latin",
        "tjk",
    ]


def test_resolve_easyocr_languages_routes_to_the_script_model() -> None:
    """Each language reaches the recognition network of its own script, so the
    caller names languages and never a script."""
    codes = easyocr_model.resolve_easyocr_codes(["iso:ru", "iso:sr-Cyrl"])

    assert codes == ["ru", "rs_cyrillic"]
    assert easyocr_model._resolve_easyocr_recognition_models(codes) == ["cyrillic_g2"]


def test_resolve_easyocr_languages_deduplicates_models() -> None:
    codes = easyocr_model.resolve_easyocr_codes(
        ["iso:de", "iso:fr", "iso:zh-Hans", "iso:de-AT"]
    )

    assert easyocr_model._resolve_easyocr_recognition_models(codes) == [
        "latin_g2",
        "zh_sim_g2",
    ]


def test_resolve_easyocr_languages_rejects_malformed_tag() -> None:
    with pytest.raises(ValueError, match="BCP-47"):
        easyocr_model.resolve_easyocr_codes(["iso:xx"])


def test_resolve_easyocr_languages_rejects_uncovered_language() -> None:
    """`haw` is a valid tag EasyOCR simply has no recognizer for."""
    with pytest.raises(ValueError, match="Unsupported EasyOCR language: iso:haw"):
        easyocr_model.resolve_easyocr_codes(["iso:haw"])


def test_resolve_easyocr_recognition_models_rejects_unsupported_code() -> None:
    with pytest.raises(ValueError, match="Unsupported EasyOCR language code: xx"):
        easyocr_model._resolve_easyocr_recognition_models(["xx"])


@pytest.mark.parametrize(("lang", "expected"), [([], ["en"]), (["de"], ["de"])])
def test_empty_lang_reaches_the_reader_as_english(
    monkeypatch, lang: list[str], expected: list[str]
) -> None:
    """EasyOCR has no engine default, and an empty `lang_list` is not one.

    `easyocr.Reader([])` falls back to the `latin_g2` checkpoint with only that
    model's symbols as its character set, so every letter is dropped from the
    recognized text -- silently. Docling names a language instead.
    """
    import easyocr

    captured: list[list[str]] = []
    monkeypatch.setattr(
        easyocr,
        "Reader",
        lambda lang_list, **kwargs: captured.append(lang_list),
    )

    EasyOcrModel(
        enabled=True,
        artifacts_path=None,
        options=EasyOcrOptions(lang=lang),
        accelerator_options=AcceleratorOptions(),
    )

    assert captured == [expected]


def test_easyocr_downloader_supports_gen1_and_gen2_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from easyocr.config import recognition_models

    filenames_by_url = {
        details["url"]: details["filename"]
        for generation in recognition_models.values()
        for details in generation.values()
    }

    def fake_download_url_with_progress(url: str, *, progress: bool) -> BytesIO:
        del progress
        archive = BytesIO()
        with zipfile.ZipFile(archive, "w") as zip_file:
            zip_file.writestr(filenames_by_url[url], b"weights")
        archive.seek(0)
        return archive

    monkeypatch.setattr(
        easyocr_model,
        "download_url_with_progress",
        fake_download_url_with_progress,
    )

    EasyOcrModel.download_models(
        detection_models=[],
        recognition_models=["arabic_g1", "japanese_g2"],
        local_dir=tmp_path,
    )

    assert (tmp_path / "arabic.pth").is_file()
    assert (tmp_path / "japanese_g2.pth").is_file()


def test_easyocr_downloader_ignores_unknown_internal_model_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_download(url: str, *, progress: bool) -> BytesIO:
        raise AssertionError((url, progress))

    monkeypatch.setattr(
        easyocr_model,
        "download_url_with_progress",
        fail_download,
    )

    local_dir = tmp_path / "models"
    EasyOcrModel.download_models(
        detection_models=["unknown"],
        recognition_models=["unknown"],
        local_dir=local_dir,
    )

    assert local_dir.is_dir()


def test_model_downloader_resolves_requested_easyocr_languages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_calls: list[dict[str, object]] = []

    def fake_download_models(**kwargs: object) -> None:
        captured_calls.append(kwargs)

    monkeypatch.setattr(EasyOcrModel, "download_models", fake_download_models)

    download_models(
        output_dir=tmp_path,
        with_layout=False,
        with_tableformer=False,
        with_code_formula=False,
        with_picture_classifier=False,
        with_rapidocr=False,
        with_easyocr=True,
        easyocr_languages=["iso:zh-Hans", "ja", "iso:zh-CN"],
    )

    assert len(captured_calls) == 1
    assert captured_calls[0]["recognition_models"] == [
        "japanese_g2",
        "zh_sim_g2",
    ]


def test_model_downloader_preserves_default_easyocr_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_calls: list[dict[str, object]] = []

    def fake_download_models(**kwargs: object) -> None:
        captured_calls.append(kwargs)

    monkeypatch.setattr(EasyOcrModel, "download_models", fake_download_models)

    download_models(
        output_dir=tmp_path,
        with_layout=False,
        with_tableformer=False,
        with_code_formula=False,
        with_picture_classifier=False,
        with_rapidocr=False,
        with_easyocr=True,
    )

    assert len(captured_calls) == 1
    assert captured_calls[0]["recognition_models"] == [
        "english_g2",
        "latin_g2",
    ]


def test_model_downloader_validates_easyocr_languages_before_io(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "models"

    with pytest.raises(ValueError, match="BCP-47"):
        download_models(
            output_dir=output_dir,
            with_layout=False,
            with_tableformer=False,
            with_code_formula=False,
            with_picture_classifier=False,
            with_rapidocr=False,
            with_easyocr=True,
            easyocr_languages=["iso:xx"],
        )

    assert not output_dir.exists()


def test_model_downloader_requires_easyocr_for_languages(tmp_path: Path) -> None:
    output_dir = tmp_path / "models"

    with pytest.raises(ValueError, match="easyocr_languages requires"):
        download_models(
            output_dir=output_dir,
            with_layout=False,
            with_tableformer=False,
            with_code_formula=False,
            with_picture_classifier=False,
            with_rapidocr=False,
            easyocr_languages=["ja"],
        )

    assert not output_dir.exists()


@pytest.mark.parametrize("model_args", [["easyocr"], ["--all"]])
def test_models_cli_accepts_repeated_easyocr_languages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_args: list[str]
) -> None:
    captured_calls: list[dict[str, object]] = []

    def fake_download_models(**kwargs: object) -> Path:
        captured_calls.append(kwargs)
        return tmp_path

    monkeypatch.setattr("docling.cli.models.download_models", fake_download_models)

    result = runner.invoke(
        app,
        [
            "models",
            "download",
            *model_args,
            "--easyocr-lang",
            "iso:zh-Hans",
            "--easyocr-lang",
            "ja",
            "--output-dir",
            str(tmp_path),
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(captured_calls) == 1
    # The CLI hands the downloader the user's tags; they are resolved there.
    assert captured_calls[0]["easyocr_languages"] == ["iso:zh-Hans", "ja"]


def test_models_cli_rejects_easyocr_languages_without_easyocr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called = False

    def fake_download_models(**kwargs: object) -> Path:
        nonlocal called
        called = True
        return tmp_path

    monkeypatch.setattr("docling.cli.models.download_models", fake_download_models)

    result = runner.invoke(
        app,
        [
            "models",
            "download",
            "--easyocr-lang",
            "ja",
            "--output-dir",
            str(tmp_path),
            "--quiet",
        ],
    )

    assert result.exit_code == 2
    assert "--easyocr-lang requires the 'easyocr' model" in _single_line_cli_output(
        result.output
    )
    assert not called


def test_models_cli_rejects_unsupported_easyocr_language(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called = False

    def fake_download_models(**kwargs: object) -> Path:
        nonlocal called
        called = True
        return tmp_path

    monkeypatch.setattr("docling.cli.models.download_models", fake_download_models)

    result = runner.invoke(
        app,
        [
            "models",
            "download",
            "easyocr",
            "--easyocr-lang",
            "iso:xx",
            "--output-dir",
            str(tmp_path),
            "--quiet",
        ],
    )

    assert result.exit_code == 2
    assert "BCP-47" in _single_line_cli_output(result.output)
    assert not called
