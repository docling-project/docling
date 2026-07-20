import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from typer.testing import CliRunner

from docling.cli.tools import app
from docling.models.stages.ocr import easyocr_model
from docling.models.stages.ocr.easyocr_model import EasyOcrModel
from docling.utils.model_downloader import download_models

pytestmark = pytest.mark.ml_ocr

runner = CliRunner()


def _single_line_cli_output(output: str) -> str:
    return " ".join(output.replace("│", "").split())


@pytest.mark.parametrize(
    ("language", "model_name"),
    [
        ("en", "english_g2"),
        ("de", "latin_g2"),
        ("ar", "arabic_g1"),
        ("bn", "bengali_g1"),
        ("hi", "devanagari_g1"),
        ("ru", "cyrillic_g2"),
        ("th", "thai_g1"),
        ("ch_tra", "zh_tra_g1"),
        ("ch_sim", "zh_sim_g2"),
        ("ja", "japanese_g2"),
        ("ko", "korean_g2"),
        ("ta", "tamil_g1"),
        ("te", "telugu_g2"),
        ("kn", "kannada_g2"),
    ],
)
def test_resolve_easyocr_language(language: str, model_name: str) -> None:
    assert easyocr_model._resolve_easyocr_recognition_models([language]) == [model_name]


def test_resolve_easyocr_languages_deduplicates_models() -> None:
    assert easyocr_model._resolve_easyocr_recognition_models(
        ["de", "fr", "ch_sim", "de", "ch_sim"]
    ) == ["latin_g2", "zh_sim_g2"]


def test_resolve_easyocr_languages_rejects_unsupported_code() -> None:
    with pytest.raises(ValueError, match="Unsupported EasyOCR language code: xx"):
        easyocr_model._resolve_easyocr_recognition_models(["xx"])


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
        easyocr_languages=["ch_sim", "ja", "ch_sim"],
    )

    assert len(captured_calls) == 1
    assert captured_calls[0]["recognition_models"] == [
        "zh_sim_g2",
        "japanese_g2",
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

    with pytest.raises(ValueError, match="Unsupported EasyOCR language code: xx"):
        download_models(
            output_dir=output_dir,
            with_layout=False,
            with_tableformer=False,
            with_code_formula=False,
            with_picture_classifier=False,
            with_rapidocr=False,
            with_easyocr=True,
            easyocr_languages=["xx"],
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
            "ch_sim",
            "--easyocr-lang",
            "ja",
            "--output-dir",
            str(tmp_path),
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(captured_calls) == 1
    assert captured_calls[0]["easyocr_languages"] == ["ch_sim", "ja"]


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
            "xx",
            "--output-dir",
            str(tmp_path),
            "--quiet",
        ],
    )

    assert result.exit_code == 2
    assert "Unsupported EasyOCR language code: xx" in _single_line_cli_output(
        result.output
    )
    assert not called
