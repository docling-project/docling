"""Tests for the ``docling-tools`` CLI.

These exercise the option surface of ``docling-tools models`` without
downloading anything: the downloader entry points are replaced with recorders
so the tests can assert which model selection the CLI derives from a given set
of flags. That mapping is the actual contract of these commands -- everything
else they do is delegated.
"""

import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import docling.cli.models as models_cli
from docling.cli.models import _AvailableModels, _default_models
from docling.cli.tools import app

runner = CliRunner()


@pytest.fixture
def recorded_download(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace ``download_models`` with a recorder returning its output dir."""
    recorded: dict[str, Any] = {}

    def _fake_download_models(**kwargs: Any) -> Path:
        recorded.update(kwargs)
        return kwargs["output_dir"]

    monkeypatch.setattr(models_cli, "download_models", _fake_download_models)
    return recorded


@pytest.fixture
def recorded_hf_download(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace ``download_hf_model`` with a recorder of per-repo calls."""
    calls: list[dict[str, Any]] = []

    def _fake_download_hf_model(**kwargs: Any) -> Path:
        calls.append(kwargs)
        return kwargs["local_dir"]

    monkeypatch.setattr(models_cli, "download_hf_model", _fake_download_hf_model)
    return calls


def _enabled(recorded: dict[str, Any]) -> set[str]:
    """The ``with_*`` selection flags the CLI turned on."""
    return {key for key, value in recorded.items() if key.startswith("with_") and value}


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_BOX_DRAWING = re.compile(r"[\u2500-\u257f]")


def _flat(output: str) -> str:
    """Reduce a Rich error panel to a single line of plain text.

    Typer renders ``BadParameter`` messages inside a bordered panel and hard
    wraps them, so error text cannot be matched against the raw output. When
    the output stream is a terminal -- as it is under CI -- Rich also emits
    colour codes, including between the wrapped halves of a sentence, so the
    escapes have to go before whitespace is collapsed or the message is still
    split.
    """
    stripped = _ANSI_ESCAPE.sub("", output)
    stripped = _BOX_DRAWING.sub(" ", stripped)
    return re.sub(r"\s+", " ", stripped)


def test_tools_help_lists_models_subcommand():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "models" in result.output


def test_tools_without_arguments_shows_help():
    result = runner.invoke(app, [])
    # no_args_is_help=True makes Typer exit with the usage screen.
    assert result.exit_code != 0
    assert "Usage" in result.output


def test_models_without_arguments_shows_help():
    result = runner.invoke(app, ["models"])
    assert result.exit_code != 0
    assert "download" in result.output


def test_download_defaults_to_the_predefined_model_set(tmp_path, recorded_download):
    result = runner.invoke(app, ["models", "download", "-o", str(tmp_path)])

    assert result.exit_code == 0
    assert recorded_download["output_dir"] == tmp_path
    assert recorded_download["force"] is False
    assert recorded_download["progress"] is True
    assert _enabled(recorded_download) == {
        "with_layout",
        "with_tableformer",
        "with_code_formula",
        "with_picture_classifier",
        "with_rapidocr",
    }
    assert len(_default_models) == 5


def test_download_all_selects_every_available_model(tmp_path, recorded_download):
    result = runner.invoke(app, ["models", "download", "-o", str(tmp_path), "--all"])

    assert result.exit_code == 0
    selection = {k: v for k, v in recorded_download.items() if k.startswith("with_")}
    assert len(selection) == len(_AvailableModels)
    assert all(selection.values()), (
        f"not selected by --all: {_enabled(recorded_download) ^ set(selection)}"
    )


def test_download_explicit_models_override_the_defaults(tmp_path, recorded_download):
    result = runner.invoke(
        app, ["models", "download", "-o", str(tmp_path), "layout", "smolvlm"]
    )

    assert result.exit_code == 0
    assert _enabled(recorded_download) == {"with_layout", "with_smolvlm"}


def test_download_rejects_all_together_with_explicit_models(tmp_path):
    result = runner.invoke(
        app, ["models", "download", "-o", str(tmp_path), "--all", "layout"]
    )

    assert result.exit_code != 0
    assert "Cannot simultaneously set" in _flat(result.output)


def test_download_rejects_unknown_model_name(tmp_path):
    result = runner.invoke(
        app, ["models", "download", "-o", str(tmp_path), "not-a-model"]
    )

    assert result.exit_code != 0


def test_download_forwards_force_flag(tmp_path, recorded_download):
    result = runner.invoke(
        app, ["models", "download", "-o", str(tmp_path), "--force", "layout"]
    )

    assert result.exit_code == 0
    assert recorded_download["force"] is True


def test_quiet_download_prints_only_the_output_directory(tmp_path, recorded_download):
    result = runner.invoke(
        app, ["models", "download", "-o", str(tmp_path), "-q", "layout"]
    )

    assert result.exit_code == 0
    assert recorded_download["progress"] is False
    assert result.output.strip() == str(tmp_path)


def test_verbose_download_prints_offline_usage_hint(tmp_path, recorded_download):
    result = runner.invoke(app, ["models", "download", "-o", str(tmp_path), "layout"])

    assert result.exit_code == 0
    # Rich wraps and colours this hint, so normalise before matching.
    output = _flat(result.output)
    assert "Models downloaded into" in output
    assert "--artifacts-path" in output


def test_easyocr_lang_is_forwarded_when_easyocr_is_selected(
    tmp_path, recorded_download
):
    result = runner.invoke(
        app,
        [
            "models",
            "download",
            "-o",
            str(tmp_path),
            "--easyocr-lang",
            "en",
            "--easyocr-lang",
            "de",
            "easyocr",
        ],
    )

    assert result.exit_code == 0
    assert recorded_download["easyocr_languages"] == ["en", "de"]


def test_easyocr_lang_requires_the_easyocr_model(tmp_path):
    result = runner.invoke(
        app,
        ["models", "download", "-o", str(tmp_path), "--easyocr-lang", "en", "layout"],
    )

    assert result.exit_code != 0
    assert "requires the 'easyocr' model" in _flat(result.output)


def test_easyocr_lang_rejects_an_unresolvable_language(tmp_path):
    result = runner.invoke(
        app,
        [
            "models",
            "download",
            "-o",
            str(tmp_path),
            "--easyocr-lang",
            "not-a-language",
            "easyocr",
        ],
    )

    assert result.exit_code != 0


def test_rapidocr_backend_lang_is_forwarded_when_rapidocr_is_selected(
    tmp_path, recorded_download
):
    result = runner.invoke(
        app,
        [
            "models",
            "download",
            "-o",
            str(tmp_path),
            "--rapidocr-backend-lang",
            "onnxruntime:el",
            "rapidocr",
        ],
    )

    assert result.exit_code == 0
    assert recorded_download["rapidocr_models"] == ["onnxruntime:el"]


def test_rapidocr_backend_lang_requires_the_rapidocr_model(tmp_path):
    result = runner.invoke(
        app,
        [
            "models",
            "download",
            "-o",
            str(tmp_path),
            "--rapidocr-backend-lang",
            "onnxruntime:el",
            "layout",
        ],
    )

    assert result.exit_code != 0
    assert "requires the 'rapidocr' model" in _flat(result.output)


def test_rapidocr_backend_lang_rejects_a_malformed_spec(tmp_path):
    result = runner.invoke(
        app,
        [
            "models",
            "download",
            "-o",
            str(tmp_path),
            "--rapidocr-backend-lang",
            "no-separator-here",
            "rapidocr",
        ],
    )

    assert result.exit_code != 0


def test_download_hf_repo_maps_repo_ids_to_local_directories(
    tmp_path, recorded_hf_download
):
    result = runner.invoke(
        app,
        [
            "models",
            "download-hf-repo",
            "-o",
            str(tmp_path),
            "docling-project/docling-models",
            "org/other",
        ],
    )

    assert result.exit_code == 0
    assert [call["repo_id"] for call in recorded_hf_download] == [
        "docling-project/docling-models",
        "org/other",
    ]
    # The repo id is flattened into a single directory name.
    assert [call["local_dir"] for call in recorded_hf_download] == [
        tmp_path / "docling-project--docling-models",
        tmp_path / "org--other",
    ]
    assert all(call["force"] is False for call in recorded_hf_download)
    assert all(call["progress"] is True for call in recorded_hf_download)


def test_quiet_download_hf_repo_prints_only_the_output_directory(
    tmp_path, recorded_hf_download
):
    result = runner.invoke(
        app,
        ["models", "download-hf-repo", "-o", str(tmp_path), "-q", "org/repo"],
    )

    assert result.exit_code == 0
    assert recorded_hf_download[0]["progress"] is False
    # --quiet documents that only the directory is printed; the per-repo
    # progress line must stay suppressed here as it is for `download`.
    assert result.output.strip() == str(tmp_path)


def test_download_hf_repo_requires_at_least_one_repo(tmp_path):
    result = runner.invoke(app, ["models", "download-hf-repo", "-o", str(tmp_path)])

    assert result.exit_code != 0
