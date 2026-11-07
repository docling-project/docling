# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for `docling models` and the short model options in `docling convert`."""

import importlib.metadata
import json
from io import StringIO

import pytest
from rich.console import Console
from typer.main import get_command
from typer.testing import CliRunner

from docling.cli import main, model_table
from docling.cli.main import _resolve_asr_options, app
from docling.cli.model_table import OTHER_FAMILY, BackendState, asr_rows, vlm_rows
from docling.datamodel import asr_presets
from docling.datamodel.asr_model_specs import AsrModelType
from docling.datamodel.pipeline_options_vlm_model import ResponseFormat

runner = CliRunner()


def _on_apple_silicon(monkeypatch: pytest.MonkeyPatch, apple_silicon: bool) -> None:
    monkeypatch.setattr(asr_presets, "is_apple_silicon", lambda: apple_silicon)
    monkeypatch.setattr(model_table, "is_apple_silicon", lambda: apple_silicon)
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "9.9.9")
    monkeypatch.setattr(model_table, "find_spec", lambda name: object())


def _row(rows: list[model_table.ModelRow], name: str) -> model_table.ModelRow:
    return next(row for row in rows if row.name == name)


def test_every_asr_preset_has_a_row_and_a_column():
    rows = {row.name: row for row in asr_rows(_resolve_asr_options)}

    for model in AsrModelType:
        assert any(
            model.value == name or model.value.startswith(f"{name}_") for name in rows
        ), model


def test_preset_without_a_family_entry_is_still_listed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        model_table,
        "_VLM_FAMILIES",
        tuple(f for f in model_table._VLM_FAMILIES if f[0] != "surya"),
    )

    result = runner.invoke(app, ["models", "--vlm"], terminal_width=200)

    assert result.exit_code == 0, result.output
    assert _row(vlm_rows(), "surya_ocr2_5").family == OTHER_FAMILY
    assert "surya_ocr2_5" in result.output


def test_families_stay_together():
    for rows in (asr_rows(_resolve_asr_options), vlm_rows()):
        seen: list[str] = []
        for row in rows:
            if not seen or seen[-1] != row.family:
                assert row.family not in seen
                seen.append(row.family)


def test_asr_table_marks_the_engine_for_this_machine(monkeypatch: pytest.MonkeyPatch):
    _on_apple_silicon(monkeypatch, True)
    on_mac = _row(asr_rows(_resolve_asr_options), "granite_speech_5_0_2b")
    assert on_mac.backends == {
        "mlx_audio": BackendState.SELECTED,
        "transformers": BackendState.SUPPORTED,
    }

    _on_apple_silicon(monkeypatch, False)
    on_linux = _row(asr_rows(_resolve_asr_options), "granite_speech_5_0_2b")
    assert on_linux.backends == {
        "mlx_audio": BackendState.SUPPORTED,
        "transformers": BackendState.SELECTED,
    }


def test_asr_model_without_a_generic_name_shows_no_selected_backend():
    row = _row(asr_rows(_resolve_asr_options), "whisper_tiny_en")

    assert row.backends == {
        "native": BackendState.SUPPORTED,
        "s2t": BackendState.SUPPORTED,
    }


def test_vlm_table_separates_declared_and_unrestricted_backends(
    monkeypatch: pytest.MonkeyPatch,
):
    _on_apple_silicon(monkeypatch, True)
    rows = vlm_rows()

    # No `supported_engines`: what the preset does not declare is only allowed.
    assert _row(rows, "granite_docling").backends == {
        "mlx": BackendState.SELECTED,
        "transformers": BackendState.SUPPORTED,
        "vllm": BackendState.ALLOWED,
        "api": BackendState.ALLOWED,
    }
    # An explicit `supported_engines` closes the other backends.
    assert _row(rows, "granite_vision").backends == {
        "transformers": BackendState.SELECTED,
        "vllm": BackendState.SUPPORTED,
        "api": BackendState.SUPPORTED,
    }
    # A fixed engine is selected on every machine.
    assert _row(rows, "gemma_12b").backends == {"mlx": BackendState.SELECTED}


def test_vlm_uses_transformers_without_mlx_vlm(monkeypatch: pytest.MonkeyPatch):
    _on_apple_silicon(monkeypatch, True)
    monkeypatch.setattr(model_table, "find_spec", lambda name: None)

    row = _row(vlm_rows(), "granite_docling")

    assert row.backends["transformers"] == BackendState.SELECTED
    assert row.backends["mlx"] == BackendState.SUPPORTED


@pytest.mark.parametrize(
    ("name", "apple_silicon", "timestamps", "speakers"),
    [
        ("whisper_tiny_en", True, True, False),
        # Granite Plus gives real times and speakers on every backend.
        ("granite_speech_4_1_2b_plus", True, True, True),
        ("granite_speech_4_1_2b_plus", False, True, True),
        # Other Granite models only have the time range of a window.
        ("granite_speech_5_0_2b", True, False, False),
        # Parakeet has sentence times on mlx-audio, and one segment on Transformers.
        ("parakeet_ctc_1_1b", True, True, False),
        ("parakeet_ctc_1_1b", False, False, False),
        ("nemotron_3_5_asr_streaming", True, True, False),
        ("nemotron_3_5_asr_streaming", False, False, False),
    ],
)
def test_asr_features_follow_the_selected_backend(
    monkeypatch: pytest.MonkeyPatch, name, apple_silicon, timestamps, speakers
):
    _on_apple_silicon(monkeypatch, apple_silicon)

    row = _row(asr_rows(_resolve_asr_options), name)

    assert row.features == {"timestamps": timestamps, "speaker_labels": speakers}


def test_every_vlm_response_format_is_classified_for_grounding():
    assert model_table.BBOX_RESPONSE_FORMATS.isdisjoint(
        model_table.NO_BBOX_RESPONSE_FORMATS
    )
    assert (
        model_table.BBOX_RESPONSE_FORMATS | model_table.NO_BBOX_RESPONSE_FORMATS
    ) == set(ResponseFormat)


@pytest.mark.parametrize(
    ("name", "grounded"),
    [
        ("granite_docling", True),
        ("dots_ocr", True),
        ("mineru2_pro", True),
        ("lightonocr3_08b", True),
        ("surya_ocr2_5", True),
        ("lightonocr", False),
        ("granite_vision", False),
        ("qwen", False),
    ],
)
def test_vlm_table_shows_bbox_grounding(name, grounded):
    assert _row(vlm_rows(), name).features == {"bbox_grounding": grounded}


def test_models_command_shows_rows_families_and_every_legend_entry():
    result = runner.invoke(app, ["models"], terminal_width=200)

    assert result.exit_code == 0, result.output
    for text in (
        "ASR models",
        "VLM models",
        "Granite Speech",
        "parakeet_ctc_1_1b",
        "granite_docling *",
        "whisper_tiny *",
        "selected on this machine",
        "supported:",
        "allowed:",
        "default model",
        "_mlx_audio",
        "Times",
        "Spk",
        "Bbox",
    ):
        assert text in result.output


def test_table_fits_a_narrow_terminal_without_cut_names():
    result = runner.invoke(app, ["models", "--asr"], terminal_width=80)

    assert result.exit_code == 0, result.output
    assert "…" not in result.output
    assert "granite_speech_5_0_470m_turboctc_nc" in result.output


@pytest.mark.parametrize(
    ("args", "present", "absent"),
    [
        (["--asr"], "parakeet_ctc_1_1b", "granite_docling"),
        (["--vlm"], "granite_docling", "parakeet_ctc_1_1b"),
        (["--family", "parakeet"], "parakeet_ctc_1_1b", "whisper_tiny"),
        (["--family", "datalab"], "chandra_ocr2", "granite_docling"),
        (["--family", "datalab"], "surya_ocr2_5", "granite_docling"),
    ],
)
def test_models_command_filters(args, present, absent):
    result = runner.invoke(app, ["models", *args], terminal_width=200)

    assert result.exit_code == 0, result.output
    assert present in result.output
    assert absent not in result.output


def test_models_command_rejects_an_unknown_family():
    result = runner.invoke(app, ["models", "--family", "nothing"])

    assert result.exit_code == 2


def test_models_command_writes_json_for_scripts():
    result = runner.invoke(app, ["models", "--asr", "--format", "json"])

    assert result.exit_code == 0, result.output
    models = json.loads(result.output)
    assert {model["kind"] for model in models} == {"asr"}
    default = next(model for model in models if model["default"])
    assert default["name"] == "whisper_tiny"
    assert "selected" in default["backends"].values()


def test_terminal_without_unicode_gets_ascii_symbols(monkeypatch: pytest.MonkeyPatch):
    legacy = Console(file=StringIO(), force_terminal=False)
    monkeypatch.setattr(type(legacy), "encoding", property(lambda self: "cp1252"))
    monkeypatch.setattr(main, "console", legacy)

    assert "".join(main._backend_symbols().values()).isascii()


def test_convert_help_points_to_the_table_and_does_not_list_the_presets():
    params = {
        param.name: param for param in get_command(app).commands["convert"].params
    }

    for name, command in (
        ("asr_model", "docling models --asr"),
        ("vlm_model", "docling models --vlm"),
    ):
        assert params[name].metavar == "NAME"
        assert command in params[name].help

    result = runner.invoke(app, ["convert", "--help"])
    assert result.exit_code == 0, result.output
    assert "whisper_tiny_en_s2t" not in result.output
    assert "nanonets_ocr2" not in result.output


def test_invalid_asr_model_is_still_rejected():
    result = runner.invoke(app, ["convert", "a.mp3", "--asr-model", "nothing"])

    assert result.exit_code == 2
