# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for VlmConvertModel cleanup when the engine fails to clean up."""

from unittest.mock import MagicMock

import pytest

from docling.models.stages.vlm_convert import vlm_convert_model
from docling.models.stages.vlm_convert.vlm_convert_model import VlmConvertModel

pytestmark = pytest.mark.ml_vlm


def _model_with_failing_cleanup() -> VlmConvertModel:
    model = VlmConvertModel.__new__(VlmConvertModel)
    model.engine = MagicMock()
    model.engine.cleanup.side_effect = RuntimeError("cleanup failed")
    return model


def test_del_logs_cleanup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    log = MagicMock()
    monkeypatch.setattr(vlm_convert_model, "_log", log)

    _model_with_failing_cleanup().__del__()

    log.warning.assert_called_once_with("Error cleaning up engine: cleanup failed")


def test_del_ignores_cleanup_failure_at_interpreter_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vlm_convert_model, "_log", None)

    _model_with_failing_cleanup().__del__()
