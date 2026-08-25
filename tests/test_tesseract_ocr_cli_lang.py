# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from unittest.mock import patch

import pandas as pd
import pytest

from docling.models.stages.ocr.tesseract_ocr_cli_model import TesseractOcrCliModel

_MODULE = "docling.models.stages.ocr.tesseract_ocr_cli_model"


class _FakeCompletedProcess:
    def __init__(self, stdout: bytes) -> None:
        self.stdout = stdout


def _model_for_listing(listing: str) -> TesseractOcrCliModel:
    """Build a model whose language list comes from the given `--list-langs` output."""
    model = TesseractOcrCliModel.__new__(TesseractOcrCliModel)
    model._safe_tesseract_cmd = "tesseract"
    with patch(
        f"{_MODULE}.subprocess.run",
        return_value=_FakeCompletedProcess(listing.encode("utf-8")),
    ):
        model._set_languages_and_prefix()
    return model


@pytest.mark.parametrize("sep", ["/", "\\"], ids=["posix", "windows"])
def test_script_packs_are_listed_with_either_separator(sep: str):
    """Windows tesseract prints `script\\Arabic`; the prefix must still be detected."""
    model = _model_for_listing(
        f"List of available languages (3):\neng\nscript{sep}Arabic\nscript{sep}Latin\n"
    )
    assert model._script_prefix == "script/"
    assert "script/Arabic" in model._tesseract_languages


def test_detected_script_resolves_against_a_windows_listing():
    """lang=["auto"] must resolve the detected script to an installed pack."""
    model = _model_for_listing(
        "List of available languages (2):\neng\nscript\\Arabic\n"
    )
    osd = pd.DataFrame({"key": ["Script"], "value": ["Arabic"]})

    lang = model._parse_language(osd)

    assert lang == "script/Arabic"
    # the resolved identifier is passed to tesseract via _sanitize_lang, which
    # rejects backslashes outright
    assert TesseractOcrCliModel._sanitize_lang(lang) == "script/Arabic"
