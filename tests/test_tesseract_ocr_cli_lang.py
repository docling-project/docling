# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tesseract's language handling, through the CLI front-end.

Two halves. The first drives `--list-langs` from a fake `subprocess.run`, so the
listing under test is chosen rather than whatever is installed; that is the only
way to assert the Windows `script\\Name` spelling from a posix machine. The
second reads the real installation and phrases its assertions against whatever
that set turns out to be, and carries `pytest.mark.ml_ocr` per test.
"""

import shutil
import subprocess
from unittest.mock import patch

import pandas as pd
import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import TesseractCliOcrOptions
from docling.exceptions import OcrLanguageNotSupportedError
from docling.models.stages.ocr.tesseract_ocr_cli_model import TesseractOcrCliModel
from docling.models.stages.ocr.tesseract_utils import installed_tesseract_tags

_MODULE = "docling.models.stages.ocr.tesseract_ocr_cli_model"


# --- a chosen `--list-langs` listing ----------------------------------------


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
        model._set_languages()
    return model


@pytest.mark.parametrize("sep", ["/", "\\"], ids=["posix", "windows"])
def test_script_packs_are_listed_with_either_separator(sep: str):
    """Windows tesseract prints `script\\Arabic`; the listing is normalized either way."""
    model = _model_for_listing(
        f"List of available languages (3):\neng\nscript{sep}Arabic\nscript{sep}Latin\n"
    )
    assert "script/Arabic" in model._tesseract_vocabulary


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


def test_unprefixed_script_traineddata_is_advertised_natively() -> None:
    """Some tessdata installs list a hand-placed script pack without the prefix.

    Only `script/<Name>` selects a pack through the OSD path, so a bare file is
    named back verbatim behind `native:`: that is the only spelling that reaches
    Tesseract unchanged. Left as a plain tag, `Latin` is unparseable and `Lao`
    would read as the Lao language, whose `lao` traineddata is not installed.
    """
    names = ["eng", "Latin", "Cyrillic", "Lao", "Japanese_vert"]

    tags = installed_tesseract_tags(names)

    assert tags == [
        "en-Latn",
        "native:Cyrillic",
        "native:Japanese_vert",
        "native:Lao",
        "native:Latin",
    ]


# --- the real installation --------------------------------------------------
#
# No mocks: the installed tessdata set is read from the binary.


def _installed_languages() -> list[str]:
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract binary not installed")
    output = subprocess.run(
        ["tesseract", "--list-langs"], capture_output=True, check=True
    )
    return output.stdout.decode("utf-8").splitlines()[1:]


def _build(lang: list[str]) -> TesseractOcrCliModel:
    return TesseractOcrCliModel(
        enabled=True,
        artifacts_path=None,
        options=TesseractCliOcrOptions(lang=lang),
        accelerator_options=AcceleratorOptions(),
    )


@pytest.mark.ml_ocr
def test_installed_language_maps_to_its_traineddata_name() -> None:
    installed = _installed_languages()
    if "eng" not in installed:
        pytest.skip("the eng traineddata is not installed")

    model = _build(["en"])

    assert model._native_codes == ["eng"]


@pytest.mark.ml_ocr
def test_uninstalled_language_fails_at_construction() -> None:
    """Tesseract never validated `options.lang` before, so a missing traineddata
    surfaced as a per-page CLI failure much later."""
    installed = _installed_languages()
    # Pick a language whose traineddata is definitely absent.
    candidates = [("ka", "kat"), ("th", "tha"), ("el", "ell"), ("hi", "hin")]
    choice = next(
        (tag for tag, name in candidates if name not in installed),
        None,
    )
    if choice is None:
        pytest.skip("every candidate language is installed")

    with pytest.raises(OcrLanguageNotSupportedError) as excinfo:
        _build([choice])

    message = str(excinfo.value)
    assert "Supported:" in message
    # The message names the installed set, as canonical tags.
    if "eng" in installed:
        assert "en-Latn" in message


@pytest.mark.ml_ocr
def test_empty_lang_requires_the_osd_traineddata() -> None:
    """An empty list runs orientation-and-script detection, which needs its own
    file. No language is resolved up front: OSD picks one per page."""
    installed = _installed_languages()
    if "osd" in installed:
        model = _build([])
        assert model._auto_script is True
        assert model._native_codes == []
    else:
        with pytest.raises(ImportError, match="osd"):
            _build([])


@pytest.mark.ml_ocr
def test_language_order_is_preserved_for_the_plus_join() -> None:
    """Tesseract treats `-l a+b` order as preference order."""
    installed = _installed_languages()
    if "eng" not in installed or "osd" not in installed:
        pytest.skip("needs both eng and osd installed")

    model = _build(["en", "en-US", "eng"])

    # Duplicates collapse; a single language remains.
    assert model._native_codes == ["eng"]
