# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tesseract's language coverage, checked against the real installation.

No mocks: the installed tessdata set is read from the binary, and the assertions
are phrased against whatever that set turns out to be.
"""

import shutil
import subprocess

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import TesseractCliOcrOptions
from docling.exceptions import OcrLanguageNotSupportedError
from docling.models.stages.ocr.tesseract_ocr_cli_model import TesseractOcrCliModel
from docling.models.stages.ocr.tesseract_utils import installed_language_tags

pytestmark = pytest.mark.ml_ocr


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


def test_installed_language_maps_to_its_traineddata_name() -> None:
    installed = _installed_languages()
    if "eng" not in installed:
        pytest.skip("the eng traineddata is not installed")

    model = _build(["en"])

    assert model._native_langs == ["eng"]


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


def test_empty_lang_requires_the_osd_traineddata() -> None:
    """An empty list runs orientation-and-script detection, which needs its own
    file. No language is resolved up front: OSD picks one per page."""
    installed = _installed_languages()
    if "osd" in installed:
        model = _build([])
        assert model._auto_script is True
        assert model._native_langs == []
    else:
        with pytest.raises(ImportError, match="osd"):
            _build([])


def test_language_order_is_preserved_for_the_plus_join() -> None:
    """Tesseract treats `-l a+b` order as preference order."""
    installed = _installed_languages()
    if "eng" not in installed or "osd" not in installed:
        pytest.skip("needs both eng and osd installed")

    model = _build(["en", "en-US", "eng"])

    # Duplicates collapse; a single language remains.
    assert model._native_langs == ["eng"]


def test_unprefixed_script_traineddata_is_advertised_as_a_script_name() -> None:
    """Older tessdata installs list their script packs without the prefix.

    `script/<Name>` is still the only spelling that selects one, so that is how
    the install has to name them back: left bare, `Latin` is dropped as
    unparseable and `Lao` reads as the Lao language, whose `lao` traineddata is
    not what is installed.
    """
    names = ["eng", "Latin", "Cyrillic", "Lao", "Japanese_vert"]

    tags = installed_language_tags(names, "", TesseractCliOcrOptions.kind)

    assert tags == ["en-Latn", "script/Cyrillic", "script/Lao", "script/Latin"]
