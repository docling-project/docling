# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Regression tests for the deprecated `force_full_page_ocr` compatibility flag.

`OcrOptions.force_full_page_ocr` is superseded by `mode=OcrMode.FULL_PAGE`; it is
kept as a deprecated view over `mode`, so the two cannot drift apart. The bridge
has to hold for attribute assignment on an already-built options object, not only
for constructor keywords, because that is how pre-deprecation code commonly
toggled the flag.
"""

import warnings

import pytest

from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrMode,
    RapidOcrOptions,
    TesseractOcrOptions,
)

OCR_OPTION_CLASSES = [EasyOcrOptions, RapidOcrOptions, TesseractOcrOptions]


@pytest.fixture(autouse=True)
def _silence_deprecation():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        yield


@pytest.mark.parametrize("options_cls", OCR_OPTION_CLASSES)
def test_force_full_page_ocr_via_constructor(options_cls):
    """The documented constructor path forces full-page OCR."""
    options = options_cls(force_full_page_ocr=True)
    assert options.mode == OcrMode.FULL_PAGE


@pytest.mark.parametrize("options_cls", OCR_OPTION_CLASSES)
def test_force_full_page_ocr_via_assignment(options_cls):
    """Setting the deprecated flag after construction must also force the mode.

    Model validators only run during validation, and these option models do not
    enable `validate_assignment`, so without an explicit bridge this assignment
    silently left `mode` at its default and full-page OCR never happened.
    """
    options = options_cls()
    assert options.mode == OcrMode.DEFAULT

    options.force_full_page_ocr = True

    assert options.mode == OcrMode.FULL_PAGE


@pytest.mark.parametrize("options_cls", OCR_OPTION_CLASSES)
def test_clearing_the_flag_leaves_an_explicit_mode_alone(options_cls):
    """Assigning a falsy flag must not reset a mode the caller chose."""
    options = options_cls(mode=OcrMode.LAYOUT_REGIONS)

    options.force_full_page_ocr = False

    assert options.mode == OcrMode.LAYOUT_REGIONS


@pytest.mark.parametrize("options_cls", OCR_OPTION_CLASSES)
def test_explicit_mode_still_wins_after_the_flag(options_cls):
    """The bridge must not pin `mode`; a later explicit mode still applies."""
    options = options_cls(force_full_page_ocr=True)
    assert options.mode == OcrMode.FULL_PAGE

    options.mode = OcrMode.LAYOUT_REGIONS

    assert options.mode == OcrMode.LAYOUT_REGIONS


@pytest.mark.parametrize("options_cls", OCR_OPTION_CLASSES)
def test_flag_reads_back_from_the_mode(options_cls):
    """The flag is a view over `mode`, so setting `mode` is enough."""
    options = options_cls()
    assert options.force_full_page_ocr is False

    options.mode = OcrMode.FULL_PAGE

    assert options.force_full_page_ocr is True


@pytest.mark.parametrize("options_cls", OCR_OPTION_CLASSES)
def test_flag_survives_a_serialization_roundtrip(options_cls):
    """A dump of a forced instance still validates back into full-page mode."""
    dumped = options_cls(force_full_page_ocr=True).model_dump()
    assert dumped["force_full_page_ocr"] is True

    assert options_cls.model_validate(dumped).mode == OcrMode.FULL_PAGE


@pytest.mark.parametrize("options_cls", OCR_OPTION_CLASSES)
def test_unrelated_assignment_is_unaffected(options_cls):
    """Assigning other fields keeps working and does not touch `mode`."""
    options = options_cls()

    options.lang = ["deu"]

    assert options.lang == ["deu"]
    assert options.mode == OcrMode.DEFAULT
