# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Per-engine translation from canonical tags to native codes.

Most of these run without any engine installed: each engine's table and mapping
are module-level or reachable on an uninitialized instance, which is what makes
the mapping reviewable at all. The last section is the exception -- what an
engine advertises depends on what is installed, so those tests need the engine.
"""

import logging
import shutil
import sys

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    KserveV2OcrOptions,
    OcrMacOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.exceptions import OcrLanguageNotSupportedError
from docling.models.stages.ocr.kserve_v2_ocr_model import KserveV2OcrModel
from docling.models.stages.ocr.ppocr_languages import (
    PPOCRV4_CODES,
    PPOCRV5_CODES,
    PPOCRV6_CODES,
    ppocr_code,
    ppocr_supported_tags,
)
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
from docling.models.stages.ocr.tesseract_utils import tesseract_code
from docling.utils.ocr_language import OcrLanguageResolver

_ONNX_VOCABULARY = PPOCRV6_CODES | PPOCRV5_CODES | PPOCRV4_CODES
_TORCH_VOCABULARY = PPOCRV6_CODES | PPOCRV4_CODES


# --- PP-OCR (RapidOCR and the KServe client share one table) ----------------


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("zh-Hans", "ch"),
        ("zh-Hant", "chinese_cht"),
        ("ja", "japan"),
        ("en", "en"),
        ("de", "de"),
        ("sr-Latn", "rs_latin"),
        # East Slavic has its own, narrower recognizer.
        ("ru", "eslav"),
        ("uk", "eslav"),
        ("be", "eslav"),
        # Any other Cyrillic language falls back to the script family.
        ("sr", "cyrillic"),
        ("mn", "cyrillic"),
        ("el", "el"),
        ("th", "th"),
        ("hi", "devanagari"),
    ],
)
def test_ppocr_tokens(tag: str, expected: str) -> None:
    assert (
        ppocr_code(OcrLanguageResolver.canonicalize_ocr_language(tag), _ONNX_VOCABULARY)
        == expected
    )


@pytest.mark.parametrize("token", ["latin", "cyrillic", "arabic", "devanagari"])
def test_ppocr_script_recognizers_are_named_by_their_own_token(token: str) -> None:
    """These are real PP-OCR models with no language to canonicalize to, so they
    are carried through to the engine exactly as the user wrote them, once the
    `native:` prefix marks them as an engine token rather than a tag."""
    language = OcrLanguageResolver.canonicalize_ocr_language(f"native:{token}")

    assert language.is_passthrough
    assert ppocr_code(language, _ONNX_VOCABULARY) == token


def test_ppocr_kannada_georgian_collision() -> None:
    """PP-OCR's `ka` is Kannada; BCP-47 `ka` is Georgian.

    Kannada must reach the `ka` recognizer, and Georgian must *not* -- it has no
    PP-OCR model at all, and silently serving it the Kannada one is the bug this
    guards.
    """
    assert (
        ppocr_code(
            OcrLanguageResolver.canonicalize_ocr_language("kn"), _TORCH_VOCABULARY
        )
        == "ka"
    )
    assert (
        ppocr_code(
            OcrLanguageResolver.canonicalize_ocr_language("ka"), _TORCH_VOCABULARY
        )
        is None
    )
    assert (
        ppocr_code(
            OcrLanguageResolver.canonicalize_ocr_language("ka"), _ONNX_VOCABULARY
        )
        is None
    )


def test_ppocr_has_no_multilingual_model() -> None:
    assert (
        ppocr_code(
            OcrLanguageResolver.canonicalize_ocr_language("mul"), _ONNX_VOCABULARY
        )
        is None
    )


def test_ppocr_non_default_script_uses_the_family() -> None:
    """PP-OCR's `az` and `uz` are the Latin ones, so a Cyrillic request for the
    same language must not silently pick the Latin recognizer."""
    assert (
        ppocr_code(
            OcrLanguageResolver.canonicalize_ocr_language("az"), _ONNX_VOCABULARY
        )
        == "az"
    )
    assert (
        ppocr_code(
            OcrLanguageResolver.canonicalize_ocr_language("az-Cyrl"), _ONNX_VOCABULARY
        )
        == "cyrillic"
    )
    assert (
        ppocr_code(
            OcrLanguageResolver.canonicalize_ocr_language("uz-Cyrl"), _ONNX_VOCABULARY
        )
        == "cyrillic"
    )


def test_ppocr_supported_tags_are_canonical() -> None:
    tags = ppocr_supported_tags(_ONNX_VOCABULARY)

    assert "zh-Hans" in tags
    # Languages are rendered back as tags, never as PP-OCR tokens...
    assert "ch" not in tags
    # ...but a script recognizer is named back as itself: that is what selects it.
    assert "cyrillic" in tags


# --- RapidOCR backend routing ----------------------------------------------


def _rapid_model(backend: str, lang: list[str]) -> RapidOcrModel:
    model = RapidOcrModel.__new__(RapidOcrModel)
    model.options = RapidOcrOptions(backend=backend, lang=lang)
    model.languages = tuple(
        OcrLanguageResolver.canonicalize_ocr_language(tag) for tag in model.options.lang
    )
    return model


@pytest.mark.parametrize("backend", ["onnxruntime", "torch"])
def test_rapidocr_georgian_is_a_coverage_error_on_every_backend(backend: str) -> None:
    """Georgian has no PP-OCR recognizer on any backend.

    It has to be asked for as `ka-Geor`: a bare `ka` given to RapidOCR is PP-OCR's
    own token for Kannada, which is the reading RapidOCR users expect.
    """
    model = _rapid_model(backend, ["ka-Geor"])

    with pytest.raises(OcrLanguageNotSupportedError) as excinfo:
        model.resolve_ocr_languages()

    message = str(excinfo.value)
    assert "ka-Geor" in message
    assert backend in message
    # The message must name what the user *can* ask for.
    assert "Supported:" in message


def test_rapidocr_native_ka_is_ppocr_kannada() -> None:
    """`native:ka` names PP-OCR's Kannada recognizer; bare `ka` is BCP-47 Georgian."""
    options = RapidOcrOptions(backend="torch", lang=["native:ka"])
    assert options.lang == ["native:ka"]
    assert _rapid_model("torch", ["native:ka"]).resolve_ocr_languages() == ["ka"]


def test_rapidocr_warns_and_truncates_extra_languages(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _rapid_model("onnxruntime", ["de", "fr", "en"])

    with caplog.at_level(logging.WARNING):
        assert model.resolve_ocr_languages() == ["de"]

    warning = caplog.text
    assert "de-Latn" in warning
    assert "fr-Latn" in warning and "en-Latn" in warning
    assert "preference" in warning


# --- KServe v2 --------------------------------------------------------------


def _kserve_model(lang: list[str]) -> KserveV2OcrModel:
    model = KserveV2OcrModel.__new__(KserveV2OcrModel)
    model.options = KserveV2OcrOptions(url="http://localhost:8000", lang=lang)
    model.languages = tuple(
        OcrLanguageResolver.canonicalize_ocr_language(tag) for tag in model.options.lang
    )
    return model


def test_kserve_warns_instead_of_silently_dropping_languages(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The extra languages used to be discarded with no message at all."""
    model = _kserve_model(["en", "zh-Hans", "ar"])

    with caplog.at_level(logging.WARNING):
        assert model.resolve_ocr_languages() == ["en"]

    assert "en-Latn" in caplog.text
    assert "zh-Hans" in caplog.text and "ar-Arab" in caplog.text


def test_kserve_maps_to_ppocr_tokens() -> None:
    assert _kserve_model(["zh-Hant"]).resolve_ocr_languages() == ["chinese_cht"]


def test_kserve_empty_lang_uses_the_ppocr_default() -> None:
    """No tag means "let the engine decide"; PP-OCR's own default is `ch`."""
    assert _kserve_model([]).resolve_ocr_languages() == ["ch"]


# --- Tesseract --------------------------------------------------------------


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        # The vocabulary *is* ISO 639-2/T, so most tags need no table entry.
        ("de", "deu"),
        ("fr", "fra"),
        ("el", "ell"),
        ("cs", "ces"),
        ("en", "eng"),
        ("kn", "kan"),
        # Georgian is `kat` here -- no collision, unlike PP-OCR.
        ("ka", "kat"),
        # ...and the deviations that do.
        ("zh-Hans", "chi_sim"),
        ("zh-Hant", "chi_tra"),
        ("sr", "srp"),
        ("sr-Latn", "srp_latn"),
        ("az-Cyrl", "aze_cyrl"),
        ("az", "aze"),
        ("uz-Cyrl", "uzb_cyrl"),
        ("ku", "kmr"),
        ("nb", "nor"),
        ("nn", "nor"),
    ],
)
def test_tesseract_language_names(tag: str, expected: str) -> None:
    assert (
        tesseract_code(OcrLanguageResolver.canonicalize_ocr_language(tag), "script/")
        == expected
    )


def test_tesseract_script_files_follow_the_installed_prefix() -> None:
    """A script file is always written `script/<Name>`, but older tessdata
    installs list those files unprefixed, so the prefix is re-applied."""
    latin = OcrLanguageResolver.canonicalize_ocr_language("script/Latin")

    assert tesseract_code(latin, "script/") == "script/Latin"
    assert tesseract_code(latin, "") == "Latin"
    cyrl = OcrLanguageResolver.canonicalize_ocr_language("script/Cyrillic")
    assert tesseract_code(cyrl, "") == "Cyrillic"


def test_tesseract_has_no_file_for_mul() -> None:
    """`mul` has no tessdata equivalent; an empty list drives per-page OSD."""
    assert (
        tesseract_code(OcrLanguageResolver.canonicalize_ocr_language("mul"), "") is None
    )


# --- what an engine advertises must be requestable --------------------------
#
# `supported_ocr_languages()` fills the "Supported:" line of
# `OcrLanguageNotSupportedError`, so it is a list users copy from. Every tag in
# it therefore has to survive being asked for again -- which is exactly what
# three engines got wrong: EasyOCR offered `av-Cyrl`/`ce-Cyrl` under codes it
# does not have, Tesseract offered the `script/*_vert` files the resolver
# refuses, and ocrmac offered Vision's own `vi-VT`, which is not a tag at all.


def _assert_every_advertised_tag_is_requestable(model) -> None:
    advertised = model.supported_ocr_languages()
    assert advertised, "the engine reported no languages at all"
    unusable = []
    for tag in advertised:
        try:
            model.map_ocr_language(OcrLanguageResolver.canonicalize_ocr_language(tag))
        except (ValueError, OcrLanguageNotSupportedError) as exc:
            unusable.append((tag, str(exc)))
    assert not unusable


def test_easyocr_advertises_only_languages_it_serves() -> None:
    pytest.importorskip("easyocr")
    from docling.models.stages.ocr.easyocr_model import EasyOcrModel

    model = EasyOcrModel(
        enabled=False,
        artifacts_path=None,
        options=EasyOcrOptions(),
        accelerator_options=AcceleratorOptions(),
    )

    _assert_every_advertised_tag_is_requestable(model)


def test_tesseract_advertises_only_languages_it_serves() -> None:
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract binary not installed")
    from docling.models.stages.ocr.tesseract_ocr_cli_model import TesseractOcrCliModel

    model = TesseractOcrCliModel(
        enabled=True,
        artifacts_path=None,
        options=TesseractCliOcrOptions(lang=["en"]),
        accelerator_options=AcceleratorOptions(),
    )

    _assert_every_advertised_tag_is_requestable(model)


@pytest.mark.skipif(sys.platform != "darwin", reason="ocrmac is macOS-only")
def test_ocrmac_advertises_only_languages_it_serves() -> None:
    pytest.importorskip("ocrmac")
    from docling.models.stages.ocr.ocr_mac_model import OcrMacModel

    model = OcrMacModel(
        enabled=True,
        artifacts_path=None,
        options=OcrMacOptions(),
        accelerator_options=AcceleratorOptions(),
    )

    _assert_every_advertised_tag_is_requestable(model)
