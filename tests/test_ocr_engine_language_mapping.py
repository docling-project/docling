# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Per-engine translation from canonical tags to native codes.

Most of these run without any engine installed: each engine's table and mapping
are module-level or reachable on an uninitialized instance, which is what makes
the mapping reviewable at all. The exceptions are RapidOCR, whose PP-OCRv6
vocabulary is read from the installed `rapidocr`, and the last section, where what
an engine advertises depends on what is installed.
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
from docling.models.base_ocr_model import BaseOcrModel
from docling.models.stages.ocr.kserve_v2_ocr_model import KserveV2OcrModel
from docling.models.stages.ocr.rapid_ocr_model import (
    RapidOcrModel,
    _ppocr_code,
    _ppocr_supported_tags,
    _rapidocr_vocabulary,
)
from docling.models.stages.ocr.tesseract_ocr_cli_model import TesseractOcrCliModel
from docling.models.stages.ocr.tesseract_utils import language_to_tesseract_code
from docling.utils.ocr_language import (
    OcrLanguage,
    OcrLanguageResolver,
    OcrLanguageSupport,
)

_ONNX_VOCABULARY = _rapidocr_vocabulary("onnxruntime")
_TORCH_VOCABULARY = _rapidocr_vocabulary("torch")


# --- PP-OCR (RapidOCR) ------------------------------------------------------


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
        # The other two script families. PP-OCR has no `zu` or `ar` recognizer,
        # so both reach a model only through the script they are written in.
        ("zu", "latin"),
        ("ar", "arabic"),
    ],
)
def test_ppocr_tokens(tag: str, expected: str) -> None:
    assert (
        _ppocr_code(
            OcrLanguageResolver.canonicalize_ocr_language(tag), _ONNX_VOCABULARY
        )
        == expected
    )


@pytest.mark.parametrize("token", ["latin", "cyrillic", "arabic", "devanagari"])
def test_ppocr_script_recognizers_are_named_by_their_own_token(token: str) -> None:
    """These are real PP-OCR models with no language to canonicalize to, so they
    are carried through to the engine exactly as the user wrote them, once the
    `native:` prefix marks them as an engine token rather than a tag."""
    language = OcrLanguageResolver.canonicalize_ocr_language(f"native:{token}")

    assert language.is_passthrough
    assert _ppocr_code(language, _ONNX_VOCABULARY) == token


def test_ppocr_kannada_georgian_collision() -> None:
    """PP-OCR's `ka` is Kannada; BCP-47 `ka` is Georgian.

    Kannada must reach the `ka` recognizer, and Georgian must *not* -- it has no
    PP-OCR model at all, and silently serving it the Kannada one is the bug this
    guards.
    """
    assert (
        _ppocr_code(
            OcrLanguageResolver.canonicalize_ocr_language("kn"), _TORCH_VOCABULARY
        )
        == "ka"
    )
    assert (
        _ppocr_code(
            OcrLanguageResolver.canonicalize_ocr_language("ka"), _TORCH_VOCABULARY
        )
        is None
    )
    assert (
        _ppocr_code(
            OcrLanguageResolver.canonicalize_ocr_language("ka"), _ONNX_VOCABULARY
        )
        is None
    )


def test_ppocr_has_no_multilingual_model() -> None:
    assert (
        _ppocr_code(
            OcrLanguageResolver.canonicalize_ocr_language("mul"), _ONNX_VOCABULARY
        )
        is None
    )


def test_ppocr_non_default_script_uses_the_family() -> None:
    """PP-OCR's `az` and `uz` are the Latin ones, so a Cyrillic request for the
    same language must not silently pick the Latin recognizer."""
    assert (
        _ppocr_code(
            OcrLanguageResolver.canonicalize_ocr_language("az"), _ONNX_VOCABULARY
        )
        == "az"
    )
    assert (
        _ppocr_code(
            OcrLanguageResolver.canonicalize_ocr_language("az-Cyrl"), _ONNX_VOCABULARY
        )
        == "cyrillic"
    )
    assert (
        _ppocr_code(
            OcrLanguageResolver.canonicalize_ocr_language("uz-Cyrl"), _ONNX_VOCABULARY
        )
        == "cyrillic"
    )


def test_ppocr_supported_tags_are_canonical() -> None:
    """Every entry of the advertised list is a tag the user can ask for again.

    It fills the "Supported:" line of `OcrLanguageNotSupportedError`, so it is a
    list people copy from.
    """
    tags = _ppocr_supported_tags(_ONNX_VOCABULARY)

    assert "zh-Hans" in tags
    # Languages are rendered back as tags, never as PP-OCR's own tokens.
    assert "ch" not in tags
    for tag in tags:
        assert OcrLanguageResolver.canonicalize_ocr_language(tag).tag == tag


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


def test_another_engines_native_code_fails_at_the_engine() -> None:
    """`chi_sim` is tesseract's token, and the resolver no longer knows which
    engine was selected, so `native:chi_sim` is accepted as written.

    PP-OCR is the one that has to reject it, and its error has to name both the
    token as the user spelled it and the codes that would have worked.
    """
    assert RapidOcrOptions(lang=["native:chi_sim"]).lang == ["native:chi_sim"]

    model = _rapid_model("onnxruntime", ["native:chi_sim"])

    with pytest.raises(OcrLanguageNotSupportedError) as excinfo:
        model.resolve_ocr_languages()

    message = str(excinfo.value)
    assert "native:chi_sim" in message
    assert "Supported:" in message


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

# KServe canonicalizes nothing: the deployed model is the only authority on the
# languages it serves, so `lang` is neither validated nor mapped, only truncated
# to the one value the request carries.


def test_kserve_sends_the_engines_own_code_untouched() -> None:
    """`chi_sim` is another engine's code and `auto` is retired, yet both survive:
    only the deployment knows what it serves."""
    options = KserveV2OcrOptions(url="http://localhost:8000", lang=["chi_sim", "auto"])

    assert options.lang == ["chi_sim", "auto"]


def test_kserve_default_lang_is_not_canonicalized() -> None:
    options = KserveV2OcrOptions(url="http://localhost:8000")

    assert options.lang == ["english", "chinese"]


def test_kserve_warns_and_sends_the_first_language(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One language fits the request; the rest are dropped, but never silently."""
    options = KserveV2OcrOptions(
        url="http://localhost:8000", transport="http", lang=["japan", "korean"]
    )
    model = KserveV2OcrModel.__new__(KserveV2OcrModel)

    with caplog.at_level(logging.WARNING):
        KserveV2OcrModel.__init__(
            model,
            enabled=True,
            artifacts_path=None,
            options=options,
            accelerator_options=AcceleratorOptions(),
        )

    assert model._lang == "japan"
    assert "japan" in caplog.text and "korean" in caplog.text


def test_the_opt_out_does_not_leak_to_other_engines() -> None:
    """Only KServe skips canonicalization; a sibling still rewrites its tags."""
    assert RapidOcrOptions(lang=["deu"]).lang == ["de-Latn"]


# --- the base-class policy --------------------------------------------------
#
# `BaseOcrModel` decides two things on every engine's behalf: what an engine that
# has not overridden `map_ocr_language` does with a request, and what
# `resolve_ocr_languages` does with the codes it collects. Every engine in the
# tree overrides the first, so the fallback is only reachable through the base.


class _BareOcrModel(BaseOcrModel):
    """An engine that adds nothing: no table, no overrides, no installation.

    `language_support` is left at the base default, which is the conservative
    single-model, single-language engine.
    """

    def __init__(self, tags: list[str]) -> None:
        self.languages = OcrLanguageResolver.canonicalize_ocr_languages(tags)

    def __call__(self, conv_res, page_batch):  # pragma: no cover - never run
        raise NotImplementedError

    @classmethod
    def get_options_type(cls):  # pragma: no cover - never run
        raise NotImplementedError


class _BareMultilingualOcrModel(_BareOcrModel):
    language_support = OcrLanguageSupport(multiple_languages=True)


def test_the_default_mapping_is_the_primary_subtag() -> None:
    """Most ISO-639 engines want `de`, not `de-Latn`."""
    model = _BareMultilingualOcrModel(["de-DE", "zh-TW"])

    assert model.resolve_ocr_languages() == ["de", "zh"]


@pytest.mark.parametrize("tag", ["native:cyrillic", "mul"])
def test_the_default_mapping_refuses_what_it_cannot_name(tag: str) -> None:
    """A passthrough names *some* engine's script recognizer and `mul` names a
    multilingual model; an engine that declared neither has neither."""
    model = _BareOcrModel([tag])

    with pytest.raises(OcrLanguageNotSupportedError, match="explicit language"):
        model.resolve_ocr_languages()


def test_two_languages_sharing_a_native_code_are_joined_once() -> None:
    """Both written Norwegians are the one `nor` traineddata, and tesseract is
    handed the result as `-l nor`, not `-l nor+nor`."""
    model = TesseractOcrCliModel.__new__(TesseractOcrCliModel)
    # Stand in for the `--list-langs` probe: the join is what is under test, not
    # which files happen to be installed on the machine running this.
    model._tesseract_vocabulary = ["nor", "deu"]
    model.languages = OcrLanguageResolver.canonicalize_ocr_languages(["nb", "nn", "de"])

    assert model.resolve_ocr_languages() == ["nor", "deu"]


def test_a_single_language_engine_keeps_the_first_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """List order is preference order, and the drop is never silent."""
    model = _BareOcrModel(["de", "fr"])

    with caplog.at_level(logging.WARNING):
        assert model.resolve_ocr_languages() == ["de"]

    assert "de-Latn" in caplog.text and "fr-Latn" in caplog.text


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
        language_to_tesseract_code(OcrLanguageResolver.canonicalize_ocr_language(tag))
        == expected
    )


def test_tesseract_script_files_pass_through_verbatim() -> None:
    """A script file names its traineddata directly, in the install's spelling."""
    assert (
        language_to_tesseract_code(OcrLanguage(native="script/Latin")) == "script/Latin"
    )
    assert language_to_tesseract_code(OcrLanguage(native="Cyrillic")) == "Cyrillic"


def test_tesseract_has_no_file_for_mul() -> None:
    """`mul` has no tessdata equivalent; an empty list drives per-page OSD."""
    assert (
        language_to_tesseract_code(OcrLanguageResolver.canonicalize_ocr_language("mul"))
        is None
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
