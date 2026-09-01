# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Docling's OCR language policy: canonicalize to BCP-47, or refuse.

These assert docling decisions -- drop the region, keep the script, never
reject `und`, reject the retired engine vocabularies -- rather than langcodes
behaviour.

Four sections, following one user-supplied string all the way to a canonical tag:
the resolver itself, the engine-native vocabularies it accepts alongside BCP-47,
the `OcrOptions` validator that calls it, and the `--ocr-lang` CLI flag that
feeds that validator.
"""

import warnings
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from docling.cli.main import app
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    NemotronOcrOptions,
    OcrAutoOptions,
    OcrMacOptions,
    OcrMode,
    RapidOcrOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.datamodel.settings import DEFAULT_PAGE_RANGE
from docling.models.stages.ocr.auto_ocr_model import OcrAutoModel
from docling.models.stages.ocr.easyocr_model import EasyOcrModel
from docling.models.stages.ocr.tesseract_utils import tesseract_code
from docling.utils.ocr_language import (
    OcrLanguage,
    OcrLanguageResolver,
)


def _canonical_tags(values: list[str]) -> list[str]:
    """The tags `OcrOptions.lang` would store for `values`."""
    return [
        language.tag
        for language in OcrLanguageResolver.canonicalize_ocr_languages(values)
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # ISO 639-2/B and /T both fold onto the 639-1 subtag; region is dropped.
        ("de", "de-Latn"),
        ("de-DE", "de-Latn"),
        ("deu", "de-Latn"),
        ("ger", "de-Latn"),
        ("DE", "de-Latn"),
        ("en", "en-Latn"),
        ("en-US", "en-Latn"),
        ("eng", "en-Latn"),
        # Simplified vs Traditional is a script distinction no ISO 639 code has.
        ("zh", "zh-Hans"),
        ("zh-CN", "zh-Hans"),
        ("zh-Hans", "zh-Hans"),
        ("zho", "zh-Hans"),
        ("zh-TW", "zh-Hant"),
        ("zh-HK", "zh-Hant"),
        ("zh-Hant", "zh-Hant"),
        # Likely-subtags supply the script the user left out.
        ("sr", "sr-Cyrl"),
        ("sr-Latn", "sr-Latn"),
        ("pa", "pa-Guru"),
        ("pa-IN", "pa-Guru"),
        ("pa-PK", "pa-Arab"),
        ("ja", "ja-Jpan"),
        ("jpn", "ja-Jpan"),
        ("ru", "ru-Cyrl"),
        # The reserved tag passes through untouched.
        ("mul", "mul"),
        ("  de  ", "de-Latn"),
    ],
)
def test_canonicalization(value: str, expected: str) -> None:
    assert OcrLanguageResolver.canonicalize_ocr_language(value).tag == expected


@pytest.mark.parametrize("value", ["und", "und-Latn", "und-latn", "und-Cyrl"])
def test_undetermined_tags_are_rejected(value: str) -> None:
    """Docling has no "undetermined" language and no script families.

    `Language.get("und-Latn").maximize()` is `en-Latn-US`, so accepting these
    would silently turn "any Latin-script document" into "English". The empty
    list carries the "let the engine decide" meaning instead.
    """
    with pytest.raises(ValueError, match="no script families"):
        OcrLanguageResolver.canonicalize_ocr_language(value)


@pytest.mark.parametrize(
    ("value", "hint"),
    [
        ("chinese", "zh-Hans"),
        # `ch` is valid BCP-47 for Chamorro, so validity alone would accept it.
        ("ch", "zh-Hans"),
        ("ch_sim", "zh-Hans"),
        ("chi_sim", "zh-Hans"),
        ("chi_tra", "zh-Hant"),
        ("chinese_cht", "zh-Hant"),
        ("english", "en"),
        ("japan", "ja"),
        ("korean", "ko"),
        ("multilingual", "mul"),
        ("eslav", "ru"),
        ("rs_latin", "sr-Latn"),
        ("tjk", "tg"),
        ("aze_cyrl", "az-Cyrl"),
    ],
)
def test_retired_engine_tokens_name_their_replacement(value: str, hint: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        OcrLanguageResolver.canonicalize_ocr_language(value)

    assert f"'{hint}'" in str(excinfo.value)


@pytest.mark.parametrize("value", ["klingon", "", "   ", "de-DE-DE", "zz"])
def test_malformed_tags_raise(value: str) -> None:
    with pytest.raises(ValueError):
        OcrLanguageResolver.canonicalize_ocr_language(value)


def test_chamorro_is_not_chinese() -> None:
    """RapidOCR's `ch` means Chinese, but BCP-47 `ch` is Chamorro.

    Accepting it would resolve to the wrong language and surface as a confusing
    coverage error much later, so it is rejected up front.
    """
    with pytest.raises(ValueError, match="zh-Hans"):
        OcrLanguageResolver.canonicalize_ocr_language("ch")


def test_reserved_tag_must_stand_alone() -> None:
    for combination in (["mul", "en"], ["en", "mul"]):
        with pytest.raises(ValueError, match="on its own"):
            OcrLanguageResolver.canonicalize_ocr_languages(combination)


@pytest.mark.parametrize("value", ["zxx", "ZXX", "  zxx  "])
def test_no_linguistic_content_is_not_an_ocr_language(value: str) -> None:
    """`zxx` used to disable the engine; skipping OCR is a pipeline switch, not a
    language, and langcodes would otherwise read the tag as `zxx-Latn`."""
    with pytest.raises(ValueError, match="do_ocr=False"):
        OcrLanguageResolver.canonicalize_ocr_language(value)


def test_empty_list_means_the_engine_decides() -> None:
    """No tag carries that meaning any more, so the empty list has to."""
    assert OcrLanguageResolver.canonicalize_ocr_languages([]) == []


@pytest.mark.parametrize(
    "value", ["auto", "osd", "latin", "cyrillic", "arabic", "devanagari", "bengali"]
)
def test_engine_decides_and_script_tokens_point_at_the_empty_list(value: str) -> None:
    """These named a script or an auto mode, neither of which is a language.

    PP-OCR defines four of them as real recognizers, so they resolve as
    passthroughs for that engine and only reach this path for everyone else.
    """
    with pytest.raises(ValueError, match="leave the OCR language list empty"):
        OcrLanguageResolver.canonicalize_ocr_language(value)


def test_duplicates_collapse_and_order_is_preserved() -> None:
    """Order is preference order for engines that join languages (Tesseract `+`)."""
    assert _canonical_tags(["fr", "de", "fr-FR", "en-GB", "deu"]) == [
        "fr-Latn",
        "de-Latn",
        "en-Latn",
    ]


def test_canonicalization_is_idempotent() -> None:
    once = _canonical_tags(["deu", "zh-TW", "sr-Latn", "pa-PK"])

    assert _canonical_tags(once) == once


def test_has_default_script_separates_the_script_variants() -> None:
    """Engines key on this to decide whether the primary subtag is enough."""
    assert OcrLanguageResolver.canonicalize_ocr_language("de").has_default_script
    assert OcrLanguageResolver.canonicalize_ocr_language("sr").has_default_script
    assert not OcrLanguageResolver.canonicalize_ocr_language(
        "sr-Latn"
    ).has_default_script
    assert not OcrLanguageResolver.canonicalize_ocr_language(
        "az-Cyrl"
    ).has_default_script


def test_ocr_language_is_hashable() -> None:
    """Engines key dicts and caches on the canonical pair."""
    assert {OcrLanguage(bcp47_language="de", bcp47_script="Latn")} == {
        OcrLanguageResolver.canonicalize_ocr_language("de-DE")
    }


def test_match_against_a_region_bearing_vocabulary() -> None:
    """Apple Vision's own vocabulary is BCP-47 with regions."""
    supported = ["en-US", "fr-FR", "de-DE", "pt-BR", "zh-Hans"]

    assert (
        OcrLanguageResolver.match_ocr_language(
            OcrLanguageResolver.canonicalize_ocr_language("de"), supported
        )
        == "de-DE"
    )
    assert (
        OcrLanguageResolver.match_ocr_language(
            OcrLanguageResolver.canonicalize_ocr_language("pt"), supported
        )
        == "pt-BR"
    )
    assert (
        OcrLanguageResolver.match_ocr_language(
            OcrLanguageResolver.canonicalize_ocr_language("zh-CN"), supported
        )
        == "zh-Hans"
    )
    assert (
        OcrLanguageResolver.match_ocr_language(
            OcrLanguageResolver.canonicalize_ocr_language("th"), supported
        )
        is None
    )


# --- engine-native vocabularies --------------------------------------------


@pytest.mark.parametrize(
    "token", ["jpn_vert", "chi_tra_vert", "script/HanS_vert", "ita_old", "equ"]
)
def test_tokens_no_language_tag_can_express_are_refused(token: str) -> None:
    """Rather than silently resolving to a different recognizer.

    `ita_old` in particular used to canonicalize to plain `it-Latn`, quietly
    selecting the modern Italian model for a historical-orthography request.
    """
    with pytest.raises(ValueError, match="cannot address by language tag"):
        OcrLanguageResolver.canonicalize_ocr_language(token)


@pytest.mark.parametrize(
    "value",
    [
        "klingon",  # not a valid BCP-47 tag
        "chi_sim",  # a legacy hint
        "osd",  # an auto token
        "script/HanS_vert",  # vertical text
    ],
)
def test_canonicalize_can_answer_none_instead_of_raising(value: str) -> None:
    """The four routes into the failure guard, for the vocabulary builders.

    They advertise what an engine serves, where a rejection reason never reaches
    a user; the default keeps the `ValueError` that carries one.
    """
    assert (
        OcrLanguageResolver.canonicalize_ocr_language(value, raise_exception=False)
        is None
    )
    with pytest.raises(ValueError):
        OcrLanguageResolver.canonicalize_ocr_language(value)


def test_unrepresentable_tokens_are_scoped_to_the_engine_that_owns_them() -> None:
    """`equ` asked of RapidOCR is better served by "not a valid tag"."""
    with pytest.raises(ValueError, match="BCP-47"):
        OcrLanguageResolver.canonicalize_ocr_language("equ")


@pytest.mark.parametrize(
    ("options_cls", "native", "expected"),
    [
        (RapidOcrOptions, ["native:ch"], ["native:ch"]),
        (RapidOcrOptions, ["native:chinese_cht"], ["native:chinese_cht"]),
        (TesseractOcrOptions, ["native:chi_tra"], ["native:chi_tra"]),
        # Only the prefix is case insensitive: a tessdata script file is
        # TitleCase and has to reach the engine spelled as it is installed.
        (
            TesseractCliOcrOptions,
            ["native:script/Cyrillic"],
            ["native:script/Cyrillic"],
        ),
        (
            EasyOcrOptions,
            ["native:ch_sim", "native:ang"],
            ["native:ch_sim", "native:ang"],
        ),
        (NemotronOcrOptions, ["native:multilingual"], ["native:multilingual"]),
        (OcrAutoOptions, ["native:chinese"], ["native:chinese"]),
        # ocrmac has no native vocabulary: Vision's codes already are BCP-47.
        (OcrMacOptions, ["en-US"], ["en-Latn"]),
    ],
)
def test_options_accept_their_own_engines_codes(
    options_cls: type, native: list[str], expected: list[str]
) -> None:
    """A `native:` token is stored as written, so revalidating `lang` cannot move it."""
    assert options_cls(lang=native).lang == expected


def test_options_reject_a_clashing_token_when_no_engine_is_chosen() -> None:
    with pytest.raises(ValidationError, match="zh-Hans"):
        OcrAutoOptions(lang=["ch"])


def test_tesseract_fraktur_keeps_its_own_traineddata() -> None:
    """`de-Latf` used to flatten to `deu` through to_alpha3(), losing Fraktur."""
    assert (
        tesseract_code(
            OcrLanguageResolver.canonicalize_ocr_language("de-Latf"), "script/"
        )
        == "deu_latf"
    )
    assert (
        tesseract_code(
            OcrLanguageResolver.canonicalize_ocr_language("frk"),
            "script/",
        )
        == "deu_latf"
    )


# --- the options validator --------------------------------------------------
#
# `OcrOptions` declares one `@field_validator("lang")`, but every engine subclass
# *redefines* `lang` with its own default and its own `ConfigDict`. The design
# assumes pydantic collects validators by field name across the MRO and merges
# `model_config` down it. Both are asserted here, because a silent regression
# would let an engine-native token through unvalidated.

_OPTION_CLASSES = [
    OcrAutoOptions,
    RapidOcrOptions,
    NemotronOcrOptions,
    EasyOcrOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
    OcrMacOptions,
]


def _build(cls, **kwargs):
    return cls(**kwargs)


@pytest.mark.parametrize("cls", _OPTION_CLASSES)
def test_base_validator_fires_on_every_subclass(cls) -> None:
    options = _build(cls, lang=["deu", "en-US", "zh-TW"])

    assert options.lang == ["de-Latn", "en-Latn", "zh-Hant"]


@pytest.mark.parametrize("cls", _OPTION_CLASSES)
def test_defaults_are_already_canonical(cls) -> None:
    """`validate_default=True` makes this an assertion, not a rewrite."""
    default = _build(cls).lang

    assert _canonical_tags(default) == default


@pytest.mark.parametrize("cls", _OPTION_CLASSES)
def test_retired_tokens_are_rejected(cls) -> None:
    with pytest.raises(ValidationError, match="no script families"):
        _build(cls, lang=["auto"])


@pytest.mark.parametrize("cls", _OPTION_CLASSES)
def test_empty_lang_is_accepted(cls) -> None:
    """An empty list is how "let the engine decide" is spelled."""
    assert _build(cls, lang=[]).lang == []


def test_assignment_is_validated() -> None:
    """The SDK mutation path documented in the FAQ goes through the validator.

    Proves `validate_assignment` on the base survives the subclass `ConfigDict`.
    """
    options = EasyOcrOptions()

    options.lang = ["fra", "de-DE"]
    assert options.lang == ["fr-Latn", "de-Latn"]

    with pytest.raises(ValidationError, match="zh-Hans"):
        options.lang = ["chinese"]


def test_force_full_page_ocr_bridge_survives_validate_assignment() -> None:
    """The deprecated flag assigns `mode` from inside a model validator, which
    `validate_assignment` re-enters; it must settle rather than recurse."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        options = EasyOcrOptions(force_full_page_ocr=True)
        assert options.mode is OcrMode.FULL_PAGE

        options.lang = ["en"]
        assert options.mode is OcrMode.FULL_PAGE

        options.scale = 2.0
        assert options.mode is OcrMode.FULL_PAGE


def test_serialized_options_round_trip() -> None:
    options = TesseractCliOcrOptions(lang=["fra", "deu"])

    restored = TesseractCliOcrOptions.model_validate(options.model_dump())

    assert restored.lang == options.lang == ["fr-Latn", "de-Latn"]


# --- the CLI ----------------------------------------------------------------
#
# `--ocr-lang`: construction, canonicalization, and error reporting.

# TERM=dumb disables the Rich styling in the CI
runner = CliRunner(env={"TERM": "dumb"})

_SOURCE = "./tests/data/pdf/sources/2305.03393v1-pg9.pdf"


def _flat_cli_output(output: str) -> str:
    """The error box still wraps and draws borders: flatten it to one line."""
    return " ".join(output.replace("│", "").split())


def _capture_ocr_options(monkeypatch, extra_args: list[str], tmp_path: Path):
    captured: dict[str, Any] = {}

    class _FakeDocumentConverter:
        def __init__(self, *, allowed_formats, format_options):
            pdf_option = format_options[InputFormat.PDF]
            captured["ocr_options"] = pdf_option.pipeline_options.ocr_options

        def convert_all(
            self,
            input_doc_paths,
            headers=None,
            raises_on_error=False,
            page_range=DEFAULT_PAGE_RANGE,
        ):
            return []

    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter", _FakeDocumentConverter
    )
    result = runner.invoke(
        app, [_SOURCE, "--output", str(tmp_path / "out"), *extra_args]
    )
    return result, captured.get("ocr_options")


def test_ocr_lang_reaches_the_options(monkeypatch, tmp_path: Path) -> None:
    """The CLI constructs the options with `lang=`; it used to assign afterwards,
    which bypassed validation entirely."""
    result, ocr_options = _capture_ocr_options(
        monkeypatch, ["--ocr-engine", "easyocr", "--ocr-lang", "zh-Hant"], tmp_path
    )

    assert result.exit_code == 0, result.output
    assert ocr_options.lang == ["zh-Hant"]


def test_ocr_lang_strips_whitespace(monkeypatch, tmp_path: Path) -> None:
    result, ocr_options = _capture_ocr_options(
        monkeypatch, ["--ocr-engine", "easyocr", "--ocr-lang", "en, de"], tmp_path
    )

    assert result.exit_code == 0, result.output
    assert ocr_options.lang == ["en-Latn", "de-Latn"]


def test_an_empty_ocr_lang_asks_the_engine_to_choose(
    monkeypatch, tmp_path: Path
) -> None:
    """`--ocr-lang ""` is the only way the CLI can say `lang=[]`, which is what
    reaches Tesseract's per-page script detection -- the mode the retired
    `--ocr-lang auto` used to select. Omitting the option is a different
    request: the engine's own default languages."""
    result, ocr_options = _capture_ocr_options(
        monkeypatch, ["--ocr-engine", "easyocr", "--ocr-lang", ""], tmp_path
    )

    assert result.exit_code == 0, result.output
    assert ocr_options.lang == []

    _, defaulted = _capture_ocr_options(
        monkeypatch, ["--ocr-engine", "easyocr"], tmp_path
    )

    assert defaulted.lang == EasyOcrOptions().lang


def test_ocr_lang_defaults_to_the_engine_default(monkeypatch, tmp_path: Path) -> None:
    result, ocr_options = _capture_ocr_options(
        monkeypatch, ["--ocr-engine", "rapidocr"], tmp_path
    )

    assert result.exit_code == 0, result.output
    assert ocr_options.lang == ["zh-Hans"]


@pytest.mark.parametrize(
    ("value", "hint"),
    [
        # `ch` is RapidOCR's token for Chinese, but the default engine is `auto`,
        # which has no engine to prefer that reading over BCP-47's Chamorro.
        ("ch", "zh-Hans"),
        ("auto", "leave the OCR language list empty"),
        ("klingon", "BCP-47"),
    ],
)
def test_retired_or_malformed_ocr_lang_fails_with_a_hint(
    tmp_path: Path, value: str, hint: str
) -> None:
    result = runner.invoke(
        app, [_SOURCE, "--output", str(tmp_path / "out"), "--ocr-lang", value]
    )

    assert result.exit_code != 0
    assert hint in _flat_cli_output(result.output)


@pytest.mark.parametrize(
    ("engine", "value", "expected"),
    [
        ("rapidocr", "native:ch", ["native:ch"]),
        ("rapidocr", "native:chinese_cht", ["native:chinese_cht"]),
        ("tesseract", "native:chi_tra", ["native:chi_tra"]),
        ("easyocr", "native:ch_sim", ["native:ch_sim"]),
        # No engine named on the command line.
        (None, "native:chinese", ["native:chinese"]),
    ],
)
def test_engine_native_ocr_lang_is_accepted(
    monkeypatch,
    tmp_path: Path,
    engine: str | None,
    value: str,
    expected: list[str],
) -> None:
    """The selected engine's own codes work, stored exactly as they were written."""
    args = ["--ocr-lang", value]
    if engine is not None:
        args = ["--ocr-engine", engine, *args]

    result, ocr_options = _capture_ocr_options(monkeypatch, args, tmp_path)

    assert result.exit_code == 0, result.output
    assert ocr_options.lang == expected


def test_another_engines_native_code_is_rejected(tmp_path: Path) -> None:
    """`chi_sim` is Tesseract's; asking RapidOCR for it is a mistake worth naming."""
    result = runner.invoke(
        app,
        [
            _SOURCE,
            "--output",
            str(tmp_path / "out"),
            "--ocr-engine",
            "rapidocr",
            "--ocr-lang",
            "chi_sim",
        ],
    )

    assert result.exit_code != 0
    assert "zh-Hans" in _flat_cli_output(result.output)
