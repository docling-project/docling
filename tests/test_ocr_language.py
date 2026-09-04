# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Docling's OCR language policy: hand it over verbatim, or canonicalize to BCP-47.

A token is a code of the selected engine, which reaches it untouched, unless it
carries the `iso:` prefix, which makes it a BCP-47 tag reduced to a
`(language, script)` pair. There is no third case: the resolver does not know
which engine was selected, so a bare token is never given a reading of its own.
These assert docling decisions -- drop the region, keep the script, refuse `und`,
`zxx` and `mul` -- rather than langcodes behaviour.

Four sections, following one user-supplied string all the way to a stored tag:
the resolver itself, the engine codes it passes through, the `OcrOptions`
validator that calls it, and the `--ocr-lang` CLI flag that feeds that validator.
"""

import warnings
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from docling.cli.main import app
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
from docling.models.stages.ocr.tesseract_utils import language_to_tesseract_code
from docling.utils.ocr_language import (
    OcrLanguage,
    OcrLanguageResolver,
)


def _iso(value: str) -> OcrLanguage:
    """Canonicalize `value` as a BCP-47 request, the way a user writes it."""
    return OcrLanguageResolver.canonicalize_ocr_language(f"iso:{value}")


def _canonical_tags(values: list[str]) -> list[str]:
    """The tags `OcrOptions.lang` would store for `values`."""
    return [
        language.tag()
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
        ("  de  ", "de-Latn"),
    ],
)
def test_canonicalization(value: str, expected: str) -> None:
    assert _iso(value).bcp47() == expected


@pytest.mark.parametrize("value", ["und", "und-Latn", "und-latn", "und-Cyrl"])
def test_undetermined_tags_are_rejected(value: str) -> None:
    """Docling has no "undetermined" language and no script families.

    `Language.get("und-Latn").maximize()` is `en-Latn-US`, so accepting these
    would silently turn "any Latin-script document" into "English". The empty
    list carries the "let the engine decide" meaning instead.
    """
    with pytest.raises(ValueError, match="no script families"):
        _iso(value)


@pytest.mark.parametrize("value", ["klingon", "", "   ", "de-DE-DE", "zz"])
def test_malformed_tags_raise(value: str) -> None:
    with pytest.raises(ValueError):
        _iso(value)


@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_value_is_not_an_engine_code_either(value: str) -> None:
    """Nothing is passed through blank: `lang=[]` is how "let the engine decide"
    is spelled, and `lang=[""]` is a mistake rather than a shorter way to say it."""
    with pytest.raises(ValueError, match="empty"):
        OcrLanguageResolver.canonicalize_ocr_language(value)


@pytest.mark.parametrize("value", ["mul", "MUL", "  mul  "])
def test_multiple_languages_is_not_a_tag_docling_accepts(value: str) -> None:
    """`mul` names "multiple languages", which is a fact about a document rather
    than a recognizer. An engine that ships a multilingual model names it in its
    own vocabulary, so that code is what reaches it."""
    with pytest.raises(ValueError, match="multiple languages"):
        _iso(value)

    assert OcrLanguageResolver.canonicalize_ocr_language("multilingual").native == (
        "multilingual"
    )


@pytest.mark.parametrize("value", ["zxx", "ZXX", "  zxx  "])
def test_no_linguistic_content_is_not_an_ocr_language(value: str) -> None:
    """`zxx` used to disable the engine; skipping OCR is a pipeline switch, not a
    language, and langcodes would otherwise read the tag as `zxx-Latn`."""
    with pytest.raises(ValueError, match="do_ocr=False"):
        _iso(value)


def test_empty_list_means_the_engine_decides() -> None:
    """No tag carries that meaning any more, so the empty list has to."""
    assert OcrLanguageResolver.canonicalize_ocr_languages([]) == []


def test_duplicates_collapse_and_order_is_preserved() -> None:
    """Order is preference order for engines that join languages (Tesseract `+`)."""
    assert _canonical_tags(
        ["iso:fr", "iso:de", "iso:fr-FR", "iso:en-GB", "iso:deu"]
    ) == [
        "iso:fr-Latn",
        "iso:de-Latn",
        "iso:en-Latn",
    ]


def test_canonicalization_is_idempotent() -> None:
    once = _canonical_tags(["iso:deu", "iso:zh-TW", "iso:sr-Latn", "iso:pa-PK"])

    assert _canonical_tags(once) == once


def test_has_default_script_separates_the_script_variants() -> None:
    """Engines key on this to decide whether the primary subtag is enough."""
    assert _iso("de").has_default_script()
    assert _iso("sr").has_default_script()
    assert not _iso("sr-Latn").has_default_script()
    assert not _iso("az-Cyrl").has_default_script()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # The script CLDR infers is not something a user has to type.
        ("de", "de"),
        ("de-Latn", "de"),
        ("de-DE", "de"),
        ("zh-CN", "zh"),
        ("sr", "sr"),
        # A non-default script names a different recognizer, so it survives.
        ("de-Latf", "de-Latf"),
        ("zh-Hant", "zh-Hant"),
        ("sr-Latn", "sr-Latn"),
        ("az-Cyrl", "az-Cyrl"),
        ("anp-Deva", "anp-Deva"),
    ],
)
def test_short_tag_drops_only_the_inferred_script(value: str, expected: str) -> None:
    """What an engine advertises: the shortest spelling that still round-trips.

    `supported_ocr_languages()` fills the "Supported:" line of
    `OcrLanguageNotSupportedError`, where the entries are rendered behind the
    prefix, so every one of them is something a user pastes back as `iso:<tag>`
    -- and asking for it again has to land on the same recognizer.
    """
    language = _iso(value)

    assert language.short_tag() == expected
    assert _iso(language.short_tag()) == language


def test_an_engine_code_is_already_as_short_as_it_gets() -> None:
    """The other half of the same error message: no prefix is added to those."""
    language = OcrLanguageResolver.canonicalize_ocr_language("script/Cyrillic")

    assert language.short_tag() == "script/Cyrillic"
    assert (
        OcrLanguageResolver.canonicalize_ocr_language(language.short_tag()) == language
    )


def test_ocr_language_is_hashable() -> None:
    """Engines key dicts and caches on the canonical pair."""
    assert {OcrLanguage(bcp47_language="de", bcp47_script="Latn")} == {_iso("de-DE")}


def test_match_against_a_region_bearing_vocabulary() -> None:
    """Apple Vision's own vocabulary is BCP-47 with regions."""
    supported = ["en-US", "fr-FR", "de-DE", "pt-BR", "zh-Hans"]

    assert OcrLanguageResolver.match_ocr_language(_iso("de"), supported) == "de-DE"
    assert OcrLanguageResolver.match_ocr_language(_iso("pt"), supported) == "pt-BR"
    assert OcrLanguageResolver.match_ocr_language(_iso("zh-CN"), supported) == "zh-Hans"
    assert OcrLanguageResolver.match_ocr_language(_iso("th"), supported) is None


# --- the engine's own codes -------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "klingon",  # structurally a primary subtag, but unregistered
        "osd",  # a tessdata file that is not a language
        "script/HanS_vert",  # not a tag at all
    ],
)
def test_canonicalize_bcp47_can_answer_none_instead_of_raising(value: str) -> None:
    """The failure guard, for the callers that build a vocabulary rather than
    serve a user.

    `installed_tesseract_languages` and `_ppocr_supported_languages` walk an
    engine's own code list and ask which entries are tags; a rejection never
    reaches anyone there, so they opt out of it. Every user-facing path keeps the
    default `ValueError`, which carries one.
    """
    assert OcrLanguageResolver.canonicalize_bcp47(value, raise_exception=False) is None
    with pytest.raises(ValueError):
        OcrLanguageResolver.canonicalize_bcp47(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("iso:de", "iso:de-Latn"),
        # The prefix is matched on the lowercased token, and what follows it is a
        # tag, which is case-insensitive too.
        ("ISO:de-DE", "iso:de-Latn"),
        ("Iso:ZH-hant", "iso:zh-Hant"),
        # Whitespace is stripped on both sides of the prefix.
        ("  iso:  zh-Hant  ", "iso:zh-Hant"),
    ],
)
def test_the_iso_prefix_is_case_insensitive(value: str, expected: str) -> None:
    assert OcrLanguageResolver.canonicalize_ocr_language(value).tag() == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("ch", "ch"),
        # An engine code is not a tag, so its casing is never touched: a tessdata
        # script file has to reach tesseract spelled as it is installed.
        ("script/Cyrillic", "script/Cyrillic"),
        ("jpn_vert", "jpn_vert"),
        ("  chi_tra  ", "chi_tra"),
    ],
)
def test_an_engine_code_reaches_the_engine_as_written(
    value: str, expected: str
) -> None:
    assert OcrLanguageResolver.canonicalize_ocr_language(value).tag() == expected


def test_both_forms_round_trip_through_tag() -> None:
    """`tag` is the storage form, and re-reading it must not move a request:
    an engine code stays bare, a tag keeps its prefix."""
    once = _canonical_tags(["arabic", "script/Cyrillic", "iso:de", "iso:zh-TW"])

    assert once == ["arabic", "script/Cyrillic", "iso:de-Latn", "iso:zh-Hant"]
    assert _canonical_tags(once) == once


def test_an_engine_code_names_no_language() -> None:
    """`bcp47` is the pair an engine's table is keyed on and `tag` is the storage
    form; only for an engine code do they differ, and that split is what keeps
    the code out of langcodes."""
    language = OcrLanguageResolver.canonicalize_ocr_language("cyrillic")

    assert language.is_passthrough()
    assert language.native == "cyrillic"
    assert language.tag() == "cyrillic"
    assert language.bcp47() == ""
    assert language.bcp47_language is None and language.bcp47_script is None
    # It names a script, so there is no language whose default script it could be.
    assert not language.has_default_script()


def test_a_tag_carries_the_prefix() -> None:
    language = _iso("de-DE")

    assert not language.is_passthrough()
    assert language.native is None
    assert language.bcp47() == "de-Latn"
    assert language.tag() == "iso:de-Latn"


@pytest.mark.parametrize(
    ("options_cls", "native", "expected"),
    [
        (RapidOcrOptions, ["ch"], ["ch"]),
        (RapidOcrOptions, ["chinese_cht"], ["chinese_cht"]),
        (TesseractOcrOptions, ["chi_tra"], ["chi_tra"]),
        (TesseractCliOcrOptions, ["script/Cyrillic"], ["script/Cyrillic"]),
        (EasyOcrOptions, ["ch_sim", "ang"], ["ch_sim", "ang"]),
        (NemotronOcrOptions, ["multilingual"], ["multilingual"]),
        (OcrAutoOptions, ["chinese"], ["chinese"]),
        # Vision's codes look like tags but are not read as any: they are what
        # the running macOS reports, region and all.
        (OcrMacOptions, ["en-US"], ["en-US"]),
    ],
)
def test_options_accept_their_own_engines_codes(
    options_cls: type, native: list[str], expected: list[str]
) -> None:
    """An engine code is stored as written, so revalidating `lang` cannot move it."""
    assert options_cls(lang=native).lang == expected


def test_tesseract_fraktur_keeps_its_own_traineddata() -> None:
    """`de-Latf` used to flatten to `deu` through to_alpha3(), losing Fraktur."""
    assert language_to_tesseract_code(_iso("de-Latf")) == "deu_latf"


# --- the options validator --------------------------------------------------
#
# `OcrOptions` declares one `@field_validator("lang")`, but every engine subclass
# *redefines* `lang` with its own default and its own `ConfigDict`. The design
# assumes pydantic collects validators by field name across the MRO and merges
# `model_config` down it. Both are asserted here, because a silent regression
# would let a malformed tag through unvalidated.

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
    options = _build(cls, lang=["iso:deu", "iso:en-US", "iso:zh-TW"])

    assert options.lang == ["iso:de-Latn", "iso:en-Latn", "iso:zh-Hant"]


@pytest.mark.parametrize("cls", _OPTION_CLASSES)
def test_defaults_are_already_canonical(cls) -> None:
    """`validate_default=True` makes this an assertion, not a rewrite."""
    default = _build(cls).lang

    assert _canonical_tags(default) == default


@pytest.mark.parametrize("cls", _OPTION_CLASSES)
def test_unregistered_tags_are_rejected(cls) -> None:
    """`auto` is structurally a valid primary subtag; only IANA rejects it.

    Bare, it is whatever the engine makes of it; behind the prefix it is a claim
    about BCP-47 that docling can check, and does.
    """
    with pytest.raises(ValidationError, match="not registered with IANA"):
        _build(cls, lang=["iso:auto"])


@pytest.mark.parametrize("cls", _OPTION_CLASSES)
def test_empty_lang_is_accepted(cls) -> None:
    """An empty list is how "let the engine decide" is spelled."""
    assert _build(cls, lang=[]).lang == []


def test_assignment_is_validated() -> None:
    """The SDK mutation path documented in the FAQ goes through the validator.

    Proves `validate_assignment` on the base survives the subclass `ConfigDict`.
    """
    options = EasyOcrOptions()

    options.lang = ["iso:fra", "iso:de-DE"]
    assert options.lang == ["iso:fr-Latn", "iso:de-Latn"]

    with pytest.raises(ValidationError, match="Invalid OCR language"):
        options.lang = ["iso:chinese"]


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
    options = TesseractCliOcrOptions(lang=["iso:fra", "deu"])

    restored = TesseractCliOcrOptions.model_validate(options.model_dump())

    assert restored.lang == options.lang == ["iso:fr-Latn", "deu"]


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
        monkeypatch, ["--ocr-engine", "easyocr", "--ocr-lang", "iso:zh-Hant"], tmp_path
    )

    assert result.exit_code == 0, result.output
    assert ocr_options.lang == ["iso:zh-Hant"]


def test_ocr_lang_strips_whitespace(monkeypatch, tmp_path: Path) -> None:
    result, ocr_options = _capture_ocr_options(
        monkeypatch,
        ["--ocr-engine", "easyocr", "--ocr-lang", "iso:en, iso:de"],
        tmp_path,
    )

    assert result.exit_code == 0, result.output
    assert ocr_options.lang == ["iso:en-Latn", "iso:de-Latn"]


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
    assert ocr_options.lang == ["ch"]


@pytest.mark.parametrize(
    ("value", "hint"), [("iso:klingon", "BCP-47"), ("iso:auto", "IANA")]
)
def test_malformed_ocr_lang_fails_with_a_hint(
    tmp_path: Path, value: str, hint: str
) -> None:
    """The rejection has to survive typer's error panel and name the vocabulary."""
    result = runner.invoke(
        app, [_SOURCE, "--output", str(tmp_path / "out"), "--ocr-lang", value]
    )

    assert result.exit_code != 0
    assert hint in _flat_cli_output(result.output)


@pytest.mark.parametrize(
    ("engine", "value", "expected"),
    [
        ("rapidocr", "ch", ["ch"]),
        ("rapidocr", "chinese_cht", ["chinese_cht"]),
        ("tesseract", "chi_tra", ["chi_tra"]),
        ("easyocr", "ch_sim", ["ch_sim"]),
        # No engine named on the command line.
        (None, "chinese", ["chinese"]),
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
