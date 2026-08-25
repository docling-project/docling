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

The load-bearing test is `test_native_alias_never_shadows_a_reachable_language`:
it is the whole safety argument for letting a native reading win over a valid
BCP-47 tag, and it is what stops a future alias from quietly hiding a language
its own engine can serve.
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
    KserveV2OcrOptions,
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
from docling.models.stages.ocr.easyocr_model import (
    _EASYOCR_LANGUAGE_CODES,
    EasyOcrModel,
)
from docling.models.stages.ocr.ppocr_languages import (
    PPOCRV4_LANGS,
    PPOCRV5_LANGS,
    PPOCRV6_LANGS,
    ppocr_supported_tags,
)
from docling.models.stages.ocr.tesseract_utils import (
    _TESSERACT_LANGUAGE_NAMES,
    map_tesseract_language,
)
from docling.utils.ocr_language import (
    OcrLanguage,
    OcrLanguageResolver,
)

#: Every engine kind that has a native vocabulary. Kinds sharing one vocabulary
#: (both Tesseract bindings, RapidOCR and the KServe client) are all listed:
#: each must reach the same table.
NATIVE_KINDS = sorted(OcrLanguageResolver._NATIVE_VOCABULARIES)

# The tokens that are a legitimate BCP-47 spelling of one language and an
# engine's own name for a different one. These are the only genuine ambiguities:
# `chi_tra`, `ch_sim` and `ch_tra` look similar but are not tags at all -- see
# `_plain_bcp47_tag` -- and everything else agrees with BCP-47 or cannot be
# parsed as a tag.
CLASHING_TOKENS = frozenset({"ch", "ka", "ang", "mah", "frk", "tab"})


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
    assert OcrLanguageResolver.parse_ocr_language(value).tag == expected


@pytest.mark.parametrize("value", ["und", "und-Latn", "und-latn", "und-Cyrl"])
def test_undetermined_tags_are_rejected(value: str) -> None:
    """Docling has no "undetermined" language and no script families.

    `Language.get("und-Latn").maximize()` is `en-Latn-US`, so accepting these
    would silently turn "any Latin-script document" into "English". The empty
    list carries the "let the engine decide" meaning instead.
    """
    with pytest.raises(ValueError, match="no script families"):
        OcrLanguageResolver.parse_ocr_language(value)


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
        OcrLanguageResolver.parse_ocr_language(value)

    assert f"'{hint}'" in str(excinfo.value)


@pytest.mark.parametrize("value", ["klingon", "", "   ", "de-DE-DE", "zz"])
def test_malformed_tags_raise(value: str) -> None:
    with pytest.raises(ValueError):
        OcrLanguageResolver.parse_ocr_language(value)


def test_chamorro_is_not_chinese() -> None:
    """RapidOCR's `ch` means Chinese, but BCP-47 `ch` is Chamorro.

    Accepting it would resolve to the wrong language and surface as a confusing
    coverage error much later, so it is rejected up front.
    """
    with pytest.raises(ValueError, match="zh-Hans"):
        OcrLanguageResolver.parse_ocr_language("ch")


def test_reserved_tag_must_stand_alone() -> None:
    for combination in (["mul", "en"], ["en", "mul"]):
        with pytest.raises(ValueError, match="on its own"):
            OcrLanguageResolver.parse_ocr_languages(combination)


@pytest.mark.parametrize("value", ["zxx", "ZXX", "  zxx  "])
def test_no_linguistic_content_is_not_an_ocr_language(value: str) -> None:
    """`zxx` used to disable the engine; skipping OCR is a pipeline switch, not a
    language, and langcodes would otherwise read the tag as `zxx-Latn`."""
    with pytest.raises(ValueError, match="do_ocr=False"):
        OcrLanguageResolver.parse_ocr_language(value)


def test_empty_list_means_the_engine_decides() -> None:
    """No tag carries that meaning any more, so the empty list has to."""
    assert OcrLanguageResolver.parse_ocr_languages([]) == ()


@pytest.mark.parametrize(
    "value", ["auto", "osd", "latin", "cyrillic", "arabic", "devanagari", "bengali"]
)
def test_engine_decides_and_script_tokens_point_at_the_empty_list(value: str) -> None:
    """These named a script or an auto mode, neither of which is a language.

    PP-OCR defines four of them as real recognizers, so they resolve as
    passthroughs for that engine and only reach this path for everyone else.
    """
    with pytest.raises(ValueError, match="leave the OCR language list empty"):
        OcrLanguageResolver.parse_ocr_language(value)


def test_duplicates_collapse_and_order_is_preserved() -> None:
    """Order is preference order for engines that join languages (Tesseract `+`)."""
    assert OcrLanguageResolver.canonicalize_ocr_language_tags(
        ["fr", "de", "fr-FR", "en-GB", "deu"]
    ) == [
        "fr-Latn",
        "de-Latn",
        "en-Latn",
    ]


def test_canonicalization_is_idempotent() -> None:
    once = OcrLanguageResolver.canonicalize_ocr_language_tags(
        ["deu", "zh-TW", "sr-Latn", "pa-PK"]
    )

    assert OcrLanguageResolver.canonicalize_ocr_language_tags(once) == once


def test_has_default_script_separates_the_script_variants() -> None:
    """Engines key on this to decide whether the primary subtag is enough."""
    assert OcrLanguageResolver.parse_ocr_language("de").has_default_script
    assert OcrLanguageResolver.parse_ocr_language("sr").has_default_script
    assert not OcrLanguageResolver.parse_ocr_language("sr-Latn").has_default_script
    assert not OcrLanguageResolver.parse_ocr_language("az-Cyrl").has_default_script


def test_reserved_flags() -> None:
    assert OcrLanguageResolver.parse_ocr_language("mul").is_multilingual
    assert OcrLanguageResolver.parse_ocr_language("mul").is_reserved
    assert not OcrLanguageResolver.parse_ocr_language("en").is_reserved


def test_ocr_language_is_hashable() -> None:
    """Engines key dicts and caches on the canonical pair."""
    assert {OcrLanguage(language="de", script="Latn")} == {
        OcrLanguageResolver.parse_ocr_language("de-DE")
    }


def test_match_against_a_region_bearing_vocabulary() -> None:
    """Apple Vision's own vocabulary is BCP-47 with regions."""
    supported = ["en-US", "fr-FR", "de-DE", "pt-BR", "zh-Hans"]

    assert (
        OcrLanguageResolver.match_ocr_language(
            OcrLanguageResolver.parse_ocr_language("de"), supported
        )
        == "de-DE"
    )
    assert (
        OcrLanguageResolver.match_ocr_language(
            OcrLanguageResolver.parse_ocr_language("pt"), supported
        )
        == "pt-BR"
    )
    assert (
        OcrLanguageResolver.match_ocr_language(
            OcrLanguageResolver.parse_ocr_language("zh-CN"), supported
        )
        == "zh-Hans"
    )
    assert (
        OcrLanguageResolver.match_ocr_language(
            OcrLanguageResolver.parse_ocr_language("th"), supported
        )
        is None
    )


# --- engine-native vocabularies --------------------------------------------


def _static_supported_tags(kind: str) -> set[str]:
    """Canonical tags the engine behind `kind` can serve, from static data only.

    Deliberately not the runtime `supported_ocr_languages()` of a live model:
    the guard has to hold on any machine, whatever happens to be installed.
    """
    vocabulary = OcrLanguageResolver._NATIVE_VOCABULARIES[kind]
    if vocabulary is OcrLanguageResolver._PPOCR:
        return set(
            ppocr_supported_tags(
                PPOCRV4_LANGS | PPOCRV5_LANGS | PPOCRV6_LANGS, RapidOcrOptions.kind
            )
        )
    if vocabulary is OcrLanguageResolver._EASYOCR:
        from docling.models.stages.ocr.easyocr_model import (
            _easyocr_code_to_tag,
            _easyocr_language_models,
        )

        return {
            tag
            for code in _easyocr_language_models()
            if (tag := _easyocr_code_to_tag(code))
        }
    if vocabulary is OcrLanguageResolver._TESSERACT:
        # Every tessdata file name, rendered back as the tag that reaches it.
        tags = set(_TESSERACT_LANGUAGE_NAMES)
        # A script file is named back as itself: that is what selects it.
        assert vocabulary.script_files is not None
        tags.update(f"script/{name}" for name in vocabulary.script_files)
        return tags
    if vocabulary is OcrLanguageResolver._NEMOTRON:
        return {"en-Latn", "mul", "zh-Hans", "zh-Hant", "ja-Jpan", "ko-Kore", "ru-Cyrl"}
    raise AssertionError(f"no static supported set for {kind!r}")


@pytest.mark.parametrize("kind", NATIVE_KINDS)
def test_native_alias_never_shadows_a_reachable_language(kind: str) -> None:
    """A native alias may only shadow a BCP-47 reading its own engine cannot serve.

    This is what makes "the native reading wins" lossless rather than a guess.
    If it ever fails, the engine has gained a model for the language its own
    token was hiding, and that token has become a genuine ambiguity.
    """
    supported = _static_supported_tags(kind)
    languages = OcrLanguageResolver._NATIVE_VOCABULARIES[kind].languages
    for token, native_meaning in languages.items():
        bcp47 = OcrLanguageResolver._plain_bcp47_tag(token)
        if bcp47 is None or bcp47 == native_meaning:
            continue  # unparseable, or both readings agree: nothing is shadowed
        assert bcp47 not in supported, (
            f"{kind} alias {token!r} means {native_meaning}, but {token!r} is "
            f"also valid BCP-47 for {bcp47}, which this engine CAN serve. Honouring "
            f"the native reading would make {bcp47} unreachable as {token!r}."
        )


def test_canonical_tags_can_never_be_mistaken_for_native_tokens() -> None:
    """Every canonical tag carries a script subtag; no native token has a hyphen.

    This is what makes `OcrAutoModel` safe: it canonicalizes with the engine-less
    vocabulary and forwards the resulting tags into each delegate's own
    engine-aware validator, which must not re-read them as native tokens.
    """
    for kind, vocabulary in OcrLanguageResolver._NATIVE_VOCABULARIES.items():
        for token, canonical in vocabulary.languages.items():
            assert "-" not in token, f"{kind} token {token!r} looks like a tag"
            assert "-" in canonical or canonical == "mul"


@pytest.mark.parametrize(
    ("token", "kind", "expected"),
    [
        # PP-OCR: `ch` is Chinese, not Chamorro; `ka` is Kannada, not Georgian.
        ("ch", RapidOcrOptions.kind, "zh-Hans"),
        ("ka", RapidOcrOptions.kind, "kn-Knda"),
        ("chinese_cht", RapidOcrOptions.kind, "zh-Hant"),
        ("japan", RapidOcrOptions.kind, "ja-Jpan"),
        ("eslav", RapidOcrOptions.kind, "ru-Cyrl"),
        # A script recognizer has no language to canonicalize to, so it is
        # carried through as the token the user wrote.
        ("latin", RapidOcrOptions.kind, "latin"),
        # Tesseract: `chi_tra` parses to zh-Hans as a tag, which is the wrong script.
        ("chi_sim", TesseractOcrOptions.kind, "zh-Hans"),
        ("chi_tra", TesseractOcrOptions.kind, "zh-Hant"),
        ("frk", TesseractOcrOptions.kind, "de-Latf"),
        ("script/Cyrillic", TesseractOcrOptions.kind, "script/Cyrillic"),
        ("script/Fraktur", TesseractCliOcrOptions.kind, "script/Fraktur"),
        # Not a tessdata token, so BCP-47 wins: Georgian.
        ("ka", TesseractOcrOptions.kind, "ka-Geor"),
        # EasyOCR.
        ("ch_sim", EasyOcrOptions.kind, "zh-Hans"),
        ("ch_tra", EasyOcrOptions.kind, "zh-Hant"),
        ("ang", EasyOcrOptions.kind, "anp-Deva"),
        ("mah", EasyOcrOptions.kind, "mag-Deva"),
        ("tjk", EasyOcrOptions.kind, "tg-Cyrl"),
        ("english", NemotronOcrOptions.kind, "en-Latn"),
        ("multilingual", NemotronOcrOptions.kind, "mul"),
    ],
)
def test_native_tokens_decode(token: str, kind: str, expected: str) -> None:
    assert OcrLanguageResolver.parse_ocr_language(token, kind).tag == expected


@pytest.mark.parametrize(
    ("token", "kind", "expected"),
    [
        ("ch-Latn", RapidOcrOptions.kind, "ch-Latn"),  # Chamorro
        ("ka-Geor", RapidOcrOptions.kind, "ka-Geor"),  # Georgian
        ("ang-Latn", EasyOcrOptions.kind, "ang-Latn"),  # Old English
        ("frk-Latn", TesseractOcrOptions.kind, "frk-Latn"),  # Frankish
    ],
)
def test_qualified_tag_escapes_the_native_table(
    token: str, kind: str, expected: str
) -> None:
    """Writing the script out reaches the BCP-47 meaning of a shadowed token.

    The tables are keyed on the bare token, so this needs no special casing --
    but it is the documented way out, so it is pinned.
    """
    assert OcrLanguageResolver.parse_ocr_language(token, kind).tag == expected


@pytest.mark.parametrize(
    ("token", "kind"),
    [
        ("chi_sim", RapidOcrOptions.kind),  # a Tesseract token
        # A PP-OCR token; BCP-47 reads Chamorro.
        ("ch", TesseractOcrOptions.kind),
        ("tjk", RapidOcrOptions.kind),  # an EasyOCR token
    ],
)
def test_another_engines_token_is_rejected(token: str, kind: str) -> None:
    with pytest.raises(ValueError, match="BCP-47"):
        OcrLanguageResolver.parse_ocr_language(token, kind)


@pytest.mark.parametrize(
    "token", ["jpn_vert", "chi_tra_vert", "script/HanS_vert", "ita_old", "equ"]
)
def test_tokens_no_language_tag_can_express_are_refused(token: str) -> None:
    """Rather than silently resolving to a different recognizer.

    `ita_old` in particular used to canonicalize to plain `it-Latn`, quietly
    selecting the modern Italian model for a historical-orthography request.
    """
    with pytest.raises(ValueError, match="cannot address by language tag"):
        OcrLanguageResolver.parse_ocr_language(token, TesseractOcrOptions.kind)


def test_unrepresentable_tokens_are_scoped_to_the_engine_that_owns_them() -> None:
    """`equ` asked of RapidOCR is better served by "not a valid tag"."""
    with pytest.raises(ValueError, match="BCP-47"):
        OcrLanguageResolver.parse_ocr_language("equ", RapidOcrOptions.kind)


def test_engine_less_vocabulary_drops_every_clashing_token() -> None:
    """`OcrAutoOptions` has no engine, so it cannot prefer a native reading."""
    assert CLASHING_TOKENS.isdisjoint(OcrLanguageResolver.UNAMBIGUOUS_NATIVE_LANGUAGES)
    # What survives is the vocabulary no engine disagrees about, including the
    # tokens whose tag-shaped appearance is only a parse artifact. Passthroughs
    # such as `latin` are deliberately absent: a script recognizer is only
    # meaningful to the engine that defines it.
    assert "latin" not in OcrLanguageResolver.UNAMBIGUOUS_NATIVE_LANGUAGES
    assert {
        "chinese",
        "japan",
        "korean",
        "multilingual",
        "chi_sim",
        "chi_tra",
        "ch_sim",
    } <= set(OcrLanguageResolver.UNAMBIGUOUS_NATIVE_LANGUAGES)


def test_engine_less_vocabulary_is_derived_not_hand_written() -> None:
    """Every surviving token is either unparseable or agrees with its tag."""
    for token, canonical in OcrLanguageResolver.UNAMBIGUOUS_NATIVE_LANGUAGES.items():
        assert OcrLanguageResolver._plain_bcp47_tag(token) in (None, canonical)


@pytest.mark.parametrize(
    ("options_cls", "native", "expected"),
    [
        (RapidOcrOptions, ["ch"], ["zh-Hans"]),
        (RapidOcrOptions, ["chinese_cht"], ["zh-Hant"]),
        (TesseractOcrOptions, ["chi_tra"], ["zh-Hant"]),
        (TesseractCliOcrOptions, ["script/Cyrillic"], ["script/Cyrillic"]),
        (EasyOcrOptions, ["ch_sim", "ang"], ["zh-Hans", "anp-Deva"]),
        (NemotronOcrOptions, ["multilingual"], ["mul"]),
        (OcrAutoOptions, ["chinese"], ["zh-Hans"]),
        # ocrmac has no native vocabulary: Vision's codes already are BCP-47.
        (OcrMacOptions, ["en-US"], ["en-Latn"]),
    ],
)
def test_options_accept_their_own_engines_codes(
    options_cls: type, native: list[str], expected: list[str]
) -> None:
    """And store them canonicalized, so nothing downstream sees native codes."""
    assert options_cls(lang=native).lang == expected


def test_options_reject_a_clashing_token_when_no_engine_is_chosen() -> None:
    with pytest.raises(ValidationError, match="zh-Hans"):
        OcrAutoOptions(lang=["ch"])


@pytest.mark.parametrize("kind", NATIVE_KINDS)
def test_decoding_is_idempotent(kind: str) -> None:
    """A canonicalized tag fed back in must not move again."""
    for token in OcrLanguageResolver._NATIVE_VOCABULARIES[kind].languages:
        once = OcrLanguageResolver.parse_ocr_language(token, kind).tag
        assert OcrLanguageResolver.parse_ocr_language(once, kind).tag == once


def test_easyocr_native_codes_round_trip() -> None:
    """Decoding a code and mapping it back must return the same code."""
    from docling.models.stages.ocr.easyocr_model import _easyocr_code

    for token, canonical in OcrLanguageResolver._EASYOCR.languages.items():
        assert (
            _easyocr_code(OcrLanguageResolver.parse_ocr_language(canonical)) == token
        ), token
    # And the forward table is what makes that possible for the three the
    # likely-subtags rules would otherwise send elsewhere.
    for tag in ("anp-Deva", "mag-Deva", "tab-Cyrl"):
        assert tag in _EASYOCR_LANGUAGE_CODES


def test_tesseract_fraktur_keeps_its_own_traineddata() -> None:
    """`de-Latf` used to flatten to `deu` through to_alpha3(), losing Fraktur."""
    assert (
        map_tesseract_language(
            OcrLanguageResolver.parse_ocr_language("de-Latf"), "script/"
        )
        == "deu_latf"
    )
    assert (
        map_tesseract_language(
            OcrLanguageResolver.parse_ocr_language("frk", TesseractOcrOptions.kind),
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
    # KServe is the one engine with a required connection field.
    if cls is KserveV2OcrOptions:
        kwargs.setdefault("url", "http://localhost:8000")
    return cls(**kwargs)


@pytest.mark.parametrize("cls", [*_OPTION_CLASSES, KserveV2OcrOptions])
def test_base_validator_fires_on_every_subclass(cls) -> None:
    options = _build(cls, lang=["deu", "en-US", "zh-TW"])

    assert options.lang == ["de-Latn", "en-Latn", "zh-Hant"]


@pytest.mark.parametrize("cls", [*_OPTION_CLASSES, KserveV2OcrOptions])
def test_defaults_are_already_canonical(cls) -> None:
    """`validate_default=True` makes this an assertion, not a rewrite."""
    default = _build(cls).lang

    assert OcrLanguageResolver.canonicalize_ocr_language_tags(default) == default


@pytest.mark.parametrize("cls", [*_OPTION_CLASSES, KserveV2OcrOptions])
def test_retired_tokens_are_rejected(cls) -> None:
    with pytest.raises(ValidationError, match="no script families"):
        _build(cls, lang=["auto"])


@pytest.mark.parametrize("cls", [*_OPTION_CLASSES, KserveV2OcrOptions])
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


@pytest.mark.parametrize("cls", [*_OPTION_CLASSES, KserveV2OcrOptions])
def test_every_engine_kind_selects_its_native_vocabulary(cls) -> None:
    """The resolver keys native vocabularies by `kind`, so a renamed `kind`
    would silently stop an engine's own codes from being accepted."""
    vocabulary = OcrLanguageResolver._vocabulary_for(cls.kind)

    if cls is OcrMacOptions:
        # Apple Vision speaks BCP-47 already; it has no native vocabulary.
        assert vocabulary is None
    else:
        assert vocabulary is not None, (
            f"{cls.__name__}.kind is {cls.kind!r}, which no vocabulary is keyed on"
        )
        assert vocabulary.languages


# --- the CLI ----------------------------------------------------------------
#
# `--ocr-lang`: construction, canonicalization, and error reporting.

runner = CliRunner()

_SOURCE = "./tests/data/pdf/sources/2305.03393v1-pg9.pdf"


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
    assert hint in " ".join(result.output.replace("│", "").split())


@pytest.mark.parametrize(
    ("engine", "value", "expected"),
    [
        ("rapidocr", "ch", ["zh-Hans"]),
        ("rapidocr", "chinese_cht", ["zh-Hant"]),
        ("tesseract", "chi_tra", ["zh-Hant"]),
        ("easyocr", "ch_sim", ["zh-Hans"]),
        # No engine named: only the codes every engine agrees on.
        (None, "chinese", ["zh-Hans"]),
    ],
)
def test_engine_native_ocr_lang_is_accepted(
    monkeypatch,
    tmp_path: Path,
    engine: str | None,
    value: str,
    expected: list[str],
) -> None:
    """The selected engine's own codes work and are stored canonicalized."""
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
    assert "zh-Hans" in " ".join(result.output.replace("│", "").split())
