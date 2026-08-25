# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Canonicalization of OCR language requests to BCP-47 (RFC 5646).

Docling exposes exactly one language vocabulary to users -- BCP-47 tags -- and
reduces every request to a `(language, script)` pair. Per-engine adapters
translate that pair into the engine's own notation (see
`docling.models.base_ocr_model.BaseOcrModel.map_ocr_language`).

Region is discarded once it has inferred the script: `zh-CN` and `zh-Hans` are
the same recognizer, and `de-DE` vs `de-AT` is a distinction no OCR engine
docling supports can act on.

`OcrLanguage` is the result type; `OcrLanguageResolver` owns the parsing itself
and the tables it consults.

Users, however, arrive with the vocabulary of the engine they were already using
-- `ch` for RapidOCR, `chi_sim` for Tesseract, `ch_sim` for EasyOCR -- so each
options class also accepts its *own* engine's tokens, keyed by `OcrOptions.kind`
in `OcrLanguageResolver._NATIVE_VOCABULARIES`. Only tokens that plain BCP-47
parsing gets wrong are listed there: the great majority of every engine's
vocabulary is ISO 639 already (`deu`, `fra`, `ru`, `ta`) and needs no entry.

Nine of those tokens read as a *different language* under BCP-47 than their
engine means. For all nine the BCP-47 reading names a language that engine has
no model for -- no OCR engine docling supports ships a Chamorro, Georgian,
Frankish, Marshallese or Old English recognizer -- so letting the native reading
win takes nothing away, and `test_native_alias_never_shadows_a_reachable_language`
enforces exactly that. A user who genuinely wants the BCP-47 reading writes the
fully-qualified tag: the tables are keyed on the bare token, so `ch-Latn`
(Chamorro), `ka-Geor` (Georgian) and `ang-Latn` (Old English) bypass them.
"""

import logging
from collections.abc import Mapping, Sequence
from functools import lru_cache
from types import MappingProxyType
from typing import ClassVar, NamedTuple

import langcodes
from pydantic import BaseModel, ConfigDict

_log = logging.getLogger(__name__)

#: Multiple languages: the engine's broadest multilingual model.
MULTIPLE = "mul"

_NO_TOKENS: Mapping[str, str] = MappingProxyType({})


class _NativeVocabulary(NamedTuple):
    """One engine family's own language codes, accepted alongside BCP-47.

    Attributes:
        languages: Native token -> canonical tag, for the tokens plain BCP-47
            parsing would get wrong.
        passthrough: Tokens naming a real model but no language -- PP-OCR's
            script recognizers. They have no `(language, script)` form, so they
            reach the engine verbatim rather than canonicalized.
        script_files: tessdata `script/<Name>` file name -> ISO 15924 code, for
            the engines that have such files. `None` for the engines that do not.
        unrepresentable: Tokens docling refuses rather than mis-resolving,
            mapped to the reason. Each names a model the canonical
            `(language, script)` form cannot distinguish, so canonicalizing
            would silently select a *different* recognizer.
    """

    languages: Mapping[str, str]
    passthrough: frozenset[str] = frozenset()
    script_files: Mapping[str, str] | None = None
    unrepresentable: Mapping[str, str] = _NO_TOKENS


class OcrLanguageSupport(BaseModel):
    """Static, engine-declared language capabilities.

    Attributes:
        multiple_languages: Whether the engine can run several languages at
            once. `False` marks a single-language engine, whose extra tags are
            dropped with a warning.
    """

    model_config = ConfigDict(frozen=True)

    multiple_languages: bool = False


class OcrLanguage(BaseModel):
    """One canonicalized OCR language request: a BCP-47 (language, script) pair.

    Attributes:
        language: Primary subtag, lowercase. May be the reserved subtag `mul`.
        script: ISO 15924 script code in title case. `None` only for the bare
            reserved tags.
        native: An engine's own token, kept verbatim, for the models no
            `(language, script)` pair can name -- PP-OCR's script recognizers
            (`arabic`, `cyrillic`) and Tesseract's `script/<Name>` files. Set
            only for a passthrough, where it excludes `language` and `script`
            and is what `tag` returns; `None` for every ordinary BCP-47 request.
    """

    model_config = ConfigDict(frozen=True)

    language: str | None = None
    script: str | None = None
    native: str | None = None

    @property
    def tag(self) -> str:
        """How this request is written back into `OcrOptions.lang`.

        A canonical BCP-47 tag (`de-Latn`, `mul`), or, for a passthrough, the
        engine's own token verbatim. Returning the token unchanged is what keeps
        `lang` idempotent: revalidating `["arabic"]` must not move it.
        """
        if self.native is not None:
            return self.native
        return f"{self.language}-{self.script}" if self.script else self.language or ""

    @property
    def is_passthrough(self) -> bool:
        """An engine token that no `(language, script)` pair can express.

        PP-OCR's script recognizers (`arabic`, `cyrillic`) and Tesseract's
        `script/<Name>` files: real models, named after a script rather than a
        language, and handed to the engine untouched.
        """
        return self.native is not None

    @property
    def is_reserved(self) -> bool:
        """One of the bare reserved tags, which must be requested alone."""
        return self.language in OcrLanguageResolver._RESERVED and self.script is None

    @property
    def is_multilingual(self) -> bool:
        return self.language == MULTIPLE

    @property
    def has_default_script(self) -> bool:
        """Whether `script` is the script CLDR considers likely for `language`.

        `de-Latn` and `en-Latn` do; `az-Cyrl` and `uz-Cyrl` do not. Engines use
        this to decide whether the primary subtag alone still identifies the
        right recognizer.
        """
        if self.language is None or self.language in OcrLanguageResolver._RESERVED:
            return False
        return self.script == OcrLanguageResolver._default_script_for_language(
            self.language
        )

    def __str__(self) -> str:
        return self.tag


class OcrLanguageResolver:
    """Parses user-supplied OCR language tokens into `OcrLanguage`.

    A namespace rather than an object: every entry point is a `@staticmethod`,
    the vocabularies and legacy tables are class variables, and the expensive
    steps memoize on their arguments alone.
    """

    # BCP-47's "undetermined". Docling does *not* accept it as an OCR language
    # an empty `lang` list already says "let the engine decide"
    _UNDETERMINED = "und"

    _RESERVED = frozenset({MULTIPLE})

    _DOCS_URL = "https://docling-project.github.io/docling/concepts/OCR/"

    # Retired ways of asking an engine to decide for itself. An empty `lang`
    # list says the same thing, so these point there rather than at a
    # replacement tag.
    _AUTO_TOKENS = frozenset({"auto", "osd", "und"})

    # Engine tokens naming a script rather than a language. Reached only when
    # the selected engine does not define them -- for PP-OCR they are real
    # recognizers and resolve as passthroughs long before this. Docling has no
    # engine-independent script family to redirect them to, so they get the same
    # message `und-<Script>` gets: name a language written in the script.
    _SCRIPT_NAME_TOKENS = frozenset(
        {"latin", "cyrillic", "arabic", "devanagari", "bengali"}
    )

    # Engine-native tokens mapped to their BCP-47 replacement, for the error
    # message. Every entry here raises: these are the tokens reached *after* the
    # selected engine's own vocabulary has been consulted, so a token in this
    # table belongs to some other engine (`chi_sim` asked of RapidOCR) or to no
    # engine at all (`auto`, `jp`). Matched before the tag is parsed, because
    # several are structurally valid BCP-47 for an entirely different language.
    _LEGACY_HINTS: Mapping[str, str] = MappingProxyType(
        {
            "chinese": "zh-Hans",
            "ch": "zh-Hans",
            "ch_sim": "zh-Hans",
            "chi_sim": "zh-Hans",
            "zh_cn": "zh-Hans",
            "ch_tra": "zh-Hant",
            "chi_tra": "zh-Hant",
            "chinese_cht": "zh-Hant",
            "zh_tw": "zh-Hant",
            "english": "en",
            "japan": "ja",
            "jp": "ja",
            "korean": "ko",
            "multilingual": MULTIPLE,
            "multi": MULTIPLE,
            # PP-OCR serves East Slavic with one recognizer; `ru` is the
            # language of it docling canonicalizes back to `eslav`, so the hint
            # round trips.
            "eslav": "ru",
            "rs_latin": "sr-Latn",
            "rs_cyrillic": "sr-Cyrl",
            "tjk": "tg",
            "aze_cyrl": "az-Cyrl",
            "uzb_cyrl": "uz-Cyrl",
            "srp_latn": "sr-Latn",
        }
    )

    #: Subtags docling's `(language, script)` form cannot carry. A region is
    #: dropped by design (`de-DE` and `de-AT` are one recognizer); the rest
    #: appearing in a parse means the tag was only accepted by attaching part of
    #: the token to a subtag that is then thrown away.
    _LOSSY_SUBTAGS = frozenset({"extlangs", "variants", "private", "extensions"})

    #: Prefix of the tessdata script-family files, e.g. `script/Cyrillic`. The
    #: bare script name is deliberately *not* accepted: `Lao` is a tessdata
    #: script file and also a valid BCP-47 primary subtag.
    TESSERACT_SCRIPT_FILE_PREFIX = "script/"

    #: tessdata `script/` file name -> ISO 15924 code, for `script/<Name>` input.
    #: Only the file names are load-bearing: a script file is handed to the
    #: engine as the token the user wrote, so the code is what documents which
    #: script each file covers.
    _TESSERACT_SCRIPT_FILES: Mapping[str, str] = MappingProxyType(
        {
            "Arabic": "Arab",
            "Armenian": "Armn",
            "Bengali": "Beng",
            "Canadian_Aboriginal": "Cans",
            "Cherokee": "Cher",
            "Cyrillic": "Cyrl",
            "Devanagari": "Deva",
            "Ethiopic": "Ethi",
            "Fraktur": "Latf",
            "Georgian": "Geor",
            "Greek": "Grek",
            "Gujarati": "Gujr",
            "Gurmukhi": "Guru",
            "Hangul": "Hang",
            "HanS": "Hans",
            "HanT": "Hant",
            "Hebrew": "Hebr",
            "Japanese": "Jpan",
            "Kannada": "Knda",
            "Khmer": "Khmr",
            "Lao": "Laoo",
            "Latin": "Latn",
            "Malayalam": "Mlym",
            "Myanmar": "Mymr",
            "Oriya": "Orya",
            "Sinhala": "Sinh",
            "Syriac": "Syrc",
            "Tamil": "Taml",
            "Telugu": "Telu",
            "Thaana": "Thaa",
            "Thai": "Thai",
            "Tibetan": "Tibt",
            "Vietnamese": "Latn",
        }
    )

    #: Tesseract tokens naming a model no language tag can express.
    _TESSERACT_UNREPRESENTABLE: Mapping[str, str] = MappingProxyType(
        {
            "chi_sim_vert": "vertical text",
            "chi_tra_vert": "vertical text",
            "jpn_vert": "vertical text",
            "kor_vert": "vertical text",
            "hans_vert": "vertical text",
            "hant_vert": "vertical text",
            "hangul_vert": "vertical text",
            "japanese_vert": "vertical text",
            "ita_old": "a historical orthography",
            "spa_old": "a historical orthography",
            "kat_old": "a historical orthography",
            "equ": "mathematical notation rather than a language",
        }
    )

    #: The PP-OCR recognizers, shared by RapidOCR and the KServe v2 client.
    #: `eslav` resolves to `ru-Cyrl` because that is what round trips: the
    #: forward table sends `ru-Cyrl` back to `eslav`. The script recognizers
    #: (`latin`, `cyrillic`, ...) are passthroughs, having no language to
    #: canonicalize to.
    _PPOCR = _NativeVocabulary(
        languages=MappingProxyType(
            {
                # PP-OCR's `ch` is Simplified Chinese; BCP-47 `ch` is Chamorro.
                "ch": "zh-Hans",
                "chinese": "zh-Hans",
                "chinese_cht": "zh-Hant",
                "japan": "ja-Jpan",
                "korean": "ko-Kore",
                # PP-OCR's `ka` is Kannada; BCP-47 `ka` is Georgian.
                "ka": "kn-Knda",
                "rs_latin": "sr-Latn",
                "eslav": "ru-Cyrl",
                "french": "fr-Latn",
                "german": "de-Latn",
            }
        ),
        passthrough=frozenset({"latin", "cyrillic", "arabic", "devanagari"}),
    )

    #: The tessdata language files, shared by both Tesseract bindings. The
    #: `*_cyrl` and `*_latn` names are listed even though `langcodes` happens to
    #: read the underscore as a subtag separator -- that is luck, not contract.
    _TESSERACT = _NativeVocabulary(
        languages=MappingProxyType(
            {
                "chi_sim": "zh-Hans",
                # Parses as the bogus `zh-tra`, which maximizes to zh-Hans -- Simplified
                # output for a Traditional request. The whole reason the native table
                # must run before langcodes rather than after it.
                "chi_tra": "zh-Hant",
                "srp_latn": "sr-Latn",
                "aze_cyrl": "az-Cyrl",
                "uzb_cyrl": "uz-Cyrl",
                "deu_latf": "de-Latf",
                # tessdata's legacy name for German Fraktur; ISO 639-3 `frk` is Frankish.
                "frk": "de-Latf",
            }
        ),
        script_files=_TESSERACT_SCRIPT_FILES,
        unrepresentable=_TESSERACT_UNREPRESENTABLE,
    )

    #: The EasyOCR recognition codes.
    _EASYOCR = _NativeVocabulary(
        languages=MappingProxyType(
            {
                # BCP-47 reads both of these as Chamorro with a junk variant subtag.
                "ch_sim": "zh-Hans",
                "ch_tra": "zh-Hant",
                "rs_cyrillic": "sr-Cyrl",
                "rs_latin": "sr-Latn",
                "tjk": "tg-Cyrl",
                # EasyOCR's `ang` is Angika, filed under Devanagari; BCP-47 `ang` is Old
                # English. CLDR's likely script for Angika is Latin, so the script has to
                # be written out here.
                "ang": "anp-Deva",
                # EasyOCR's `mah` is Magahi; BCP-47 normalizes `mah` to `mh`, Marshallese.
                "mah": "mag-Deva",
                # Tabasaran is written in Cyrillic; CLDR's likely script for it is Latin.
                "tab": "tab-Cyrl",
            }
        ),
    )

    #: The nemotron-OCR recognizers.
    _NEMOTRON = _NativeVocabulary(
        languages=MappingProxyType(
            {
                "english": "en-Latn",
                "multilingual": "mul",
            }
        ),
    )

    #: `OcrOptions.kind` -> the engine vocabulary whose own codes are accepted
    #: alongside BCP-47. Several kinds share one: the two Tesseract bindings read
    #: the same tessdata names, and the KServe client addresses the same PP-OCR
    #: recognizers as local RapidOCR. A kind that is absent -- `ocrmac`, whose
    #: engine speaks BCP-47 already, or any out-of-tree engine -- accepts BCP-47
    #: and nothing else. `auto` is handled by `_vocabulary_for`.
    _NATIVE_VOCABULARIES: Mapping[str, _NativeVocabulary] = MappingProxyType(
        {
            "rapidocr": _PPOCR,
            "kserve_v2_ocr": _PPOCR,
            "tesseract": _TESSERACT,
            "tesserocr": _TESSERACT,
            "easyocr": _EASYOCR,
            "nemotron-ocr": _NEMOTRON,
        }
    )

    #: The kind that has not chosen an engine yet.
    _AUTO_KIND = "auto"

    #: Native tokens accepted when no engine has been chosen yet, as
    #: `OcrAutoOptions` does until its selection loop runs. Assigned right below
    #: the class body: deriving it needs the staticmethods defined further down.
    #: See `_derive_unambiguous_native_languages`.
    UNAMBIGUOUS_NATIVE_LANGUAGES: ClassVar[Mapping[str, str]]

    #: The `auto` kind's vocabulary: the tokens above, and no passthroughs --
    #: a passthrough is only meaningful to the engine that defines it. Assigned
    #: below the class body for the same reason.
    _AUTO_VOCABULARY: ClassVar[_NativeVocabulary]

    @staticmethod
    def canonicalize_ocr_language_tags(
        values: Sequence[str], kind: str | None = None
    ) -> list[str]:
        """Canonicalize a user-supplied language list into BCP-47 tags.

        `kind` is the `OcrOptions.kind` of the selected engine, whose native
        codes are accepted alongside BCP-47; see `parse_ocr_language`.
        """
        return [
            language.tag
            for language in OcrLanguageResolver.parse_ocr_languages(values, kind)
        ]

    @staticmethod
    def parse_ocr_languages(
        values: Sequence[str], kind: str | None = None
    ) -> tuple[OcrLanguage, ...]:
        """Parse a list of language requests, enforcing the reserved-tag rule.

        An empty list is valid and means "the engine's own default"; every
        engine decides what that is, and for Tesseract it is per-page script
        detection.

        Duplicates are dropped, preserving the order the user wrote -- for
        engines that join languages (Tesseract's `+`) that order is preference
        order.
        """
        languages: list[OcrLanguage] = []
        for value in values:
            language = OcrLanguageResolver.parse_ocr_language(value, kind)
            if language not in languages:
                languages.append(language)

        reserved = [lang.tag for lang in languages if lang.is_reserved]
        if reserved and len(languages) > 1:
            raise ValueError(
                f"The reserved OCR language tag {reserved[0]!r} must be used on its "
                f"own, but it was combined with "
                f"{[lang.tag for lang in languages if lang.tag != reserved[0]]}."
            )
        return tuple(languages)

    @staticmethod
    def parse_ocr_language(value: str, kind: str | None = None) -> OcrLanguage:
        """Parse and canonicalize one user-supplied OCR language.

        `kind` is the `OcrOptions.kind` of the selected engine, whose own
        language codes are then accepted alongside BCP-47, e.g. `"rapidocr"` so
        that PP-OCR's `ch` means Simplified Chinese rather than BCP-47's
        Chamorro. `None`, or a kind with no native vocabulary, accepts BCP-47
        only.

        Raises:
            ValueError: The value is empty, belongs to a different engine, names
                a model no language tag can express, or is not a valid BCP-47
                tag.
        """
        token = value.strip()
        if not token:
            raise OcrLanguageResolver._invalid(value, "the value is empty.")

        lowered = token.lower()
        if lowered in OcrLanguageResolver._RESERVED:
            return OcrLanguage(language=lowered)

        # Before anything else: the selected engine's own vocabulary wins.
        # Several native tokens are structurally valid BCP-47 for a different
        # language (`ch` is Chamorro, `ka` is Georgian) and one, `chi_tra`,
        # parses to the right language with the *wrong* script -- Simplified for
        # a Traditional request. Consulting the table afterwards would resolve
        # all of those wrongly.
        if kind is not None:
            native = OcrLanguageResolver._resolve_native_ocr_language(token, kind)
            if native is not None:
                return native

        # Runs before parsing for the same reason: a legacy token can be valid
        # BCP-47 for something else entirely.
        if (
            lowered in OcrLanguageResolver._AUTO_TOKENS
            or lowered in OcrLanguageResolver._SCRIPT_NAME_TOKENS
        ):
            raise OcrLanguageResolver._no_undetermined(value)

        hint = OcrLanguageResolver._LEGACY_HINTS.get(lowered)
        if hint is not None:
            raise OcrLanguageResolver._invalid(value, f"use {hint!r} instead.")

        return OcrLanguageResolver._parse_bcp47(value)

    @staticmethod
    def match_ocr_language(
        language: OcrLanguage, supported: Sequence[str], *, max_distance: int = 10
    ) -> str | None:
        """Pick the closest entry of a BCP-47-ish engine vocabulary, or `None`.

        Only useful where the engine's own vocabulary is itself BCP-47 and
        carries regions (Apple Vision). Everything else should use an explicit
        table.
        """
        if not supported:
            return None
        match, distance = langcodes.closest_match(
            language.tag, list(supported), max_distance=max_distance
        )
        return None if match == OcrLanguageResolver._UNDETERMINED else match

    @staticmethod
    @lru_cache(maxsize=256)
    def _default_script_for_language(language: str) -> str | None:
        """The script CLDR likely-subtags associate with a primary subtag."""
        try:
            return langcodes.Language.get(language).maximize().script
        except langcodes.LanguageTagError:
            return None

    @staticmethod
    def _invalid(value: str, reason: str) -> ValueError:
        return ValueError(
            f"Invalid OCR language {value!r}. Docling uses BCP-47 language tags; "
            f"{reason} See {OcrLanguageResolver._DOCS_URL}"
        )

    @staticmethod
    def _plain_bcp47_tag(value: str) -> str | None:
        """The canonical tag `value` has as a plain BCP-47 tag, or `None`.

        `None` means "not a legitimate tag for anything", which covers both
        outright invalid tokens and the ones `langcodes` accepts only by parking
        part of the token in a subtag docling then discards: `chi_tra` parses as
        `zh` plus an extlang `tra`, which canonicalizes to `zh-Hans` -- the wrong
        script, and not a spelling any user of Chinese would actually write.

        Used to reason *about* the native vocabularies: an alias only shadows a
        real BCP-47 reading when this returns one, so this is what separates a
        genuine ambiguity like `ch` (Chamorro) from a parse accident like
        `chi_tra`.
        """
        try:
            parsed = langcodes.Language.get(value.strip(), normalize=True)
        except langcodes.LanguageTagError:
            return None
        if OcrLanguageResolver._LOSSY_SUBTAGS.intersection(parsed.to_dict()):
            return None
        try:
            return OcrLanguageResolver._parse_bcp47(value).tag
        except ValueError:
            return None

    @staticmethod
    @lru_cache(maxsize=256)
    def _resolve_native_ocr_language(value: str, kind: str) -> "OcrLanguage | None":
        """What the engine behind `kind` makes of `value`, or `None`.

        `None` means "not one of this engine's tokens" -- or that the engine has
        no native vocabulary at all -- and the caller falls through to ordinary
        BCP-47 parsing. A token naming a script recognizer rather than a
        language comes back as a passthrough, carrying the token itself.

        Raises:
            ValueError: The token names a real model of this engine that a
                `(language, script)` pair cannot express, such as Tesseract's
                vertical-text or historical-orthography files.
        """
        vocabulary = OcrLanguageResolver._vocabulary_for(kind)
        if vocabulary is None:
            return None

        token = value.strip()
        lowered = token.lower()

        prefix = OcrLanguageResolver.TESSERACT_SCRIPT_FILE_PREFIX
        if vocabulary.script_files is not None and lowered.startswith(prefix):
            name = token[len(prefix) :]
            if name.lower().endswith("_vert"):
                raise OcrLanguageResolver._unrepresentable(value, "vertical text")
            for file_name in vocabulary.script_files:
                if file_name.lower() == name.lower():
                    # A real traineddata file named after a script, not a language.
                    return OcrLanguage(native=f"{prefix}{file_name}")
            return None

        if lowered in vocabulary.passthrough:
            return OcrLanguage(native=lowered)

        # Scoped to the engine that owns them: `equ` asked of RapidOCR is better
        # served by the ordinary "not a valid tag" message.
        reason = vocabulary.unrepresentable.get(lowered)
        if reason is not None:
            raise OcrLanguageResolver._unrepresentable(value, reason)

        canonical = vocabulary.languages.get(lowered)
        return (
            None if canonical is None else OcrLanguageResolver._parse_bcp47(canonical)
        )

    @staticmethod
    def _vocabulary_for(kind: str) -> "_NativeVocabulary | None":
        """The native vocabulary an engine kind selects, or `None` for BCP-47 only."""
        if kind == OcrLanguageResolver._AUTO_KIND:
            return OcrLanguageResolver._AUTO_VOCABULARY
        return OcrLanguageResolver._NATIVE_VOCABULARIES.get(kind)

    @staticmethod
    def _no_undetermined(value: str) -> ValueError:
        return ValueError(
            f"The OCR language {value!r} is not supported. Docling has no "
            f"'undetermined' language and no script families: leave the OCR language "
            f"list empty to let the engine choose (which is how Tesseract's per-page "
            f"script detection is enabled), or name a language written in the script "
            f"you want. See {OcrLanguageResolver._DOCS_URL}"
        )

    @staticmethod
    def _no_linguistic_content(value: str) -> ValueError:
        return ValueError(
            f"The OCR language {value!r} (no linguistic content) is not supported. "
            f"To skip OCR, turn it off instead: `--no-ocr` on the CLI, or "
            f"`do_ocr=False` in the pipeline options. See "
            f"{OcrLanguageResolver._DOCS_URL}"
        )

    @staticmethod
    def _unrepresentable(value: str, reason: str) -> ValueError:
        return ValueError(
            f"The OCR language {value!r} names a model docling cannot address by "
            f"language tag, because it selects {reason}. Docling stores OCR languages "
            f"as BCP-47 tags, which cannot express that distinction. Name the "
            f"language itself instead. See {OcrLanguageResolver._DOCS_URL}"
        )

    @staticmethod
    @lru_cache(maxsize=256)
    def _parse_bcp47(value: str) -> OcrLanguage:
        """Canonicalize one value as a plain BCP-47 tag, with no engine vocabulary."""
        token = value.strip()
        if not token:
            raise OcrLanguageResolver._invalid(value, "the value is empty.")

        lowered = token.lower()
        if lowered in OcrLanguageResolver._RESERVED:
            return OcrLanguage(language=lowered)

        try:
            parsed = langcodes.Language.get(token, normalize=True)
        except langcodes.LanguageTagError as err:
            raise OcrLanguageResolver._invalid(value, f"{err}.") from err

        # Load-bearing: tokens like `klingon` are structurally valid primary
        # subtags and only the IANA registry rejects them.
        if not parsed.is_valid():
            raise OcrLanguageResolver._invalid(
                value, "the tag is not registered with IANA."
            )

        if lowered == "zxx":
            # A valid BCP-47 tag that langcodes maximizes to `zxx-Latn-US`; docling
            # has no recognizer for "no linguistic content" and will not pretend to.
            raise OcrLanguageResolver._no_linguistic_content(value)

        if (
            parsed.language is None
            or parsed.language == OcrLanguageResolver._UNDETERMINED
        ):
            # `und` and `und-<Script>`. Docling has no undetermined language and
            # no script families: leaving `lang` empty already means "engine
            # decides", and a script is named by naming a language written in it.
            raise OcrLanguageResolver._no_undetermined(value)

        script = parsed.script or parsed.maximize().script
        return OcrLanguage(language=parsed.language, script=script)

    @staticmethod
    def _derive_unambiguous_native_languages() -> Mapping[str, str]:
        """The native tokens that are safe with no engine chosen.

        A token qualifies when every vocabulary that has it agrees on the
        meaning, and when its plain BCP-47 reading does not contradict that
        meaning. The second condition is what drops `ch`, `ka`, `ang`, `mah`,
        `frk` and `tab`: with no engine selected there is nothing to prefer the
        native reading over the standard one, so docling asks the user to
        disambiguate rather than guessing. `chi_tra` and friends survive it,
        because `_plain_bcp47_tag` refuses to call their lossy parse a reading at
        all.

        Derived rather than written out, so that adding an alias to a per-engine
        table cannot silently widen the engine-less vocabulary.
        """
        seen: dict[str, str] = {}
        conflicting: set[str] = set()
        for vocabulary in OcrLanguageResolver._NATIVE_VOCABULARIES.values():
            for token, canonical in vocabulary.languages.items():
                if token in seen and seen[token] != canonical:
                    conflicting.add(token)
                seen[token] = canonical

        return MappingProxyType(
            {
                token: canonical
                for token, canonical in seen.items()
                if token not in conflicting
                and OcrLanguageResolver._plain_bcp47_tag(token) in (None, canonical)
            }
        )


OcrLanguageResolver.UNAMBIGUOUS_NATIVE_LANGUAGES = (
    OcrLanguageResolver._derive_unambiguous_native_languages()
)
OcrLanguageResolver._AUTO_VOCABULARY = _NativeVocabulary(
    languages=OcrLanguageResolver.UNAMBIGUOUS_NATIVE_LANGUAGES
)
