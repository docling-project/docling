# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Canonicalization of OCR language requests to BCP-47 (RFC 5646).

The user can provide as input language either a supported BCP-47 code or a language native to the
OCR engine.

Docling reduces every request to a `(language, script)` pair.
Per-engine adapters translate that pair into the engine's own notation
(see `docling.models.base_ocr_model.BaseOcrModel.map_ocr_language`).

Region is discarded once it has inferred the script: `zh-CN` and `zh-Hans` are
the same recognizer, and `de-DE` vs `de-AT` is a distinction no OCR engine
docling supports can act on.
"""

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Literal, overload

import langcodes
from pydantic import BaseModel, ConfigDict

_log = logging.getLogger(__name__)


class OcrLanguageSupport(BaseModel):
    """What one engine instance can serve, split by how the user must write it.

    Every engine answers the same question about each code in its own
    vocabulary -- can a canonical tag name this model? -- and the two answers go
    to different halves. Keeping them apart is what stops a code that cannot be
    canonicalized from being dropped on the floor, which is how RapidOCR's
    script recognizers went unadvertised.

    Attributes:
        bcp47: Canonical tags, in the shortest spelling that reaches each
            recognizer (`OcrLanguage.short_tag`).
        native: The engine's own codes for the models no `(language, script)`
            pair can name, bare: the `native:` prefix is added when the list is
            rendered, not stored here.
    """

    model_config = ConfigDict(frozen=True)

    bcp47: list[str] = []
    native: list[str] = []


class OcrLanguage(BaseModel):
    """One canonicalized OCR language request: a BCP-47 (language, script) pair.

    Attributes:
        bcp47_language: Primary subtag, lowercase. May be the reserved subtag
            `mul`.
        bcp47_script: ISO 15924 script code in title case. `None` only for the
            bare reserved tags.
        native: An engine's own token, stripped of the `native:` prefix, for
            the models no `(bcp47_language, bcp47_script)` pair can name --
            PP-OCR's script recognizers (`arabic`, `cyrillic`) and Tesseract's
            `script/<Name>` files. Set only for a passthrough, where it excludes
            `bcp47_language` and `bcp47_script` and is what `tag` re-prefixes;
            `None` for every ordinary BCP-47 request.
    """

    model_config = ConfigDict(frozen=True)

    bcp47_language: str | None = None
    bcp47_script: str | None = None
    native: str | None = None

    @property
    def tag(self) -> str:
        """How this request is written back into `OcrOptions.lang`.

        A canonical BCP-47 tag (`de-Latn`, `mul`), or, for a passthrough, the
        engine's own token behind the `native:` prefix. Re-attaching the prefix
        is what keeps `lang` idempotent: revalidating `["native:arabic"]` must
        not move it.
        """
        if self.native is not None:
            return f"{OcrLanguageResolver._NATIVE_PREFIX}{self.native}"
        return self.bcp47

    @property
    def bcp47(self) -> str:
        """The `(language, script)` pair written as a BCP-47 tag.

        What an engine's own table is keyed on. Unlike `tag` it never carries
        the `native:` prefix, so it is empty for a passthrough, which names no
        language at all.
        """
        return (
            f"{self.bcp47_language}-{self.bcp47_script}"
            if self.bcp47_script
            else self.bcp47_language or ""
        )

    @property
    def short_tag(self) -> str:
        """The shortest spelling that canonicalizes back to this language.

        What an engine advertises. The script is dropped when it is the one CLDR
        infers anyway, so `de-Latn` is offered as `de` and `zh-Hans` as `zh`,
        while `zh-Hant`, `sr-Latn` and `de-Latf` keep theirs: written bare they
        would name a different recognizer. A passthrough and the reserved `mul`
        are already as short as they get.
        """
        if self.native is not None or not self.has_default_script:
            return self.tag
        assert self.bcp47_language is not None
        return self.bcp47_language

    @property
    def is_passthrough(self) -> bool:
        """An engine token that no `(language, script)` pair can express.

        PP-OCR's script recognizers (`arabic`, `cyrillic`) and Tesseract's
        `script/<Name>` files: real models, named after a script rather than a
        language, and handed to the engine untouched.
        """
        return self.native is not None

    @property
    def is_multilingual(self) -> bool:
        return self.bcp47_language == OcrLanguageResolver.MULTIPLE

    @property
    def has_default_script(self) -> bool:
        """Whether `script` is the script CLDR considers likely for `language`.

        `de-Latn` and `en-Latn` do; `az-Cyrl` and `uz-Cyrl` do not. Engines use
        this to decide whether the primary subtag alone still identifies the
        right recognizer.
        """
        if (
            self.bcp47_language is None
            or self.bcp47_language == OcrLanguageResolver.MULTIPLE
        ):
            return False
        return self.bcp47_script == OcrLanguageResolver._default_script_for_language(
            self.bcp47_language
        )

    def __str__(self) -> str:
        return self.tag


class OcrLanguageResolver:
    """Canonicalizes user-supplied OCR language tokens into `OcrLanguage`.

    A namespace rather than an object: every entry point is a `@staticmethod`,
    the vocabularies and legacy tables are class variables, and the expensive
    steps memoize on their arguments alone.
    """

    # Multiple languages: the engine's broadest multilingual model.
    MULTIPLE = "mul"

    _OCR_DOCS_URL = "https://docling-project.github.io/docling/concepts/OCR/"

    # When docling's input language has the following prefix, it passthrough the OCR engine
    _NATIVE_PREFIX = "native:"

    # BCP-47's "undetermined". Docling does *not* accept it as an OCR language
    _UNDETERMINED = "und"

    # BCP-47's "no linguistic content". Docling does *not* accept it as an OCR language
    _NO_LINGUISTIC_CONTENT = "zxx"

    @staticmethod
    def canonicalize_ocr_languages(values: Sequence[str]) -> list[OcrLanguage]:
        """Canonicalize a list of language requests, enforcing the reserved-tag rule.

        Callers that store strings -- `OcrOptions.lang`, the remote CLI -- read
        `.tag` off each result.

        An empty list is valid and means "the engine's own default"; every
        engine decides what that is, and for Tesseract it is per-page script
        detection.

        Duplicates are dropped, preserving the order the user wrote

        Raise a ValueError if the "multiple" language has been used together with other languages
        """
        languages: list[OcrLanguage] = []
        for value in values:
            language = OcrLanguageResolver.canonicalize_ocr_language(value)
            if language not in languages:
                languages.append(language)

        # Validate if the "multiple" language has been used together with other languages
        reserved = [
            lang.tag
            for lang in languages
            if lang.bcp47_language == OcrLanguageResolver.MULTIPLE
            and lang.bcp47_script is None
        ]
        if reserved and len(languages) > 1:
            raise ValueError(
                f"The reserved OCR language tag {reserved[0]!r} must be used on its "
                f"own, but it was combined with "
                f"{[lang.tag for lang in languages if lang.tag != reserved[0]]}."
            )
        return languages

    @overload
    @staticmethod
    def canonicalize_ocr_language(
        value: str,
        *,
        raise_exception: Literal[True] = True,
    ) -> OcrLanguage: ...

    @overload
    @staticmethod
    def canonicalize_ocr_language(
        value: str,
        *,
        raise_exception: Literal[False],
    ) -> "OcrLanguage | None": ...

    @staticmethod
    def canonicalize_ocr_language(
        value: str,
        *,
        raise_exception: bool = True,
    ) -> "OcrLanguage | None":
        """Canonicalize one user-supplied OCR language.

        `raise_exception`: When `False`, a value that cannot be resolved
            returns `None` instead of raising.

        Raises:
            ValueError
        """
        try:
            token = value.strip()
            if not token:
                raise OcrLanguageResolver._invalid(value, "the value is empty.")
            lowered = token.lower()

            # Passthrough the native tokens
            if lowered.startswith(OcrLanguageResolver._NATIVE_PREFIX):
                native = token[len(OcrLanguageResolver._NATIVE_PREFIX) :].strip()
                return OcrLanguage(native=native)

            # Expect a BCP47 input
            return OcrLanguageResolver._parse_bcp47(lowered)
        except ValueError:
            if raise_exception:
                raise
            return None

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
        # The BCP-47 pair, never `tag`: a passthrough's engine code is not a tag
        # and has nothing to match against.
        match, distance = langcodes.closest_match(
            language.bcp47, list(supported), max_distance=max_distance
        )
        return None if match == OcrLanguageResolver._UNDETERMINED else match

    @staticmethod
    @lru_cache(maxsize=256)
    def _default_script_for_language(bcp47_language: str) -> str | None:
        """The script CLDR likely-subtags associate with a primary subtag."""
        try:
            return langcodes.Language.get(bcp47_language).maximize().script
        except langcodes.LanguageTagError:
            return None

    @staticmethod
    def _invalid(value: str, reason: str) -> ValueError:
        return ValueError(
            f"Invalid OCR language {value!r}. Docling uses BCP-47 language tags; "
            f"{reason} See {OcrLanguageResolver._OCR_DOCS_URL}"
        )

    @staticmethod
    @lru_cache(maxsize=256)
    def _parse_bcp47(bcp47_tobe: str) -> OcrLanguage:
        """Canonicalize one value as a plain BCP-47 tag, with no engine vocabulary."""

        if bcp47_tobe == OcrLanguageResolver.MULTIPLE:
            return OcrLanguage(bcp47_language=bcp47_tobe)

        try:
            parsed = langcodes.Language.get(bcp47_tobe, normalize=True)
        except langcodes.LanguageTagError as err:
            raise OcrLanguageResolver._invalid(bcp47_tobe, f"{err}.") from err

        if not parsed.is_valid():
            raise OcrLanguageResolver._invalid(
                bcp47_tobe, "the tag is not registered with IANA."
            )

        if bcp47_tobe == OcrLanguageResolver._NO_LINGUISTIC_CONTENT:
            # Docling has no recognizer for "no linguistic content"
            raise OcrLanguageResolver._invalid(
                bcp47_tobe,
                "it names 'no linguistic content'. To skip OCR, turn it off "
                "instead: `--no-ocr` on the CLI, or `do_ocr=False` in the "
                "pipeline options.",
            )

        if (
            parsed.language is None
            or parsed.language == OcrLanguageResolver._UNDETERMINED
        ):
            # Docling has no undetermined language and no script families
            raise OcrLanguageResolver._invalid(
                bcp47_tobe,
                "docling has no 'undetermined' language and no script families: "
                "leave the OCR language list empty to let the engine choose "
                "(which is how Tesseract's per-page script detection is enabled), "
                "or name a language written in the script you want.",
            )

        # An explicit script wins, so `de-Latf` keeps Fraktur; only a tag written
        # without one falls back to CLDR's likely script, sending `de` to `de-Latn`.
        script = parsed.script or parsed.maximize().script
        return OcrLanguage(bcp47_language=parsed.language, bcp47_script=script)
