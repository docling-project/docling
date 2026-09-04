# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Canonicalization of OCR language requests to BCP-47 (RFC 5646).

An OCR language is written either as a code native to the selected engine, which is what a bare
value means and reaches the engine untouched, or as a BCP-47 tag behind the `iso:` prefix.

A tagged request is reduced to a `(language, script)` pair.
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
            recognizer (`OcrLanguage.short_tag()`), bare: the `iso:` prefix is
            added when the list is rendered, not stored here.
        native: The engine's own codes for the models no `(language, script)`
            pair can name.
    """

    model_config = ConfigDict(frozen=True)

    bcp47: list[str] = []
    native: list[str] = []


class OcrLanguage(BaseModel):
    """One OCR language request: an engine's own token, or a BCP-47 pair.

    Attributes:
        bcp47_language: Primary subtag, lowercase. Set only for a request
            written behind the `iso:` prefix.
        bcp47_script: ISO 15924 script code in title case.
        native: An engine's own token, exactly as the user wrote it. This is
            what a value carrying no `iso:` prefix becomes, whatever it looks
            like: `deu`, `ch`, `script/Cyrillic`, the stem of a traineddata file
            of your own. It excludes `bcp47_language` and `bcp47_script`, and is
            `None` for every `iso:` request.
    """

    model_config = ConfigDict(frozen=True)

    bcp47_language: str | None = None
    bcp47_script: str | None = None
    native: str | None = None

    def tag(self) -> str:
        """How this request is written back into `OcrOptions.lang`.

        The engine's own token, bare, or a canonical BCP-47 tag behind the
        `iso:` prefix. Re-attaching the prefix is what keeps `lang` idempotent:
        revalidating `["iso:de-Latn"]` must not read it as an engine token.
        """
        if self.native is not None:
            return self.native
        return f"{OcrLanguageResolver._ISO_PREFIX}{self.bcp47()}"

    def bcp47(self) -> str:
        """The `(language, script)` pair written as a BCP-47 tag.

        What an engine's own table is keyed on. Unlike `tag()` it never carries
        the `iso:` prefix, so it is empty for an engine token, which names no
        language at all.
        """
        return (
            f"{self.bcp47_language}-{self.bcp47_script}"
            if self.bcp47_script
            else self.bcp47_language or ""
        )

    def short_tag(self) -> str:
        """The shortest spelling that canonicalizes back to this language"""
        if self.native is not None:
            return self.native
        if not self.has_default_script():
            return self.bcp47()
        assert self.bcp47_language is not None
        return self.bcp47_language

    def is_passthrough(self) -> bool:
        """Utility function to check if native is none None"""
        return self.native is not None

    def has_default_script(self) -> bool:
        """Whether `script` is the script CLDR considers likely for `language`.

        `de-Latn` and `en-Latn` do; `az-Cyrl` and `uz-Cyrl` do not. Engines use
        this to decide whether the primary subtag alone still identifies the
        right recognizer.
        """
        if self.bcp47_language is None:
            return False
        try:
            # The script CLDR likely-subtags associate with the primary subtag
            default_script = (
                langcodes.Language.get(self.bcp47_language).maximize().script
            )
        except langcodes.LanguageTagError:
            default_script = None
        return self.bcp47_script == default_script

    def __str__(self) -> str:
        return self.tag()


class OcrLanguageResolver:
    """Canonicalizes user-supplied OCR language tokens into `OcrLanguage`.

    A namespace rather than an object: every entry point is a `@staticmethod`,
    the vocabularies and legacy tables are class variables, and the expensive
    steps memoize on their arguments alone.
    """

    _OCR_DOCS_URL = "https://docling-project.github.io/docling/concepts/OCR/"

    # Docling's input language is read as a BCP-47 tag only behind this prefix.
    _ISO_PREFIX = "iso:"

    # BCP-47's "undetermined". Docling does *not* accept it as an OCR language
    _UNDETERMINED = "und"

    # BCP-47's "no linguistic content". Docling does *not* accept it as an OCR language
    _NO_LINGUISTIC_CONTENT = "zxx"

    # BCP-47's "multiple languages". Docling does *not* accept it as an OCR language
    _MULTIPLE = "mul"

    @staticmethod
    def canonicalize_ocr_languages(values: Sequence[str]) -> list[OcrLanguage]:
        """Canonicalize a list of language requests.

        Callers that store strings -- `OcrOptions.lang`, the remote CLI -- read
        `.tag()` off each result.

        An empty list is valid and means "the engine's own default"; every
        engine decides what that is, and for Tesseract it is per-page script
        detection.

        Duplicates are dropped, preserving the order the user wrote
        """
        languages: list[OcrLanguage] = []
        for value in values:
            language = OcrLanguageResolver.canonicalize_ocr_language(value)
            if language not in languages:
                languages.append(language)
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

            # Only an `iso:` request is a BCP-47 tag. The prefix is matched case-insensitively
            if token.lower().startswith(OcrLanguageResolver._ISO_PREFIX):
                bcp47 = token[len(OcrLanguageResolver._ISO_PREFIX) :]
                return OcrLanguageResolver.canonicalize_bcp47(bcp47)

            # Passthrough the codes native to the engine
            return OcrLanguage(native=token)
        except ValueError:
            if raise_exception:
                raise
            return None

    @overload
    @staticmethod
    def canonicalize_bcp47(
        value: str,
        *,
        raise_exception: Literal[True] = True,
    ) -> OcrLanguage: ...

    @overload
    @staticmethod
    def canonicalize_bcp47(
        value: str,
        *,
        raise_exception: Literal[False],
    ) -> "OcrLanguage | None": ...

    @staticmethod
    @lru_cache(maxsize=256)
    def canonicalize_bcp47(
        value: str,
        *,
        raise_exception: bool = True,
    ) -> "OcrLanguage | None":
        """Canonicalize a value already known to be a tag, written without `iso:`.

        `raise_exception`: When `False`, a value that is not a tag docling
            accepts returns `None` instead of raising.

        Raises:
            ValueError
        """
        bcp47_tobe = value.strip().lower()
        try:
            if not bcp47_tobe:
                raise OcrLanguageResolver._invalid(bcp47_tobe, "the tag is empty.")

            if bcp47_tobe == OcrLanguageResolver._MULTIPLE:
                # No BCP-47 tag names a multilingual recognizer
                raise OcrLanguageResolver._invalid(
                    bcp47_tobe,
                    "it names 'multiple languages', which is not a language docling "
                    "can recognize. An engine that ships a multilingual model names "
                    "it with its own code, so write that code without the prefix "
                    "(`multilingual` for nemotron-OCR).",
                )

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
            language.bcp47(), list(supported), max_distance=max_distance
        )
        return None if match == OcrLanguageResolver._UNDETERMINED else match

    @staticmethod
    def _invalid(value: str, reason: str) -> ValueError:
        return ValueError(
            f"Invalid OCR language {OcrLanguageResolver._ISO_PREFIX}{value}. The "
            f"`{OcrLanguageResolver._ISO_PREFIX}` prefix marks a BCP-47 language tag; "
            f"{reason} See {OcrLanguageResolver._OCR_DOCS_URL}"
        )
