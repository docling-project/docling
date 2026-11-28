# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT
# Adapted from mwparserfromhell v0.7.2 (MIT).
# SPDX-FileCopyrightText: 2012-2025 Ben Kurtovic

"""State used while recognizing tags and finding valid tokenization routes."""

from dataclasses import dataclass, field
from enum import Enum

from docling.backend.xml.mediawiki._syntax import normalize_tag_name


class ParserError(ValueError):
    """The wikitext token stream violates an internal parser invariant."""


class TokenizerWorkLimitError(ParserError):
    """Tokenization exhausted the work budget for this page."""


class BadRoute(Exception):
    """Raised internally when the current tokenization route is invalid."""

    def __init__(self, context: int = 0) -> None:
        super().__init__()
        self.context = context


class TagOpenData:
    """Stores data about an HTML open tag, like ``<ref name="foo">``."""

    CX_NAME = 1 << 0
    CX_ATTR_READY = 1 << 1
    CX_ATTR_NAME = 1 << 2
    CX_ATTR_VALUE = 1 << 3
    CX_QUOTED = 1 << 4
    CX_NOTE_SPACE = 1 << 5
    CX_NOTE_EQUALS = 1 << 6
    CX_NOTE_QUOTE = 1 << 7

    def __init__(self) -> None:
        self.context = self.CX_NAME
        self.padding_buffer = {"first": "", "before_eq": "", "after_eq": ""}
        self.quoter: str | None = None
        self.reset = 0


class Sentinel(Enum):
    START = 0
    END = 1


START = Sentinel.START
END = Sentinel.END

MARKERS: set[str | Sentinel] = {*"""{}[]<>|=&'"#*;:/-!\n""", START, END}
URI_SCHEME_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+.-"


@dataclass
class ClosingPositions:
    """Last possible closing delimiters, including a position for each tag name."""

    brace: int = -1
    bracket: int = -1
    comment: int = -1
    table: int = -1
    tag_end: int = -1
    tags: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_segments(cls, segments: list[str]) -> "ClosingPositions":
        positions = cls()
        previous = before_previous = ""
        closing_tag_start: int | None = None
        for index, segment in enumerate(segments):
            if segment == "}":
                if previous == "}":
                    positions.brace = index - 1
                elif previous == "|":
                    positions.table = index - 1
            elif segment == "]":
                positions.bracket = index
            elif segment == ">":
                positions.tag_end = index
                if previous == before_previous == "-":
                    positions.comment = index - 2
                if closing_tag_start is not None:
                    name = normalize_tag_name(
                        "".join(segments[closing_tag_start + 2 : index])
                    )
                    positions.tags[name] = closing_tag_start
                    closing_tag_start = None
            elif segment == "<":
                closing_tag_start = None
            elif segment == "/" and previous == "<":
                closing_tag_start = index - 1
            before_previous, previous = previous, segment
        return positions
