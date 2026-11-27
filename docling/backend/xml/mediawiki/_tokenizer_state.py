# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT
# Adapted from mwparserfromhell v0.7.2 (MIT).
# SPDX-FileCopyrightText: 2012-2025 Ben Kurtovic

"""State used while recognizing tags and finding valid tokenization routes."""

from dataclasses import dataclass
from enum import Enum


class ParserError(ValueError):
    """The wikitext token stream violates an internal parser invariant."""


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
    """Last possible closing delimiter for routes that can scan to EOF."""

    brace: int = -1
    bracket: int = -1
    comment: int = -1
    table: int = -1
    tag: int = -1

    @classmethod
    def from_segments(cls, segments: list[str]) -> "ClosingPositions":
        positions = cls()
        previous = before_previous = ""
        for index, segment in enumerate(segments):
            if segment == "}":
                if previous == "}":
                    positions.brace = index - 1
                elif previous == "|":
                    positions.table = index - 1
            elif segment == "]":
                positions.bracket = index
            elif segment == ">" and previous == before_previous == "-":
                positions.comment = index - 2
            elif segment == "/" and previous == "<":
                positions.tag = index - 1
            before_previous, previous = previous, segment
        return positions
