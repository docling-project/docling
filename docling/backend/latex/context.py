# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import threading
from dataclasses import dataclass
from typing import Optional

from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    Formatting,
    NodeItem,
)

_log = logging.getLogger(__name__)


@dataclass
class ParseContext:
    doc: DoclingDocument
    parent: NodeItem | None = None
    formatting: Formatting | None = None
    text_label: DocItemLabel | None = None


class MacroExpansionBudget:
    """Limit the expansion of user-defined macros in one document.

    Each expansion of a ``\\def`` or ``\\newcommand`` macro spends one
    expansion and the length of its result. When a limit is reached, or when
    ``stop_event`` is set because the conversion timed out, no more macros
    expand: the callers keep the remaining text unexpanded.
    """

    def __init__(
        self, max_chars: int, max_expansions: int, stop_event: threading.Event
    ) -> None:
        self.max_chars = max_chars
        self.max_expansions = max_expansions
        self.stop_event = stop_event
        self.chars = 0
        self.expansions = 0
        self._exceeded = False

    def is_stopped(self) -> bool:
        return self.stop_event.is_set()

    def is_exhausted(self) -> bool:
        return self._exceeded or self.stop_event.is_set()

    def remaining_chars(self) -> int:
        return max(0, self.max_chars - self.chars)

    def exceed(self) -> None:
        """Stop all further expansion, with one warning for the document."""
        if self._exceeded:
            return
        self._exceeded = True
        _log.warning(
            "LaTeX macro expansion limit reached "
            f"(max_macro_expansions={self.max_expansions}, "
            f"max_macro_expansion_chars={self.max_chars}). "
            "The remaining macros are kept unexpanded."
        )

    def spend(self, chars: int) -> bool:
        """Record one expansion of ``chars`` characters, if the budget allows it."""
        if self.is_exhausted():
            return False
        if (
            self.expansions + 1 > self.max_expansions
            or self.chars + chars > self.max_chars
        ):
            self.exceed()
            return False
        self.expansions += 1
        self.chars += chars
        return True
