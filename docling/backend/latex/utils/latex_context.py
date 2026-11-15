# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The pylatexenc parsing context shared by every LatexWalker the backend builds."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - import-time guard
    from pylatexenc.latexwalker import get_default_latex_context_db
    from pylatexenc.macrospec import EnvironmentSpec, MacroSpec, MacroStandardArgsParser

    _PYLATEXENC_AVAILABLE = True

    class _CmidruleArgsParser(MacroStandardArgsParser):
        """Read ``\\cmidrule(trim)[width]{a-b}``.

        The parenthesized trim is not expressible in an argspec. It is consumed
        here so it does not remain as text in the next cell.
        """

        def __init__(self) -> None:
            super().__init__("[{")

        def parse_args(self, w, pos, parsing_state=None):  # type: ignore[no-untyped-def]
            source = w.s
            index = pos
            limit = len(source)
            while index < limit and source[index] in " \t\r\n":
                index += 1
            if index < limit and source[index] == "(":
                close = source.find(")", index + 1, index + 32)
                if close != -1:
                    index = close + 1
            parsed, arg_pos, arg_len = super().parse_args(
                w, index, parsing_state=parsing_state
            )
            return parsed, pos, arg_pos + arg_len - pos

except ImportError:  # pragma: no cover - import-time guard
    _PYLATEXENC_AVAILABLE = False


def _build_context_db() -> Any:
    if not _PYLATEXENC_AVAILABLE:
        return None
    db = get_default_latex_context_db()
    # pylatexenc knows tabular, tabular* and tabularx but not longtable, whose
    # column specification would otherwise stay in the body and be read as the
    # first cell of the table.
    db.add_context_category(
        "docling-tables",
        environments=[EnvironmentSpec("longtable", "[{")],
        macros=[
            MacroSpec("cline", "{"),
            MacroSpec("specialrule", "{{{"),
            MacroSpec("rule", "[{{"),
            MacroSpec("toprule", "["),
            MacroSpec("midrule", "["),
            MacroSpec("bottomrule", "["),
            MacroSpec("cmidrule", _CmidruleArgsParser()),
        ],
        prepend=True,
    )
    db.add_context_category(
        "docling-metadata-and-bibliography",
        macros=[
            MacroSpec("title", "[{"),
            MacroSpec("affiliation", "{"),
            MacroSpec("affil", "{"),
            MacroSpec("address", "{"),
            MacroSpec("institute", "{"),
            MacroSpec("email", "{"),
            MacroSpec("altaffiliation", "[{"),
            MacroSpec("thanks", "{"),
            MacroSpec("inst", "{"),
            MacroSpec("preprint", "{"),
            MacroSpec("parbox", "[{{"),
            MacroSpec("caption", "*[{"),
            MacroSpec("captionof", "{{"),
            MacroSpec("bibitem", "[{"),
            MacroSpec("bibliographystyle", "{"),
            MacroSpec("newtheorem", "*{[{["),
        ],
        environments=[EnvironmentSpec("thebibliography", "{")],
        prepend=True,
    )
    # pylatexenc does not know hyperref's \href[options]{URL}{text}, so its two
    # arguments would stay in the body and be read as text glued together.
    db.add_context_category(
        "docling-links",
        macros=[MacroSpec("href", "[{{")],
        prepend=True,
    )
    return db


LATEX_CONTEXT_DB: Any = _build_context_db()
