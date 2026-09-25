# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The pylatexenc parsing context shared by every LatexWalker the backend builds."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - import-time guard
    from pylatexenc.latexwalker import get_default_latex_context_db
    from pylatexenc.macrospec import EnvironmentSpec

    _PYLATEXENC_AVAILABLE = True
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
        prepend=True,
    )
    return db


LATEX_CONTEXT_DB: Any = _build_context_db()
