# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Helpers for the structured mathematical representation carried on a formula item."""

from __future__ import annotations

from docling_core.types.doc import NodeItem
from docling_core.types.doc.document import FormulaItem


def has_native_mathml(element: NodeItem) -> bool:
    """True when *element* already carries MathML read from the source document.

    Used by the formula understanding stages to skip an equation whose authored
    representation is already known, and by tests to assert that skip.
    """
    if not isinstance(element, FormulaItem) or element.meta is None:
        return False
    return element.meta.formula is not None and bool(element.meta.formula.mathml)
