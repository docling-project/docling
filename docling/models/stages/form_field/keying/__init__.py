# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Geometric keying of native AcroForm widgets to their printed captions.

``assign`` takes one page's widgets, layout clusters and detected table cells
and returns an ``Assignment``: for every value, the caption that keys it,
chosen by a small integer program over geometric candidates. A value inside a
detected table is keyed only by text of its own cell. The module reads no
files, keeps no state and does not modify its inputs.
"""

from docling.models.stages.form_field.keying.inputs import regions, scope_of
from docling.models.stages.form_field.keying.solver import assign
from docling.models.stages.form_field.keying.types import (
    PUSHBUTTON_FLAG,
    WIDGET_COVERAGE,
    Assignment,
    Candidate,
    Label,
    Scope,
    is_skipped,
)

__all__ = [
    "PUSHBUTTON_FLAG",
    "WIDGET_COVERAGE",
    "Assignment",
    "Candidate",
    "Label",
    "Scope",
    "assign",
    "is_skipped",
    "regions",
    "scope_of",
]
