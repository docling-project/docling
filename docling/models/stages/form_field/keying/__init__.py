# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Geometric keying of native AcroForm widgets to their printed captions.

``assign`` takes a page ``Snapshot`` (widgets, layout regions, detected table
cells) and returns an ``Assignment``: for every value, the caption that keys it,
chosen by a small integer program over geometric candidates. Values inside
detected tables are keyed from the table's own cells. The module is pure: it
reads no files and keeps no state, and it is what the offline replay
(``scripts/replay_acroform_keying.py``) evaluates against the reviewed
ground truth.
"""

from docling.models.stages.form_field.keying.candidates import candidates_for
from docling.models.stages.form_field.keying.geometry import anchors, overlap
from docling.models.stages.form_field.keying.inputs import inputs, regions, scope_of
from docling.models.stages.form_field.keying.solver import assign
from docling.models.stages.form_field.keying.types import (
    INLINE_WIDGET_COVERAGE,
    Assignment,
    Candidate,
    DetectedTable,
    Label,
    NativeWidget,
    Region,
    Scope,
    Side,
    Snapshot,
    Tables,
    Value,
)

__all__ = [
    "INLINE_WIDGET_COVERAGE",
    "Assignment",
    "Candidate",
    "DetectedTable",
    "Label",
    "NativeWidget",
    "Region",
    "Scope",
    "Side",
    "Snapshot",
    "Tables",
    "Value",
    "anchors",
    "assign",
    "candidates_for",
    "inputs",
    "overlap",
    "regions",
    "scope_of",
]
