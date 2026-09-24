# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Input contract and internal data of the AcroForm keying.

The snapshot models mirror the frozen replay inputs (native widgets, layout
regions and detected table cells of one page), so the same code keys a live
page and a saved one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from docling_core.types.doc import BoundingBox, Size, TableCell
from docling_core.types.doc.page import BoundingRectangle, TextCell
from pydantic import BaseModel, Field

# Match the existing form stage's text-container coverage threshold. This
# tolerates imperfect text bounds only for inline-clause proposals; detected
# table/cell ownership still requires complete containment in scope_of().
INLINE_WIDGET_COVERAGE = 0.8

Side = Literal["inside", "up", "down", "left", "right"]


class NativeWidget(BaseModel):
    """Explicit snapshot contract, independent of installed parser bindings."""

    index: int
    rect: BoundingRectangle
    widget_text: str | None = None
    widget_description: str | None = None
    widget_field_name: str | None = None
    widget_field_type: str | None = None
    widget_field_flags: int
    widget_appearance_state: str | None


class Region(BaseModel):
    id: int
    label: str
    bbox: BoundingBox
    cells: list[TextCell] = Field(default_factory=list)
    children: list[Region] = Field(default_factory=list)


class DetectedTable(BaseModel):
    table_cells: list[TableCell]


class Tables(BaseModel):
    table_map: dict[int, DetectedTable] = Field(default_factory=dict)


class Snapshot(BaseModel):
    page: int
    size: Size
    widgets: list[NativeWidget]
    layout: list[Region]
    tables: Tables


@dataclass(frozen=True)
class Scope:
    table: int | None = None
    cell: int | None = None

    @property
    def eligible(self) -> bool:
        return self.table is None or self.cell is not None


@dataclass
class Value:
    native: NativeWidget
    bbox: BoundingBox
    scope: Scope

    @property
    def checkbox(self) -> bool:
        return self.native.widget_field_type == "/Btn"


@dataclass
class Label:
    text: str
    bbox: BoundingBox
    atoms: frozenset[int]
    scope: Scope
    role: str
    fragment: bool = False
    stack: bool = False  # Joined from vertically adjacent layout blocks.


@dataclass
class Candidate:
    members: tuple[int, ...]  # Positions in the supplied value list, not y order.
    label: int
    kind: Literal[
        "field_key",
        "option_caption",
        "composite_field",
        "inline_clause",
        "choice_group",
        "table_cell",
    ]
    features: dict[str, float]
    side: Side | None = None  # For single-value captions: where the label faces it.
    context: int | None = None  # A secondary label, e.g. a table column header.

    @property
    def cost(self) -> float:
        return sum(self.features.values())


@dataclass
class Assignment:
    values: list[Value]
    labels: list[Label]
    candidates: list[Candidate]
    selected: list[int]
    solver_status: str
    objective: float
