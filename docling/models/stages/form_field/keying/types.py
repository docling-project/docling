# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Internal data of the AcroForm keying."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from docling_core.types.doc import BoundingBox, DocItemLabel
from docling_core.types.doc.page import PdfWidget

# How much of a widget a text container or FORM region must cover to hold it.
# This tolerates imperfect text bounds; detected table/cell ownership still
# requires complete containment in scope_of().
WIDGET_COVERAGE = 0.8
# Field flag bit 17 of a /Btn widget (PDF 32000-1, table 226): a push button.
PUSHBUTTON_FLAG = 1 << 16

Side = Literal["inside", "up", "down", "left", "right"]


def is_skipped(widget: PdfWidget, bbox: BoundingBox) -> bool:
    """Widgets that carry no field value for the document.

    A widget of zero height or width is an artifact (Well-Tagged PDF 1.0,
    8.9.2.4.13). Push buttons trigger actions and hold no value.
    """
    if bbox.width <= 0 or bbox.height <= 0:
        return True
    return widget.widget_field_type == "/Btn" and bool(
        widget.widget_field_flags & PUSHBUTTON_FLAG
    )


@dataclass(frozen=True)
class Scope:
    table: int | None = None
    cell: int | None = None

    @property
    def eligible(self) -> bool:
        return self.table is None or self.cell is not None


@dataclass
class Value:
    native: PdfWidget
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
    role: DocItemLabel | None  # Layout label of the source block; None for table cells.
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


@dataclass(frozen=True)
class TableSlot:
    """The grid cell of a detected table that a value sits in."""

    table: int  # Layout id of the table region.
    rows: tuple[int, int]  # Start and end row offsets of the cell.
    columns: tuple[int, int]  # Start and end column offsets of the cell.
    key: int | None = None  # Label of lettered text in the same cell, if any.


@dataclass
class Assignment:
    values: list[Value]
    labels: list[Label]
    candidates: list[Candidate]
    selected: list[int]
    solver_status: str
    # Positions in the value list -> their cell, for values in detected tables.
    slots: dict[int, TableSlot] = field(default_factory=dict)
    # Text atom -> the (cluster id, cell index) pairs it was read from; more
    # than one cluster when the layout put the cell in two clusters.
    sources: dict[int, set[tuple[int, int]]] = field(default_factory=dict)
