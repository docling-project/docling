# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Learned choice of field keys: geometric and text candidates, scored by trees.

``classify`` is an alternative to the integer program of ``assign`` for
values outside detected tables. Every such value gets candidate keys from two
sources, merged by their word sets:

- the geometric candidates of ``candidates_for`` and ``in_checkbox_blocks``,
  with their cost terms, before any global choice;
- text candidates from the page's words (``texts.line_candidates``): pieces
  of lines that the layout keeps whole, stacked lines, row captions and
  column headers.

A tree ensemble (``trees.TreeEnsemble``) scores every candidate of a value
from geometric, box, text and competition features, and the value takes its
best candidate. Values in detected tables keep the keys of ``table_fields``,
as in ``assign``. The result is an ``Assignment``, so the form stage uses it
as it uses the solver's.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from docling_core.types.doc import BoundingBox, DocItemLabel, TableCell
from docling_core.types.doc.page import PdfWidget

from docling.datamodel.base_models import Cluster
from docling.models.stages.form_field.keying.candidates import candidates_for
from docling.models.stages.form_field.keying.geometry import side_of as facing
from docling.models.stages.form_field.keying.inputs import (
    MAX_KEY_TEXTS,
    inputs,
    regions,
)
from docling.models.stages.form_field.keying.solver import (
    MAX_KEY_VALUE_PAIRS,
    NULL_COST,
    abstained,
    in_checkbox_blocks,
)
from docling.models.stages.form_field.keying.symmetry import (
    mark_sequences,
    sequences,
)
from docling.models.stages.form_field.keying.tables import add_hints, table_fields
from docling.models.stages.form_field.keying.texts import (
    KINDS,
    captionish,
    gap,
    key_of,
    line_candidates,
    span_overlap,
)
from docling.models.stages.form_field.keying.trees import TreeEnsemble
from docling.models.stages.form_field.keying.types import (
    Assignment,
    Candidate,
    KeyText,
    Scope,
    Value,
)
from docling.models.stages.form_field.keying.words import Box, PageText, page_text

STATUS = "classified"
# A word centre this close outside a key text's box still belongs to it.
SLACK = 0.5

GEO_TERMS = (
    "cell",
    "misaligned",
    "distance",
    "role",
    "fragment",
    "partial",
    "sequence",
    "alignment",
    "obstruction",
    "group",
)
GEO_KINDS = (
    "field_key",
    "option_key",
    "composite_field",
    "inline_clause",
    "checkbox_block",
)
SIDES = ("inside", "up", "down", "left", "right")
ROLES = (
    "text",
    "section_header",
    "list_item",
    "caption",
    "page_header",
    "page_footer",
    "footnote",
    "title",
    "checkbox_selected",
    "checkbox_unselected",
    "other",
)
VALUE_TYPES = ("text", "checkbox", "signature")

SOURCE_NAMES = ["s_geo", "s_lines"] + [f"s_kind_{k}" for k in KINDS]
GEO_NAMES = (
    [f"g_{t}" for t in GEO_TERMS]
    + [
        "g_cost",
        "g_members",
        "g_active",
        "g_stack",
        "g_rank",
        "g_cost_minus_best",
        "g_n",
    ]
    + [f"g_kind_{k}" for k in GEO_KINDS]
    + [f"g_side_{k}" for k in SIDES]
    + [f"g_role_{k}" for k in ROLES]
)
BOX_NAMES = [
    "b_gap",
    "b_dx",
    "b_dy",
    "b_left",
    "b_right",
    "b_above",
    "b_below",
    "b_xover",
    "b_yover",
    "b_in_value",
    "b_value_in",
    "b_w",
    "b_h",
    "b_lines",
    "b_same_row",
    "b_same_col",
    "b_between_values",
    "b_gap_rank",
    "b_gap_rank_side",
    "b_dx_page",
    "b_dy_page",
]
TEXT_NAMES = [
    "t_words",
    "t_chars",
    "t_colon",
    "t_period",
    "t_question",
    "t_digits",
    "t_alpha",
    "t_upper",
    "t_linenum_first",
    "t_captionish",
    "t_currency",
    "t_capital_first",
]
VALUE_NAMES = ["v_w", "v_h", "v_ncands", "v_ngeo", "v_x", "v_y", "v_order"] + [
    f"v_type_{t}" for t in VALUE_TYPES
]
PAGE_NAMES = ["p_h", "p_values", "p_lines"]
COMPETITION_NAMES = [
    "c_share",
    "c_rank_by_gap",
    "c_nearest_other_gap",
    "c_same_side_closer",
]
FEATURE_NAMES = (
    SOURCE_NAMES
    + GEO_NAMES
    + BOX_NAMES
    + TEXT_NAMES
    + VALUE_NAMES
    + PAGE_NAMES
    + COMPETITION_NAMES
)

LINE_NUMBER = re.compile(
    r"^(\(?[0-9]{1,3}[a-z]?\)?|\(?[a-z]\)|[a-z][.)])$", re.IGNORECASE
)


@dataclass
class Geometric:
    """The cheapest geometric candidate of a word set for one value."""

    candidate: Candidate
    kind: str  # A Candidate kind, or "checkbox_block".
    cost: float  # Per member.
    active: bool  # Cheaper than leaving its members without keys.


@dataclass
class Option:
    """One candidate key of one value: a set of printed words."""

    words: tuple[int, ...]  # Page order.
    box: Box
    text: str
    geometric: Geometric | None = None
    kind: str | None = None  # Text candidate kind (texts.KINDS).


@dataclass
class Page:
    """Everything the classifier reads from one page."""

    values: list[Value]
    key_texts: list[KeyText]
    text: PageText
    size: tuple[float, float]  # Width, height.
    words_of: dict[int, list[int]] = field(default_factory=dict)  # Atom -> its words.
    options: dict[int, list[Option]] = field(default_factory=dict)  # By value position.


def value_type(value: Value) -> str:
    kind = value.native.widget_field_type
    if kind == "/Btn":
        return "checkbox"
    if kind == "/Sig":
        return "signature"
    return "text"


def role_of(label: DocItemLabel | None) -> str:
    if label is None:
        return "other"
    return label.value if label.value in ROLES else "other"


def as_box(bbox: BoundingBox) -> Box:
    return (bbox.l, bbox.t, bbox.r, bbox.b)


def words_in(text: PageText, boxes: Sequence[Box]) -> list[int]:
    """Words whose centre lies in one of the boxes."""
    if not text.words or not boxes:
        return []
    centres = _centres(text)
    inside = np.zeros(len(centres), dtype=bool)
    for b in boxes:
        inside |= (
            (centres[:, 0] >= b[0] - SLACK)
            & (centres[:, 0] <= b[2] + SLACK)
            & (centres[:, 1] >= b[1] - SLACK)
            & (centres[:, 1] <= b[3] + SLACK)
        )
    return np.flatnonzero(inside).tolist()


def _centres(text: PageText) -> np.ndarray:
    return np.array(
        [((w.raw[0] + w.raw[2]) / 2, (w.raw[1] + w.raw[3]) / 2) for w in text.words],
        dtype=np.float64,
    ).reshape(-1, 2)


def atom_words(key_texts: list[KeyText], text: PageText) -> dict[int, list[int]]:
    """The words of each text atom: those inside its own key text's box."""
    return {
        next(iter(k.atoms)): words_in(text, [as_box(k.bbox)])
        for k in key_texts
        if len(k.atoms) == 1
    }


def geometric_candidates(
    values: list[Value],
    key_texts: list[KeyText],
    h: float,
    found: list[Cluster],
    sources,
) -> list[tuple[Candidate, str]]:
    """The geometric candidates with their kinds, before any global choice."""
    proposed = candidates_for(values, key_texts, h)
    mark_sequences(proposed, sequences(values, key_texts, proposed, NULL_COST, h))
    out: list[tuple[Candidate, str]] = [
        (c, c.kind) for c in proposed if c.kind != "choice_group"
    ]
    out += [
        (c, "checkbox_block")
        for c in in_checkbox_blocks(values, key_texts, found, sources)
    ]
    for c, _ in out:
        unknown = set(c.features) - set(GEO_TERMS)
        if unknown:
            raise ValueError(f"cost terms without a feature: {sorted(unknown)}")
    return out


def page_options(
    page: Page,
    geometric: list[tuple[Candidate, str]],
    words_of: dict[int, list[int]],
    wanted: list[int],
) -> None:
    """Merge the geometric and text candidates of every wanted value by word set."""
    text, key_texts = page.text, page.key_texts
    cache: dict[int, tuple[int, ...] | None] = {}

    def key_words(k: int) -> tuple[int, ...] | None:
        if k not in cache:
            kt = key_texts[k]
            if any(atom in words_of for atom in kt.atoms):
                ids = [i for atom in sorted(kt.atoms) for i in words_of.get(atom, [])]
            else:
                ids = words_in(text, [as_box(kt.bbox)])
            found = key_of(text.words, ids)
            cache[k] = None if found is None else found[0]
        return cache[k]

    per_value: dict[int, list[tuple[tuple[int, ...], Geometric]]] = defaultdict(list)
    want = set(wanted)
    for c, kind in geometric:
        words = key_words(c.key)
        if words is None:
            continue
        geo = Geometric(
            c,
            kind,
            c.cost / len(c.members),
            kind == "checkbox_block" or c.cost < NULL_COST * len(c.members),
        )
        for i in c.members:
            if i in want:
                per_value[i].append((words, geo))
    # The layout's multi-line key texts are offered as text candidates too.
    blocks = []
    for k, kt in enumerate(key_texts):
        block = key_words(k) if len(kt.atoms) > 1 else None
        if block is not None:
            blocks.append(block)
    boxes = [as_box(v.bbox) for v in page.values]
    lines = line_candidates(text, boxes, wanted, blocks)
    for i in wanted:
        by: dict[tuple[int, ...], Option] = {}
        for words, geo in per_value[i]:
            key = tuple(sorted(words))
            option = by.get(key)
            if option is None:
                found = key_of(text.words, words)
                assert found is not None
                option = by[key] = Option(found[0], found[1], found[2])
            if option.geometric is None or geo.cost < option.geometric.cost:
                option.geometric = geo
        for kind, words in lines[i]:
            key = tuple(sorted(words))
            option = by.get(key)
            if option is None:
                found = key_of(text.words, words)
                assert found is not None
                option = by[key] = Option(found[0], found[1], found[2])
            if option.kind is None:
                option.kind = kind
        page.options[i] = list(by.values())


def text_features(text: str) -> list[float]:
    words = text.split()
    chars = [c for c in text if not c.isspace()]
    n = max(1, len(chars))
    alpha = sum(c.isalpha() for c in chars)
    return [
        len(words),
        len(chars),
        text.rstrip().endswith(":"),
        text.rstrip().endswith("."),
        "?" in text,
        sum(c.isdigit() for c in chars) / n,
        alpha / n,
        sum(c.isupper() for c in chars) / max(1, alpha),
        bool(words) and bool(LINE_NUMBER.match(words[0])),
        captionish(text),
        any(c in text for c in "$€£%"),
        bool(words) and words[0][:1].isupper(),
    ]


def where(c: Box, b: Box, h: float) -> str:
    """Where a candidate box lies from a value box: inside, left, right, up, down, diag."""
    x = span_overlap(c[0], c[2], b[0], b[2])
    y = span_overlap(c[1], c[3], b[1], b[3])
    if x > 0 and y > 0:
        return "inside"
    if y > 0:
        return "left" if c[2] <= b[0] + 0.5 * h else "right"
    if x > 0:
        return "up" if c[3] <= b[1] + 0.5 * h else "down"
    return "diag"


def features(page: Page) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """One row of FEATURE_NAMES per (value position, option index)."""
    h = page.text.h
    boxes = [as_box(v.bbox) for v in page.values]
    n = len(boxes)
    width, height = page.size
    page_row = [h, n, len(page.text.lines)]
    holders: dict[tuple[int, ...], list[tuple[float, int]]] = defaultdict(list)
    for p, options in page.options.items():
        for o in options:
            holders[tuple(sorted(o.words))].append((gap(o.box, boxes[p]), p))
    rows: list[list[float]] = []
    index: list[tuple[int, int]] = []
    for p, options in page.options.items():
        b = boxes[p]
        bw, bh = b[2] - b[0], b[3] - b[1]
        costs = {
            k: o.geometric.cost
            for k, o in enumerate(options)
            if o.geometric is not None
        }
        ranked = sorted(costs, key=lambda k: costs[k])
        rank = {k: r for r, k in enumerate(ranked)}
        best = costs[ranked[0]] if ranked else math.nan
        gaps = [gap(o.box, b) for o in options]
        sides = [where(o.box, b, h) for o in options]
        by_gap = {
            k: r
            for r, k in enumerate(sorted(range(len(options)), key=lambda k: gaps[k]))
        }
        by_side: dict[int, int] = {}
        for side in set(sides):
            same = sorted(
                (k for k in range(len(options)) if sides[k] == side),
                key=lambda k: gaps[k],
            )
            by_side.update({k: r for r, k in enumerate(same)})
        others = [o for q, o in enumerate(boxes) if q != p]
        kind = value_type(page.values[p])
        value_row = [
            bw / h,
            bh / h,
            len(options),
            len(ranked),
            (b[0] + b[2]) / 2 / width,
            (b[1] + b[3]) / 2 / height,
            p / max(1, n),
        ] + [float(kind == t) for t in VALUE_TYPES]
        for k, o in enumerate(options):
            c = o.box
            g = o.geometric
            source = [float(g is not None), float(o.kind is not None)] + [
                float(o.kind == t) for t in KINDS
            ]
            if g is not None:
                cand = g.candidate
                kt = page.key_texts[cand.key]
                geo = (
                    [cand.features.get(t, 0.0) for t in GEO_TERMS]
                    + [
                        g.cost,
                        len(cand.members),
                        float(g.active),
                        float(kt.stack),
                        rank[k],
                        g.cost - best,
                        len(ranked),
                    ]
                    + [float(g.kind == t) for t in GEO_KINDS]
                    + [float(cand.side == t) for t in SIDES]
                    + [float(role_of(kt.layout_label) == t) for t in ROLES]
                )
            else:
                geo = [math.nan] * len(GEO_NAMES)
            x = span_overlap(c[0], c[2], b[0], b[2])
            y = span_overlap(c[1], c[3], b[1], b[3])
            cw, ch = max(1e-6, c[2] - c[0]), max(1e-6, c[3] - c[1])
            inter = x * y
            env = (min(c[0], b[0]), min(c[1], b[1]), max(c[2], b[2]), max(c[3], b[3]))
            between = sum(
                1
                for o2 in others
                if o2[0] >= env[0] - 1
                and o2[2] <= env[2] + 1
                and o2[1] >= env[1] - 1
                and o2[3] <= env[3] + 1
            )
            box = [
                gaps[k] / h,
                ((c[0] + c[2]) - (b[0] + b[2])) / 2 / h,
                ((c[1] + c[3]) - (b[1] + b[3])) / 2 / h,
                (b[0] - c[2]) / h,
                (c[0] - b[2]) / h,
                (b[1] - c[3]) / h,
                (c[1] - b[3]) / h,
                x / max(1e-6, min(cw, bw)),
                y / max(1e-6, min(ch, bh)),
                inter / (cw * ch),
                inter / max(1e-6, bw * bh),
                cw / h,
                ch / h,
                ch / h / 1.2,
                float(sides[k] in ("left", "right")),
                float(sides[k] in ("up", "down")),
                between,
                by_gap[k],
                by_side[k],
                ((c[0] + c[2]) - (b[0] + b[2])) / 2 / width,
                ((c[1] + c[3]) - (b[1] + b[3])) / 2 / height,
            ]
            held = holders[tuple(sorted(o.words))]
            mine = gaps[k]
            other = [d for d, q in held if q != p]
            closer = sum(
                1
                for d, q in held
                if q != p and d < mine and where(c, boxes[q], h) == sides[k]
            )
            competition = [
                len(held),
                sum(1 for d, _ in held if d < mine),
                (min(other) / h) if other else math.nan,
                closer,
            ]
            rows.append(
                source
                + geo
                + box
                + text_features(o.text)
                + value_row
                + page_row
                + competition
            )
            index.append((p, k))
    x_all = np.asarray(rows, dtype=np.float64).reshape(-1, len(FEATURE_NAMES))
    return x_all, index


@dataclass
class Prepared:
    """A page ready for the classifier, or the reason it abstains."""

    page: Page | None
    assignment: Assignment  # Table keys; free-form keys are added by classify.
    found: list[Cluster]
    geometric: list[tuple[Candidate, str]]


def prepare(
    widgets: list[PdfWidget],
    clusters: list[Cluster],
    table_cells: dict[int, list[TableCell]],
    page_size: tuple[float, float],
    rules: list[BoundingBox] | None,
    words: Sequence[tuple[str, Box, bool]],
    lines: Sequence[Box],
) -> Prepared:
    """Inputs, table keys and candidates of one page (``classify`` without the trees).

    ``words`` are the page's word cells as (text, top-left box, painted by a
    widget) and ``lines`` its text line boxes (see ``words.page_text``).
    """
    width, height = page_size
    found = regions(clusters)
    read = inputs(widgets, found, table_cells, height)
    values, key_texts = read.values, read.key_texts
    too_big = (
        len(key_texts) > MAX_KEY_TEXTS
        or len(values) * len(key_texts) > MAX_KEY_VALUE_PAIRS
    )
    geometric = (
        []
        if too_big
        else geometric_candidates(values, key_texts, read.h, found, read.sources)
    )
    # Appends the cell keys to key_texts, after the free-form candidates.
    tables, slots = table_fields(
        found, table_cells, rules or [], height, values, key_texts
    )
    if too_big:
        status = f"skipped: {len(values)} values x {len(key_texts)} key texts exceed the page limit"
        assignment = abstained(values, key_texts, tables, status, slots)
        assignment.sources, assignment.painted_cells = read.sources, read.painted_cells
        return Prepared(None, assignment, found, geometric)
    assignment = Assignment(
        values,
        key_texts,
        list(tables),
        list(range(len(tables))),
        STATUS,
        slots,
        read.sources,
        read.painted_cells,
    )
    text = page_text(words, lines)
    page = Page(values, key_texts, text, (width, height), atom_words(key_texts, text))
    wanted = [i for i, v in enumerate(values) if v.scope == Scope()]
    page_options(page, geometric, page.words_of, wanted)
    return Prepared(page, assignment, found, geometric)


def choose(
    probabilities: np.ndarray, index: list[tuple[int, int]], no_key: float
) -> dict[int, tuple[int, float]]:
    """Per value position: its most probable option, unless below ``no_key``."""
    best: dict[int, tuple[int, float]] = {}
    for (p, k), probability in zip(index, probabilities):
        if p not in best or probability > best[p][1]:
            best[p] = (k, float(probability))
    return {p: kp for p, kp in best.items() if kp[1] >= no_key}


def _key_text(page: Page, option: Option, atoms: frozenset[int]) -> KeyText:
    """The chosen words as a key text: trimmed text and box, covered atoms."""
    g = option.geometric
    label = None if g is None else page.key_texts[g.candidate.key].layout_label
    stack = False if g is None else page.key_texts[g.candidate.key].stack
    return KeyText(
        option.text,
        BoundingBox(l=option.box[0], t=option.box[1], r=option.box[2], b=option.box[3]),
        atoms,
        Scope(),
        label,
        fragment=False,
        stack=stack,
    )


def assemble(prepared: Prepared, chosen: dict[int, tuple[int, float]]) -> Assignment:
    """The chosen options as candidates of the page's Assignment.

    Values that chose the words of one multi-value geometric candidate
    (a composite field, an inline clause) form that candidate again;
    every other value gets a single-value candidate. A text atom becomes part
    of a key, and so leaves the body, only when the chosen keys use all of
    its words: a line partly used as a key stays in the body.
    """
    page = prepared.page
    assignment = prepared.assignment
    assert page is not None
    words_of = page.words_of
    picked = {p: page.options[p][k] for p, (k, _) in chosen.items()}
    used = {i for o in picked.values() for i in o.words}
    covered = {
        atom
        for atom, ids in words_of.items()
        if any(not page.text.words[i].fill for i in ids)
        and all(i in used or page.text.words[i].fill for i in ids)
    }
    atoms_of: dict[tuple[int, ...], frozenset[int]] = {}

    def atoms(option: Option) -> frozenset[int]:
        if option.words not in atoms_of:
            mine = set(option.words)
            atoms_of[option.words] = frozenset(
                a for a in covered if mine.intersection(words_of[a])
            )
        return atoms_of[option.words]

    groups: list[tuple[tuple[int, ...], Option, Candidate | None]] = []
    done: set[int] = set()
    for p in sorted(picked):
        g = picked[p].geometric
        if p in done or g is None or len(g.candidate.members) < 2:
            continue
        members = g.candidate.members
        if all(q in picked and picked[q].words == picked[p].words for q in members):
            groups.append((members, picked[p], g.candidate))
            done.update(members)
    for p in sorted(picked):
        if p not in done:
            groups.append(((p,), picked[p], None))
    key_texts = assignment.key_texts
    shadows: list[Candidate] = []
    out: list[Candidate] = []
    for members, option, geometric in groups:
        g = option.geometric
        value = page.values[members[0]]
        if geometric is not None:
            kind, side = geometric.kind, geometric.side
        else:
            kind = "option_key" if value.checkbox else "field_key"
            side = None
            if g is not None and len(g.candidate.members) == 1:
                side = g.candidate.side
            else:
                seen = facing(
                    BoundingBox(
                        l=option.box[0],
                        t=option.box[1],
                        r=option.box[2],
                        b=option.box[3],
                    ),
                    value.bbox,
                )
                side = None if seen is None else seen[0]
        key_texts.append(_key_text(page, option, atoms(option)))
        out.append(Candidate(members, len(key_texts) - 1, kind, {}, side))
        # add_hints compares key texts by index: a shadow carries the index
        # of the same geometric key text, so it is never its own hint.
        shadow_key = g.candidate.key if g is not None else len(key_texts) - 1
        shadows.append(Candidate(members, shadow_key, kind, {}, side))
    add_hints(shadows, [c for c, _ in prepared.geometric], page.values, NULL_COST)
    for c, shadow in zip(out, shadows):
        c.hint = shadow.hint
    start = len(assignment.candidates)
    assignment.candidates += out
    assignment.selected += list(range(start, start + len(out)))
    return assignment


def classify(
    widgets: list[PdfWidget],
    clusters: list[Cluster],
    table_cells: dict[int, list[TableCell]],
    page_size: tuple[float, float],
    rules: list[BoundingBox] | None,
    words: Sequence[tuple[str, Box, bool]],
    lines: Sequence[Box],
    model: TreeEnsemble,
) -> Assignment:
    """Choose the key of every widget value on one page with the tree ensemble.

    The arguments are those of ``assign`` (with the page width), plus the
    page's word cells and text lines and the fitted trees. Nothing passed
    in is modified.
    """
    if tuple(model.names) != tuple(FEATURE_NAMES):
        raise ValueError("the tree ensemble was fitted on other features")
    prepared = prepare(widgets, clusters, table_cells, page_size, rules, words, lines)
    if prepared.page is None:
        return prepared.assignment
    x, index = features(prepared.page)
    chosen = choose(model.predict(x), index, model.no_key)
    return assemble(prepared, chosen)
