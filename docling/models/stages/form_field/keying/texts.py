# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Text candidates of a value: pieces of printed lines near it.

The layout gives key texts as whole text lines and blocks. These candidates
come from the words instead: lines cut at fill runs, at a value between two
words and at wide gaps; whole lines; leading sentences; stacks of up to three
lines; the captions along the value's row and the header above it in its
column, at any distance. The classifier chooses among them and the
geometric candidates.
"""

from __future__ import annotations

from collections.abc import Sequence

from docling.models.stages.form_field.keying.words import Box, PageText, Word

# Text heights: line candidates farther from the value are dropped.
RADIUS = 8.0
# Line candidates kept per value, nearest first.
MAX_LINE_CANDIDATES = 40
# Text heights: a wider horizontal gap between two words cuts a piece.
GAP_SPLIT = 1.5
# Text heights: vertical gap between the stacked lines of one key.
STACK_GAP = 0.8
MAX_STACK = 3
# Words ending a leading sentence inside a piece.
SENTENCE_END = (".", ":", "?")
# Text heights: farther apart, two pieces of a row are not one caption.
ROW_REACH = 4.0
# Caption pieces joined at most, walking away from the value along its row.
ROW_CAPTIONS = 3
# Caption pieces taken above a value in its column, nearest first.
HEADERS_ABOVE = 4

KINDS = ("segment", "bridged", "prefix", "line", "stack", "block", "row")


def gap(a: Box, b: Box) -> float:
    """Manhattan gap between two boxes, 0 when they overlap (geometry.gap)."""
    return max(0.0, a[0] - b[2], b[0] - a[2]) + max(0.0, a[1] - b[3], b[1] - a[3])


def span_overlap(lo: float, hi: float, other_lo: float, other_hi: float) -> float:
    return max(0.0, min(hi, other_hi) - max(lo, other_lo))


def union(boxes: Sequence[Box]) -> Box:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def captionish(text: str) -> bool:
    """Text that can be a caption: a word with two letters or more.

    Line numbers such as "16b", dot leaders and currency signs are not.
    """
    return any(sum(c.isalpha() for c in t) >= 2 for t in text.split())


def key_of(
    words: Sequence[Word], ids: Sequence[int]
) -> tuple[tuple[int, ...], Box, str] | None:
    """Word ids (fill words out, page order), box and text of a word set."""
    keep = [i for i in dict.fromkeys(ids) if not words[i].fill]
    if not keep:
        return None
    keep.sort(
        key=lambda i: (
            words[i].line if words[i].line is not None else 1 << 30,
            words[i].raw[0],
        )
    )
    box = union([words[i].box for i in keep])
    return (
        tuple(keep),
        (round(box[0], 3), round(box[1], 3), round(box[2], 3), round(box[3], 3)),
        " ".join(words[i].text for i in keep),
    )


def cut_line(
    ids: list[int], words: Sequence[Word], boxes: Sequence[Box], h: float, at_fill: bool
) -> list[list[int]]:
    """One text line's word ids cut into pieces.

    Cuts where a value lies between two words, at gaps wider than GAP_SPLIT
    text heights and, with ``at_fill``, at fill words. Without it, a fill
    run bridges the gap but stays out of the key.
    """
    pieces: list[list[int]] = []
    current: list[int] = []
    last: Box | None = None
    for i in ids:
        word = words[i]
        if word.fill and at_fill:
            if current:
                pieces.append(current)
            current, last = [], None
            continue
        box = word.box
        if last is not None:
            lo, hi = last[2], box[0]
            cut = hi - lo > GAP_SPLIT * h
            if not cut and hi > lo:
                top, bottom = min(last[1], box[1]), max(last[3], box[3])
                cut = any(
                    b[0] < hi
                    and b[2] > lo
                    and span_overlap(top, bottom, b[1], b[3]) > 0
                    for b in boxes
                )
            if cut:
                pieces.append(current)
                current = []
        last = box
        if not word.fill:
            current.append(i)
    if current:
        pieces.append(current)
    return [p for p in pieces if p]


def prefixes(ids: list[int], words: Sequence[Word], most: int = 3) -> list[list[int]]:
    """Leading spans of a piece ending at a word that ends in . : or ?

    Shortest first: a sentence or caption that ends inside a long line.
    """
    out = []
    for k, i in enumerate(ids[:-1]):
        if words[i].text.rstrip().endswith(SENTENCE_END) and any(
            c.isalpha() for c in words[i].text
        ):
            out.append(ids[: k + 1])
            if len(out) >= most:
                break
    return out


def pieces_of(
    text: PageText, boxes: Sequence[Box], h: float
) -> list[tuple[str, list[int]]]:
    """Text pieces of the page as (kind, word ids).

    Line pieces cut at fill words ("segment") or bridging them ("bridged"),
    their leading sentences ("prefix") and whole lines ("line").
    """
    words = text.words
    out = []
    for ids in text.lines:
        seen: set[tuple[int, ...]] = set()
        for kind, cut in (
            ("segment", cut_line(ids, words, boxes, h, at_fill=True)),
            ("bridged", cut_line(ids, words, boxes, h, at_fill=False)),
        ):
            for piece in cut:
                if tuple(piece) not in seen:
                    seen.add(tuple(piece))
                    out.append((kind, piece))
                for lead in prefixes(piece, words):
                    if tuple(lead) not in seen:
                        seen.add(tuple(lead))
                        out.append(("prefix", lead))
        whole = [i for i in ids if not words[i].fill]
        if whole and tuple(whole) not in seen:
            out.append(("line", whole))
    return out


def stacks(pieces: list[tuple[tuple[int, ...], Box]], h: float) -> list[list[int]]:
    """Word sets of 2 to MAX_STACK vertically adjacent pieces, top to bottom.

    Left edges or centres aligned, and a vertical gap under STACK_GAP text
    heights.
    """
    order = sorted(range(len(pieces)), key=lambda k: pieces[k][1][1])
    below: dict[int, list[int]] = {}
    for x, k in enumerate(order):
        b = pieces[k][1]
        for k2 in order[x + 1 :]:
            c = pieces[k2][1]
            if c[1] - b[3] > STACK_GAP * h:
                break
            if c[1] <= b[1] + 0.5 * h:  # Same row.
                continue
            aligned = (
                abs(c[0] - b[0]) <= h or abs((c[0] + c[2]) - (b[0] + b[2])) <= 2 * h
            )
            if aligned and span_overlap(b[0], b[2], c[0], c[2]) > 0:
                below.setdefault(k, []).append(k2)
    out = []
    for k in range(len(pieces)):
        chains = [[k]]
        for _ in range(MAX_STACK - 1):
            longer = [
                [*chain, k2] for chain in chains for k2 in below.get(chain[-1], [])
            ]
            out.extend(longer)
            chains = longer
            if not chains:
                break
    return [[i for k in chain for i in pieces[k][0]] for chain in out]


# A text candidate: kind, word ids, box, text.
Piece = tuple[str, tuple[int, ...], Box, str]


def row_runs(
    b: Box, pieces: list[Piece], boxes: Sequence[Box], h: float, side: str
) -> list[list[int]]:
    """Captions along the value's row on one side ("left" or "right").

    Walking away from the value over the row's pieces (dot leaders, line
    numbers and signs are bridged), each caption piece met gives two
    candidates: the run from it to the value, and the caption pieces alone.
    The walk stops at another value, at a gap over ROW_REACH text heights
    once a caption was met, or after ROW_CAPTIONS caption pieces.
    """
    row = []
    for k, (kind, _, bb, _) in enumerate(pieces):
        if kind not in ("segment", "line"):
            continue
        if span_overlap(bb[1], bb[3], b[1], b[3]) < 0.5 * min(
            bb[3] - bb[1], b[3] - b[1]
        ):
            continue
        if side == "left" and bb[2] <= b[0] + 0.5 * h:
            row.append((b[0] - bb[2], k))
        elif side == "right" and bb[0] >= b[2] - 0.5 * h:
            row.append((bb[0] - b[2], k))
    row.sort()
    others = [
        o for o in boxes if o is not b and span_overlap(o[1], o[3], b[1], b[3]) > 0
    ]
    out: list[list[int]] = []
    run: list[int] = []
    captions: list[int] = []
    edge: float | None = None
    for _, k in row:
        bb = pieces[k][2]
        if edge is not None:
            far = (edge - bb[2]) if side == "left" else (bb[0] - edge)
            if captions and far > ROW_REACH * h:
                break
            lo, hi = (bb[2], edge) if side == "left" else (edge, bb[0])
            if any(o[0] < hi and o[2] > lo for o in others):
                break
        if run and any(set(pieces[k][1]) & set(pieces[j][1]) for j in run):
            continue  # A line holding pieces already taken.
        run.append(k)
        edge = bb[0] if side == "left" else bb[2]
        if captionish(pieces[k][3]):
            captions.append(k)
            out.append([i for j in run for i in pieces[j][1]])
            if len(captions) > 1 or len(run) > 1:
                out.append([i for j in captions for i in pieces[j][1]])
            if len(captions) >= ROW_CAPTIONS:
                break
    return out


def column_above(b: Box, pieces: list[Piece], h: float) -> list[int]:
    """The HEADERS_ABOVE nearest caption pieces above the value in its column.

    At any distance (a column header), with the stacks ending at each.
    Returns indices into ``pieces``.
    """
    above = []
    for k, (kind, _, bb, text) in enumerate(pieces):
        if kind not in ("segment", "bridged", "line") or not captionish(text):
            continue
        if bb[3] <= b[1] + 0.5 * h and span_overlap(
            bb[0], bb[2], b[0], b[2]
        ) >= 0.3 * min(bb[2] - bb[0], b[2] - b[0]):
            above.append((b[1] - bb[3], k))
    above.sort()
    out = []
    for _, k in above[:HEADERS_ABOVE]:
        top = pieces[k][2]
        out.append(k)
        out += [
            j
            for j, (kind, _, bb, _) in enumerate(pieces)
            if kind == "stack"
            and abs(bb[3] - top[3]) < 0.5
            and bb[0] <= top[0] + h
            and bb[2] >= top[2] - h
        ]
    return out


def line_candidates(
    text: PageText,
    boxes: Sequence[Box],
    wanted: Sequence[int],
    blocks: Sequence[Sequence[int]] = (),
) -> dict[int, list[tuple[str, tuple[int, ...]]]]:
    """Text candidates of the values at positions ``wanted``, as (kind, word ids).

    Per value: the pieces within RADIUS text heights, nearest first and at
    most MAX_LINE_CANDIDATES, then the captions along its row and the header
    above it at any distance. ``boxes`` are every value of the page, which
    cut lines and stop row walks; ``blocks`` are more word sets to offer
    (the layout's multi-line key texts).
    """
    words, h = text.words, text.h
    pieces: list[Piece] = []
    index: dict[tuple[int, ...], int] = {}

    def add(kind: str, ids: Sequence[int]) -> int | None:
        found = key_of(words, ids)
        if found is None:
            return None
        if found[0] in index:
            return index[found[0]]
        index[found[0]] = len(pieces)
        pieces.append((kind, *found))
        return index[found[0]]

    for kind, ids in pieces_of(text, boxes, h):
        add(kind, ids)
    cut = [(ids, bb) for kind, ids, bb, _ in pieces if kind in ("segment", "bridged")]
    for ids in stacks(cut, h):
        add("stack", ids)
    for ids in blocks:
        add("block", ids)
    near_all = [(k, bb) for k, (_, _, bb, _) in enumerate(pieces)]
    out = {}
    for p in wanted:
        b = boxes[p]
        near = sorted((gap(bb, b), k) for k, bb in near_all if gap(bb, b) <= RADIUS * h)
        keep = [k for _, k in near[:MAX_LINE_CANDIDATES]]
        far = column_above(b, pieces, h)
        for side in ("left", "right"):
            for ids in row_runs(b, pieces, boxes, h, side):
                k = add("row", ids)
                if k is not None:
                    far.append(k)
        for k in far:
            if k not in keep:
                keep.append(k)
        out[p] = [(pieces[k][0], pieces[k][1]) for k in keep]
    return out
