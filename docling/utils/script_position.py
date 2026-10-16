# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Find superscript and subscript text from the size and baseline of PDF characters.

A character is a superscript or subscript when it is drawn smaller than the body text of
its line and raised or lowered against the body's baseline, within the line.
"""

from __future__ import annotations

import difflib
import itertools
import statistics
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from docling_core.types.doc.document import Script

# A script is drawn at most this share of the body size...
MAX_SCRIPT_SIZE = 0.85
# ...and shifted by at least this share of it. Word lowers a subscript by under a tenth.
MIN_SHIFT = 0.05
MIN_SHIFT_POINTS = 0.4
# Baselines this close (share of the largest size) are the same baseline.
BASELINE_TOLERANCE = 0.04
BASELINE_TOLERANCE_POINTS = 0.25
# Baseline groups at least this share of the largest letter size can be the body.
BODY_SIZE_RATIO = 0.9
# Largest distance (share of the body size) between a script and the text it marks.
TOUCH_GAP = 0.6
# Letters this close to body letters on both sides sit inside a word ("LaTeX").
IN_WORD_GAP = 0.15
# A new line starts where the baseline moves by this share of the line's size, or where
# the pen returns to the line's start (within this share of the size; a narrow base
# letter, as under a stacked script, is wider).
LINE_JUMP = 0.85
LINE_START = 0.15
# Marks raised as footnote references with no letter or digit.
FOOTNOTE_MARKS = frozenset("*†‡§¶‖#")
# Share of an item's characters that must be found among the PDF characters.
MIN_ALIGNED = 0.8
# Matching is quadratic, so longer items are skipped.
MAX_ALIGN_CHARS = 5000

_PUNCTUATION_FOLD = str.maketrans(
    {
        "\N{LEFT SINGLE QUOTATION MARK}": "'",
        "\N{RIGHT SINGLE QUOTATION MARK}": "'",
        "\N{SINGLE LOW-9 QUOTATION MARK}": "'",
        "\N{PRIME}": "'",
        "\N{LEFT DOUBLE QUOTATION MARK}": '"',
        "\N{RIGHT DOUBLE QUOTATION MARK}": '"',
        "\N{DOUBLE LOW-9 QUOTATION MARK}": '"',
        "\N{DOUBLE PRIME}": '"',
        "\N{HYPHEN}": "-",
        "\N{NON-BREAKING HYPHEN}": "-",
        "\N{FIGURE DASH}": "-",
        "\N{EN DASH}": "-",
        "\N{EM DASH}": "-",
        "\N{MINUS SIGN}": "-",
    }
)


class ScriptChar(Protocol):
    """A character as detection needs it, see ``pdf_char_geometry.PdfChar``."""

    @property
    def text(self) -> str: ...
    @property
    def left(self) -> float: ...
    @property
    def right(self) -> float: ...
    @property
    def baseline(self) -> float: ...
    @property
    def size(self) -> float: ...
    @property
    def horizontal(self) -> bool: ...


@dataclass(frozen=True)
class ScriptRun:
    """A piece of a text item's text at one script position."""

    text: str
    script: Script
    # Indexes of the PDF characters matched to the run.
    chars: tuple[int, ...]


def script_positions(chars: Sequence[ScriptChar]) -> list[Script]:
    """Return the script position of each character, in content order."""
    positions = [Script.BASELINE] * len(chars)
    for line in _split_lines(chars):
        for index, script in _line_scripts(chars, line).items():
            positions[index] = script
    return positions


def script_runs(text: str, chars: Sequence[ScriptChar]) -> list[ScriptRun] | None:
    """Split an item's text into baseline, superscript and subscript runs.

    The text is matched to the PDF characters first, since it differs from them in
    ligatures, dashes and spaces, and only the matched characters are measured. As in the
    Word backend, a run has no whitespace at its edges. Returns None when the text has no
    script or cannot be matched.
    """
    mapping = _align(text, chars)
    if mapping is None:
        return None
    matched = sorted(set(mapping.values()))
    positions = dict(
        zip(matched, script_positions([chars[i] for i in matched]), strict=True)
    )

    roles: list[Script | None] = [None] * len(text)
    for index, char_index in mapping.items():
        roles[index] = positions[char_index]
    if all(r in (None, Script.BASELINE) for r in roles):
        return None

    solid = [i for i, ch in enumerate(text) if not ch.isspace()]
    # Unmatched characters take the position of matched neighbours that agree.
    for k, index in enumerate(solid):
        if roles[index] is not None:
            continue
        before = next(
            (roles[j] for j in reversed(solid[:k]) if roles[j] is not None), None
        )
        after = next((roles[j] for j in solid[k + 1 :] if roles[j] is not None), None)
        roles[index] = before if before == after and before else Script.BASELINE
    # Whitespace inside a run belongs to it; whitespace between runs is left out.
    for left, right in itertools.pairwise(solid):
        if roles[left] == roles[right]:
            for i in range(left + 1, right):
                roles[i] = roles[left]

    pieces: list[tuple[str, Script, list[int]]] = []
    for i, ch in enumerate(text):
        role = roles[i]
        if role is None:
            continue
        if not pieces or pieces[-1][1] != role:
            pieces.append(("", role, []))
        piece_text, _, piece_chars = pieces[-1]
        if i in mapping and mapping[i] not in piece_chars:
            piece_chars.append(mapping[i])
        pieces[-1] = (piece_text + ch, role, piece_chars)
    return [ScriptRun(t, role, tuple(c)) for t, role, c in pieces]


def _measurable(char: ScriptChar) -> bool:
    return char.horizontal and char.size > 0 and not char.text.isspace()


def _split_lines(chars: Sequence[ScriptChar]) -> list[list[int]]:
    """Split the measurable characters into visual lines, as lists of indexes.

    A line ends where the baseline moves by most of a line height, or where the pen returns
    to the line's start. A script stacked over another (x with i below, 2 above) does
    neither, however wide it is.
    """
    lines: list[list[int]] = []
    line: list[int] = []
    line_size = 0.0
    for index, char in enumerate(chars):
        if not _measurable(char):
            continue
        if line:
            size = max(char.size, line_size)
            previous = chars[line[-1]]
            returns_to_start = (
                char.left < previous.left
                and char.left <= chars[line[0]].left + LINE_START * size
            )
            if (
                abs(char.baseline - previous.baseline) > LINE_JUMP * size
                or returns_to_start
            ):
                lines.append(line)
                line, line_size = [], 0.0
        line.append(index)
        line_size = max(line_size, char.size)
    if line:
        lines.append(line)
    return lines


def _baseline_groups(chars: Sequence[ScriptChar], line: list[int]) -> list[list[int]]:
    largest = max(chars[i].size for i in line)
    tolerance = max(BASELINE_TOLERANCE_POINTS, BASELINE_TOLERANCE * largest)
    groups: list[list[int]] = []
    for index in sorted(line, key=lambda i: chars[i].baseline):
        if (
            groups
            and abs(
                chars[index].baseline
                - statistics.median(chars[i].baseline for i in groups[-1])
            )
            <= tolerance
        ):
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def _letter_size(chars: Sequence[ScriptChar], group: list[int]) -> float:
    """Return the largest size of the group's letters and digits, 0 if it has none."""
    return max((chars[i].size for i in group if chars[i].text.isalnum()), default=0.0)


def _line_scripts(chars: Sequence[ScriptChar], line: list[int]) -> dict[int, Script]:
    """Find the superscript and subscript characters of one visual line."""
    groups = _baseline_groups(chars, line)
    if len(groups) < 2:
        return {}
    # The body: among the groups set in the largest letters, the one with the most
    # letters. Letters set the size of text; a large symbol (a sum sign) does not.
    largest = max(_letter_size(chars, group) for group in groups)
    if largest == 0:
        return {}
    body = max(
        (g for g in groups if _letter_size(chars, g) >= BODY_SIZE_RATIO * largest),
        key=lambda g: (sum(chars[i].text.isalnum() for i in g), len(g)),
    )
    body_size = statistics.median(chars[i].size for i in body)
    body_baseline = statistics.median(chars[i].baseline for i in body)
    min_shift = max(MIN_SHIFT_POINTS, MIN_SHIFT * body_size)

    shifted: dict[int, Script] = {}
    for index in line:
        char = chars[index]
        shift = char.baseline - body_baseline
        # A script sits within its line: it is moved by less than its own size. A drop cap
        # or text from another line is moved further.
        if (
            char.size <= MAX_SCRIPT_SIZE * body_size
            and min_shift <= abs(shift) <= char.size
        ):
            shifted[index] = Script.SUPER if shift > 0 else Script.SUB

    scripts: dict[int, Script] = {}
    for script, run in itertools.groupby(line, key=shifted.get):
        run = list(run)
        if script is not None and _keep_run(
            chars, run, line=line, shifted=shifted, body_size=body_size
        ):
            scripts.update(dict.fromkeys(run, script))
    return scripts


def _keep_run(
    chars: Sequence[ScriptChar],
    run: list[int],
    *,
    line: list[int],
    shifted: dict[int, Script],
    body_size: float,
) -> bool:
    """Tell a script from other small, shifted text on the line."""
    position = line.index(run[0])
    neighbours = [
        k
        for k in (
            line[position - 1] if position > 0 else None,
            line[position + len(run)] if position + len(run) < len(line) else None,
        )
        if k is not None
    ]
    distances = [_distance(chars[k], chars, run) for k in neighbours]

    # It touches the text it marks, unlike a small page number or URL further along.
    touches_body = any(
        k not in shifted and d <= TOUCH_GAP * body_size
        for k, d in zip(neighbours, distances)
    )
    # A subscript and a superscript stacked over one base.
    stacked = any(k in shifted and shifted[k] != shifted[run[0]] for k in neighbours)
    if not (touches_body or stacked):
        return False

    # Symbol fonts (arrows, bullets) are often drawn small and off the baseline.
    text = [chars[i].text for i in run]
    if not (any(t.isalnum() for t in text) or all(t in FOOTNOTE_MARKS for t in text)):
        return False

    in_word = (
        len(neighbours) == 2
        and all(t.isalpha() for t in text)
        and all(
            k not in shifted
            and chars[k].text.isalpha()
            and d <= IN_WORD_GAP * body_size
            for k, d in zip(neighbours, distances)
        )
    )
    return not in_word


def _distance(char: ScriptChar, chars: Sequence[ScriptChar], run: list[int]) -> float:
    """Return the horizontal distance between a character and a run, 0 if they overlap."""
    left = min(chars[i].left for i in run)
    right = max(chars[i].right for i in run)
    return max(0.0, char.left - right, left - char.right)


def _folded(text: str) -> list[tuple[str, int]]:
    """Fold the non-space characters for matching, each with its index in ``text``."""
    out = []
    for index, ch in enumerate(text):
        if ch.isspace():
            continue
        # NFKC unfolds ligatures into several characters.
        for part in unicodedata.normalize("NFKC", ch).translate(_PUNCTUATION_FOLD):
            if not part.isspace():
                out.append((part, index))
    return out


def _align(text: str, chars: Sequence[ScriptChar]) -> dict[int, int] | None:
    """Map the characters of ``text`` to PDF character indexes, or None if too few match."""
    ours = _folded(text)
    theirs = _folded("".join(c.text for c in chars))
    if not ours:
        return None
    if [a for a, _ in ours] == [b for b, _ in theirs]:
        blocks = [(0, 0, len(ours))]
    elif max(len(ours), len(theirs)) > MAX_ALIGN_CHARS:
        return None
    else:
        matcher = difflib.SequenceMatcher(
            None,
            "".join(a for a, _ in ours),
            "".join(b for b, _ in theirs),
            autojunk=False,
        )
        blocks = [tuple(block) for block in matcher.get_matching_blocks()]
    mapping: dict[int, int] = {}
    for a, b, size in blocks:
        for k in range(size):
            mapping.setdefault(ours[a + k][1], theirs[b + k][1])
    solid = sum(not ch.isspace() for ch in text)
    if len(mapping) < MIN_ALIGNED * solid:
        return None
    return mapping
