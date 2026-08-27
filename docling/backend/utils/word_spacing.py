# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Rejoin a drop cap with the word it begins.

A PDF has no notion of a word. The parser groups glyphs into word cells and then
joins those cells into a text line, putting a space at every boundary. That is
right when the boundary is a real gap, and wrong when the two runs are separate
only because the PDF switched font size in the middle of a word.

A drop cap is exactly that case. The oversized initial letter is always its own
text run, so "Realized" is parsed as a 34pt "R" abutting a 12pt "ealized" and
emitted as "R ealized".

The signature is narrow on purpose: a line-initial run of one letter, far taller
than the run beside it, with no gap between their boxes. Horizontal gap alone is
not enough evidence -- in rotated or diagrammatic text the boxes of two genuinely
separate words routinely touch -- so the height jump and the single letter are
what make this safe to act on. A line is rewritten only by deleting that one
space, never by inserting or altering anything.
"""

from docling_core.types.doc.page import TextCell

#: How much taller the initial letter must be than the text beside it. The issue
#: reports drop caps at roughly 2x body height; this leaves a little headroom.
DROP_CAP_HEIGHT_RATIO = 1.8

#: The largest gap between the two boxes, as a fraction of the body run's height,
#: that still counts as no gap at all. A real inter-word space runs to about a
#: quarter of the font size, well above this.
DROP_CAP_GAP_RATIO = 0.05


def _squash(text: str) -> str:
    return text.replace(" ", "")


def _align(
    line: TextCell, words: list[TextCell], start: int
) -> tuple[list[TextCell], int]:
    """Take the word cells that spell out ``line``, starting at ``words[start]``.

    Word cells and line cells share one reading order, so the words belonging to a
    line are consecutive. Returns the cells and the index to resume from; an empty
    list means the two lists disagreed and the line must be left alone.
    """
    target = _squash(line.text)
    taken: list[TextCell] = []
    acc = ""
    i = start
    while i < len(words) and len(acc) < len(target):
        acc += _squash(words[i].text)
        taken.append(words[i])
        i += 1
    if acc != target:
        # The lists have drifted apart. Give up on this line rather than guess,
        # and resume where we started so one odd line cannot desync the rest.
        return [], start
    return taken, i


def _is_drop_cap(initial: TextCell, following: TextCell) -> bool:
    """Is ``initial`` an oversized first letter of the word ``following`` finishes?"""
    if len(initial.text) != 1 or not initial.text.isalpha():
        return False
    if not following.text or not following.text[0].isalpha():
        return False

    initial_box = initial.rect.to_bounding_box()
    following_box = following.rect.to_bounding_box()
    if following_box.height <= 0:
        return False
    if initial_box.height < DROP_CAP_HEIGHT_RATIO * following_box.height:
        return False

    gap = following_box.l - initial_box.r
    return gap <= DROP_CAP_GAP_RATIO * following_box.height


def repair_drop_caps(lines: list[TextCell], words: list[TextCell]) -> int:
    """Rejoin drop caps with their words, in place.

    Args:
        lines: The page's text line cells. Mutated in place.
        words: The page's word cells, used only as geometric evidence.

    Returns:
        How many lines were rewritten.
    """
    if not lines or not words:
        return 0

    ordered_lines = sorted(lines, key=lambda c: c.index)
    ordered_words = sorted(words, key=lambda c: c.index)

    repaired = 0
    cursor = 0
    for line in ordered_lines:
        in_line, cursor = _align(line, ordered_words, cursor)
        if len(in_line) < 2:
            continue

        initial, following = in_line[0], in_line[1]
        if not _is_drop_cap(initial, following):
            continue

        # Delete only the one space between the two runs, by walking the line's
        # own characters. Nothing else in the line can be touched.
        head = line.text.find(initial.text)
        if head < 0:
            continue
        after = head + len(initial.text)
        if line.text[after : after + 1] != " ":
            continue
        if not line.text.startswith(following.text, after + 1):
            continue

        line.text = line.text[:after] + line.text[after + 1 :]
        repaired += 1

    return repaired
