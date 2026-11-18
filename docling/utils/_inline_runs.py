# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from collections.abc import Callable, Sequence
from typing import TypeVar

_RunT = TypeVar("_RunT")


def trim_run_edges(
    runs: Sequence[_RunT],
    *,
    get_text: Callable[[_RunT], str],
    with_text: Callable[[_RunT, str], _RunT],
    is_trimmable: Callable[[_RunT], bool] = lambda run: True,
) -> list[_RunT]:
    """Trim the outer edges of a block of inline runs and keep interior whitespace.

    Leading and trailing runs that hold only whitespace are dropped. The first and the
    last remaining run lose their outer whitespace. Whitespace between runs is never
    changed. Trimmable runs with no text are dropped.

    Args:
        runs: The runs of one block, in document order. The sequence is not changed.
        get_text: Return the text of a run.
        with_text: Return a copy of a run with new text.
        is_trimmable: Return False for a run that must stay as it is, even when it has
            no text. Trimming stops at such a run on either edge.

    Returns:
        A new list with the trimmed runs.
    """
    trimmed = list(runs)

    start = 0
    while start < len(trimmed) and is_trimmable(trimmed[start]):
        head = with_text(trimmed[start], get_text(trimmed[start]).lstrip())
        if get_text(head):
            trimmed[start] = head
            break
        start += 1

    end = len(trimmed)
    while end > start and is_trimmable(trimmed[end - 1]):
        tail = with_text(trimmed[end - 1], get_text(trimmed[end - 1]).rstrip())
        if get_text(tail):
            trimmed[end - 1] = tail
            break
        end -= 1

    return [run for run in trimmed[start:end] if get_text(run) or not is_trimmable(run)]
