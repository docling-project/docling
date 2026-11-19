# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Split a long recording into windows at pauses in the speech.

Import this module only where it is used: it needs numpy, and it must not reach the
import path of a docling-slim install.
"""

import itertools
import logging
from typing import Final

import numpy as np
import numpy.typing as npt

_log = logging.getLogger(__name__)

_FRAME_SECONDS: Final = 0.02
_MIN_PAUSE_SECONDS: Final = 0.2
# A frame is quiet below this fraction of the range from the noise floor to speech.
# The range comes from each file, because a fixed level fails on quiet recordings.
_QUIET_FRACTION: Final = 0.15


def _quiet_frames(
    waveform: npt.NDArray[np.floating], frame_size: int
) -> npt.NDArray[np.bool_]:
    frame_count = waveform.size // frame_size
    frames = waveform[: frame_count * frame_size].reshape(frame_count, frame_size)
    energy = np.sqrt(np.mean(frames**2, axis=1))
    floor, speech = np.percentile(energy, [10, 90])
    return energy < floor + _QUIET_FRACTION * (speech - floor)


def _longest_pause_middle(
    quiet: npt.NDArray[np.bool_], first: int, last: int
) -> float | None:
    """Return the middle frame of the longest quiet run in `first..last`, if any."""
    min_frames = round(_MIN_PAUSE_SECONDS / _FRAME_SECONDS)
    best_length = min_frames - 1
    best_middle: float | None = None
    run_start: int | None = None
    for index in range(first, last + 1):
        if not quiet[index]:
            run_start = None
            continue
        if run_start is None:
            run_start = index
        length = index - run_start + 1
        if length > best_length:
            best_length = length
            best_middle = (run_start + index) / 2
    return best_middle


def split_on_pauses(
    waveform: npt.NDArray[np.floating],
    sample_rate: int,
    target_s: float,
    search_s: float,
) -> list[tuple[float, float]]:
    """Split a mono waveform into consecutive windows and return their edges in seconds.

    Each cut is at the middle of the longest pause within `target_s` +- `search_s` of
    the window start, or at `target_s` when that range has no pause. A pause is not
    always a sentence boundary. Windows do not overlap, so no text repeats.

    A pause that would leave a last window below `target_s - search_s` is not a cut
    point. The cut at `target_s` has no such limit, because a recording can be too short
    to give two windows of that length. The last window has at least
    `search_s` seconds, so a small `search_s` can give a short last window.
    """
    if search_s >= target_s:
        raise ValueError(
            f"search_s ({search_s}) must be smaller than target_s ({target_s})."
        )
    total_s = waveform.size / sample_rate
    if total_s <= target_s + search_s:
        return [(0.0, total_s)] if waveform.size else []

    frame_size = round(sample_rate * _FRAME_SECONDS)
    frame_s = frame_size / sample_rate
    quiet = _quiet_frames(waveform, frame_size)

    edges = [0.0]
    while total_s - edges[-1] > target_s + search_s:
        start = edges[-1]
        latest_cut = min(start + target_s + search_s, total_s - (target_s - search_s))
        middle = _longest_pause_middle(
            quiet,
            round((start + target_s - search_s) / frame_s),
            min(round(latest_cut / frame_s), quiet.size - 1),
        )
        if middle is None:
            _log.debug("No pause near %.1f s; cutting at the target", start + target_s)
            edges.append(start + target_s)
        else:
            edges.append(middle * frame_s)
    edges.append(total_s)
    return list(itertools.pairwise(edges))
