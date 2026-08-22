"""Tests for docling.utils.speaker_diarization.

assign_speakers is pure overlap logic and runs fully in core CI. diarize() is
only exercised for the missing-dependency fallback so Resemblyzer / audio
decoding stay out of the default pytest lane.
"""

from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

from docling.utils.speaker_diarization import (
    DiarizationResult,
    SpeakerSegment,
    assign_speakers,
    diarize,
)


def _item(
    start: float | None,
    end: float | None,
    *,
    speaker: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(start_time=start, end_time=end, speaker=speaker)


# --------------------------------------------------------------------------- #
# assign_speakers
# --------------------------------------------------------------------------- #


def test_assign_speakers_empty_diarization_leaves_items_unchanged() -> None:
    items = [_item(0.0, 1.0)]
    result = assign_speakers(items, DiarizationResult())

    assert result is items
    assert items[0].speaker is None


def test_assign_speakers_empty_transcript_returns_empty_list() -> None:
    diarization = DiarizationResult(
        segments=[SpeakerSegment(0.0, 1.0, "SPEAKER_00")],
        num_speakers=1,
        speaker_ids=["SPEAKER_00"],
    )

    assert assign_speakers([], diarization) == []


def test_assign_speakers_picks_speaker_with_maximum_overlap() -> None:
    items = [_item(0.4, 1.2)]
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(0.0, 0.5, "SPEAKER_00"),
            SpeakerSegment(0.5, 2.0, "SPEAKER_01"),
        ],
        num_speakers=2,
        speaker_ids=["SPEAKER_00", "SPEAKER_01"],
    )

    assign_speakers(items, diarization)

    # Overlap with SPEAKER_00 is 0.1s; with SPEAKER_01 is 0.7s.
    assert items[0].speaker == "SPEAKER_01"


def test_assign_speakers_leaves_speaker_unset_when_no_overlap() -> None:
    items = [_item(5.0, 6.0)]
    diarization = DiarizationResult(
        segments=[SpeakerSegment(0.0, 1.0, "SPEAKER_00")],
        num_speakers=1,
        speaker_ids=["SPEAKER_00"],
    )

    assign_speakers(items, diarization)

    assert items[0].speaker is None


def test_assign_speakers_missing_times_collapse_to_zero_length_and_stay_unset() -> None:
    # None start → 0.0; None end → start. Zero-length intervals never produce
    # positive overlap, so speaker stays unset without erroring.
    items = [_item(None, None), _item(1.2, None)]
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(0.0, 0.5, "SPEAKER_00"),
            SpeakerSegment(1.0, 2.0, "SPEAKER_01"),
        ],
        num_speakers=2,
        speaker_ids=["SPEAKER_00", "SPEAKER_01"],
    )

    assign_speakers(items, diarization)

    assert items[0].speaker is None
    assert items[1].speaker is None


def test_assign_speakers_keeps_earlier_speaker_on_tied_overlap() -> None:
    items = [_item(0.0, 2.0)]
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(0.0, 1.0, "SPEAKER_00"),
            SpeakerSegment(1.0, 2.0, "SPEAKER_01"),
        ],
        num_speakers=2,
        speaker_ids=["SPEAKER_00", "SPEAKER_01"],
    )

    assign_speakers(items, diarization)

    # Both overlaps are 1.0s; assignment uses strict > so the first winner stays.
    assert items[0].speaker == "SPEAKER_00"


# --------------------------------------------------------------------------- #
# diarize — dependency fallback only
# --------------------------------------------------------------------------- #


def test_diarize_returns_empty_result_when_dependencies_missing(
    monkeypatch, tmp_path: Path
) -> None:
    real_import = builtins.__import__

    def _block_resemblyzer(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "resemblyzer" or name.startswith("resemblyzer."):
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _block_resemblyzer)

    result = diarize(tmp_path / "missing.wav")

    assert result == DiarizationResult()
    assert result.segments == []
    assert result.num_speakers == 0
    assert result.speaker_ids == []
