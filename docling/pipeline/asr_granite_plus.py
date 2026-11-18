# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Word timing and speaker labels from Granite Speech Plus, on any backend.

The model has one mode for word timing and one for speaker turns, so each option costs
one decoding pass. This module runs the passes through a callback and merges the output.

Import this module only where it is used: `split_on_pauses` needs numpy, which must not
reach the import path of a docling-slim install.
"""

import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Final, Literal

from docling.pipeline.asr_conversation import _ConversationItem, _ConversationWord
from docling.pipeline.asr_windows import split_on_pauses

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

_log = logging.getLogger(__name__)

PlusTask = Literal["asr", "timestamps", "saa"]
PlusDecode = Callable[["npt.NDArray[np.floating]", PlusTask], str]

# Verbatim from the model card. The model falls back to plain text on other wording.
PLUS_PROMPTS: Final[dict[PlusTask, str]] = {
    "asr": " can you transcribe the speech into a written format?",
    "timestamps": (
        " Timestamps: Transcribe the speech. After each word, add a timestamp tag "
        "showing the end time in centiseconds, e.g. hello [T:45] world [T:82]"
    ),
    "saa": (
        " Speaker attribution: Transcribe and denote who is speaking by adding "
        "[Speaker 1]: and [Speaker 2]: tags before speaker turns."
    ),
}
PLUS_SYSTEM_PROMPT: Final = (
    "Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant"
)
# The output limit of the model card. A timestamp tag costs several tokens for each word.
PLUS_TOKENS_PER_SECOND: Final[dict[PlusTask, int]] = {
    "asr": 12,
    "timestamps": 48,
    "saa": 16,
}

_TIMESTAMP_TAG: Final = re.compile(r"\[T:(\d+)\]")
_SPEAKER_TAG: Final = re.compile(r"\[Speaker (\d+)\]:")
_SILENCE: Final = "_"
_TIMESTAMP_ROLLOVER_SECONDS: Final = 10.0

# The model numbers speakers by order of appearance in each call, so a short window
# swaps the numbers. It works well on 2 to 5 minutes, and the overlap between windows
# shows which number is which person.
_SPEAKER_WINDOW_SECONDS: Final = 240.0
_SPEAKER_SEARCH_SECONDS: Final = 30.0
_SPEAKER_OVERLAP_SECONDS: Final = 60.0
_MIN_MATCH_WORDS: Final = 3
# Align in parts, because a full recording is too long for `SequenceMatcher`.
_ALIGN_SECONDS: Final = 300.0


@dataclass(frozen=True)
class _TimedWord:
    text: str
    start: float
    end: float
    after_silence: bool


@dataclass(frozen=True)
class _SpeakerWord:
    text: str
    speaker: int
    # An estimate from the position of the word in its window.
    time: float


def _key(word: str) -> str:
    return re.sub(r"[^\w']", "", word.lower())


def _timed_words(text: str, start: float, end: float) -> list[_TimedWord]:
    """Read the timestamp tags of one window.

    Each tag gives the end of its word in the window, modulo 10 seconds. A silence is
    the token `_` with its own tag. Text without a tag is not lost: it gets the
    remaining time of the window.
    """
    parts = _TIMESTAMP_TAG.split(text)
    words: list[_TimedWord] = []
    previous_end = 0.0
    rollovers = 0.0
    after_silence = True
    for token, tag in zip(parts[::2], parts[1::2]):
        tag_seconds = int(tag) / 100
        while tag_seconds + rollovers < previous_end:
            rollovers += _TIMESTAMP_ROLLOVER_SECONDS
        word_end = tag_seconds + rollovers
        token = token.strip()
        if token and token != _SILENCE:
            words.append(
                _TimedWord(
                    token,
                    min(start + previous_end, end),
                    min(start + word_end, end),
                    after_silence,
                )
            )
            after_silence = False
        else:
            after_silence = True
        previous_end = word_end
    tail = parts[-1].strip()
    if tail and tail != _SILENCE:
        words.append(
            _TimedWord(tail, min(start + previous_end, end), end, after_silence)
        )
    return words


def _speaker_turns(text: str) -> list[tuple[int, list[str]]]:
    """Read the speaker tags of one window. Text before the first tag is the first turn."""
    parts = _SPEAKER_TAG.split(text)
    turns = [(1, parts[0].split())] if parts[0].strip() else []
    turns.extend(
        (int(speaker), words.split())
        for speaker, words in zip(parts[1::2], parts[2::2])
    )
    return turns


def _speaker_map(
    votes: Counter[tuple[int, int]], local_speakers: list[int], recent: list[int]
) -> dict[int, int]:
    """Map the speakers of a window to the speakers of the recording.

    A long monologue in the overlap shows only one person. The other number of the
    window is then probably the person who spoke before, so it reuses a known speaker
    before it gets a new one.
    """
    mapping: dict[int, int] = {}
    for (local, known), _ in votes.most_common():
        if local not in mapping and known not in mapping.values():
            mapping[local] = known
    next_new = max(recent, default=0) + 1
    for local in local_speakers:
        if local in mapping:
            continue
        free = [known for known in recent if known not in mapping.values()]
        if free:
            mapping[local] = free[0]
        else:
            mapping[local] = next_new
            next_new += 1
    return mapping


def _stitch_window(
    done: list[_SpeakerWord], window: list[_SpeakerWord]
) -> list[_SpeakerWord]:
    """Add a window to the recording, with stable speakers and without repeated text."""
    if not done:
        return window
    tail = done[-1000:]
    tail_start = len(done) - len(tail)
    matcher = SequenceMatcher(
        None,
        [_key(word.text) for word in tail],
        [_key(word.text) for word in window],
        autojunk=False,
    )
    blocks = [
        block
        for block in matcher.get_matching_blocks()
        if block.size >= _MIN_MATCH_WORDS
    ]
    votes: Counter[tuple[int, int]] = Counter()
    for block in blocks:
        for offset in range(block.size):
            votes[
                (window[block.b + offset].speaker, tail[block.a + offset].speaker)
            ] += 1
    # Most recent first, so that a number without votes reuses the last known speaker.
    recent = list(dict.fromkeys(word.speaker for word in reversed(done)))
    local_speakers = list(dict.fromkeys(word.speaker for word in window))
    mapping = _speaker_map(votes, local_speakers, recent)
    mapped = [
        _SpeakerWord(word.text, mapping[word.speaker], word.time) for word in window
    ]
    if not blocks:
        _log.debug("No text in the overlap of two speaker windows")
        return done + mapped
    # Keep the first copy up to the end of the last match, and the new text after it.
    last = blocks[-1]
    return done[: tail_start + last.a + last.size] + mapped[last.b + last.size :]


def _speaker_words(
    waveform: "npt.NDArray[np.floating]", sample_rate: int, decode: PlusDecode
) -> list[_SpeakerWord]:
    words: list[_SpeakerWord] = []
    for start, end in split_on_pauses(
        waveform, sample_rate, _SPEAKER_WINDOW_SECONDS, _SPEAKER_SEARCH_SECONDS
    ):
        start = max(0.0, start - (_SPEAKER_OVERLAP_SECONDS if words else 0.0))
        first, last = round(start * sample_rate), round(end * sample_rate)
        turns = _speaker_turns(decode(waveform[first:last], "saa"))
        count = sum(len(turn) for _, turn in turns)
        window: list[_SpeakerWord] = []
        for speaker, turn in turns:
            for word in turn:
                position = len(window) / count
                window.append(
                    _SpeakerWord(word, speaker, start + position * (end - start))
                )
        words = _stitch_window(words, window)
    return words


def _speakers_of_timed_words(
    timed: list[_TimedWord], speakers: list[_SpeakerWord]
) -> list[int]:
    """Give each timed word a speaker. A word without a match takes its neighbor's."""
    assigned: list[int | None] = [None] * len(timed)
    next_timed = 0
    next_speaker = 0
    while next_timed < len(timed):
        limit = timed[next_timed].start + _ALIGN_SECONDS
        timed_end = next_timed
        while timed_end < len(timed) and timed[timed_end].start < limit:
            timed_end += 1
        # The speaker times are estimates, so this part also looks a little further.
        speaker_end = next_speaker
        while (
            speaker_end < len(speakers)
            and speakers[speaker_end].time < limit + _SPEAKER_SEARCH_SECONDS
        ):
            speaker_end += 1
        matcher = SequenceMatcher(
            None,
            [_key(word.text) for word in timed[next_timed:timed_end]],
            [_key(word.text) for word in speakers[next_speaker:speaker_end]],
            autojunk=False,
        )
        last_match = next_speaker
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                assigned[next_timed + block.a + offset] = speakers[
                    next_speaker + block.b + offset
                ].speaker
            if block.size:
                last_match = next_speaker + block.b + block.size
        next_timed = timed_end
        next_speaker = max(last_match, next_speaker)

    resolved: list[int] = []
    known = next((speaker for speaker in assigned if speaker is not None), 1)
    for speaker in assigned:
        known = speaker if speaker is not None else known
        resolved.append(known)
    return resolved


def _items_from_timed_words(
    timed: list[_TimedWord], speakers: list[int] | None
) -> list[_ConversationItem]:
    """Group words to one item for each phrase. A silence or a new speaker ends it."""
    items: list[_ConversationItem] = []
    phrase: list[_TimedWord] = []
    phrase_speaker: int | None = None

    def end_phrase() -> None:
        if not phrase:
            return
        items.append(
            _ConversationItem(
                text=" ".join(word.text for word in phrase),
                start_time=phrase[0].start,
                end_time=phrase[-1].end,
                speaker_id=phrase_speaker,
                speaker=None if phrase_speaker is None else f"Speaker {phrase_speaker}",
                words=[
                    _ConversationWord(
                        text=word.text, start_time=word.start, end_time=word.end
                    )
                    for word in phrase
                ],
            )
        )
        phrase.clear()

    for index, word in enumerate(timed):
        speaker = None if speakers is None else speakers[index]
        if word.after_silence or speaker != phrase_speaker:
            end_phrase()
        phrase_speaker = speaker
        phrase.append(word)
    end_phrase()
    return items


def _items_from_speaker_words(
    speakers: list[_SpeakerWord], total_seconds: float
) -> list[_ConversationItem]:
    """Give one item for each speaker turn, with estimated times."""
    items: list[_ConversationItem] = []
    start = 0
    for index in range(1, len(speakers) + 1):
        if index < len(speakers) and speakers[index].speaker == speakers[start].speaker:
            continue
        speaker = speakers[start].speaker
        items.append(
            _ConversationItem(
                text=" ".join(word.text for word in speakers[start:index]),
                start_time=speakers[start].time,
                end_time=(
                    speakers[index].time if index < len(speakers) else total_seconds
                ),
                speaker_id=speaker,
                speaker=f"Speaker {speaker}",
            )
        )
        start = index
    return items


def transcribe_plus(
    waveform: "npt.NDArray[np.floating]",
    sample_rate: int,
    window_seconds: float,
    search_seconds: float,
    word_timestamps: bool,
    speaker_attribution: bool,
    decode: PlusDecode,
) -> list[_ConversationItem]:
    """Run the selected passes on a mono recording and merge their output."""
    windows = split_on_pauses(waveform, sample_rate, window_seconds, search_seconds)
    if not word_timestamps and not speaker_attribution:
        items: list[_ConversationItem] = []
        for start, end in windows:
            first, last = round(start * sample_rate), round(end * sample_rate)
            text = decode(waveform[first:last], "asr").strip()
            if text:
                items.append(
                    _ConversationItem(text=text, start_time=start, end_time=end)
                )
        return items

    if not word_timestamps:
        return _items_from_speaker_words(
            _speaker_words(waveform, sample_rate, decode),
            waveform.size / sample_rate,
        )

    speakers = (
        _speaker_words(waveform, sample_rate, decode) if speaker_attribution else None
    )
    timed: list[_TimedWord] = []
    for start, end in windows:
        first, last = round(start * sample_rate), round(end * sample_rate)
        timed.extend(
            _timed_words(decode(waveform[first:last], "timestamps"), start, end)
        )
    return _items_from_timed_words(
        timed,
        None if speakers is None else _speakers_of_timed_words(timed, speakers),
    )
