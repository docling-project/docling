# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the word timing and speaker labels of Granite Speech Plus."""

from collections import Counter

import numpy as np
import pytest

from docling.cli.main import _resolve_asr_options
from docling.datamodel.asr_model_specs import (
    GRANITE_SPEECH_4_1_2B_PLUS_PRESET,
    AsrModelType,
)
from docling.datamodel.pipeline_options_asr_model import (
    InferenceAsrFramework,
    InlineAsrMlxAudioOptions,
    InlineAsrTransformersOptions,
)
from docling.pipeline.asr_granite_plus import (
    _items_from_speaker_words,
    _items_from_timed_words,
    _speaker_map,
    _speaker_turns,
    _speakers_of_timed_words,
    _SpeakerWord,
    _stitch_window,
    _timed_words,
    transcribe_plus,
)
from docling.pipeline.video_pipeline import _gives_finished_phrases

SAMPLE_RATE = 16000


def _window(speaker_words: list[tuple[int, str]], time: float = 0.0):
    return [
        _SpeakerWord(word, speaker, time + index)
        for index, (speaker, word) in enumerate(
            (speaker, word) for speaker, text in speaker_words for word in text.split()
        )
    ]


# ---------------------------------------------------------------- word timing


def test_timestamp_tags_give_times_and_mark_silences():
    words = _timed_words(
        "_ [T:91] shakespeare [T:173] on [T:201] _ [T:328] by [T:352]", 5.0, 20.0
    )

    assert [(w.text, w.after_silence) for w in words] == [
        ("shakespeare", True),
        ("on", False),
        ("by", True),
    ]
    assert words[1].start == pytest.approx(6.73)
    assert words[1].end == pytest.approx(7.01)


def test_timestamps_continue_after_the_ten_second_rollover():
    words = _timed_words("one [T:950] two [T:20] three [T:990] four [T:5]", 0.0, 40.0)

    assert [word.end for word in words] == pytest.approx([9.5, 10.2, 19.9, 20.05])


def test_timestamps_stay_inside_the_window():
    words = _timed_words("one [T:900] two [T:950]", 30.0, 35.0)

    assert words[-1].end == 35.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The model stopped tagging: the text must stay in the document.
        ("one [T:100] two three", ["one", "two three"]),
        ("no tags at all", ["no tags at all"]),
        ("", []),
        ("_ [T:100]", []),
    ],
)
def test_text_without_tags_is_kept(text, expected):
    assert [word.text for word in _timed_words(text, 0.0, 8.0)] == expected


def test_a_silence_or_a_new_speaker_ends_an_item():
    words = _timed_words(
        "_ [T:10] one [T:20] two [T:30] three [T:40] _ [T:50] four [T:60]", 0.0, 9.0
    )

    items = _items_from_timed_words(words, [1, 1, 2, 2])

    assert [(item.text, item.speaker, item.speaker_id) for item in items] == [
        ("one two", "Speaker 1", 1),
        ("three", "Speaker 2", 2),
        ("four", "Speaker 2", 2),
    ]
    assert len(items[0].words) == 2


def test_items_without_speaker_pass_have_no_label():
    items = _items_from_timed_words(_timed_words("one [T:20]", 0.0, 9.0), None)

    assert items[0].speaker is None


# ---------------------------------------------------------------- speakers


def test_speaker_tags_give_turns_and_keep_text_before_the_first_tag():
    assert _speaker_turns("first [Speaker 1]: hello you [Speaker 2]: hi") == [
        (1, ["first"]),
        (1, ["hello", "you"]),
        (2, ["hi"]),
    ]


def test_window_with_swapped_numbers_keeps_the_people_stable():
    # The real failure: each call numbers speakers by order of appearance, so the same
    # person is Speaker 2 in one window and Speaker 1 in the next.
    done = _window([(1, "the host asks about the bank"), (2, "i run the bank today")])
    window = _window(
        [(1, "i run the bank today"), (2, "what about rates"), (1, "they are fine")]
    )

    words = _stitch_window(done, window)

    assert [(word.speaker, word.text) for word in words[-7:]] == [
        (2, "today"),
        (1, "what"),
        (1, "about"),
        (1, "rates"),
        (2, "they"),
        (2, "are"),
        (2, "fine"),
    ]


def test_overlap_text_is_not_repeated():
    done = _window([(1, "one two three four five six")])
    window = _window([(1, "four five six seven eight")])

    words = _stitch_window(done, window)

    assert " ".join(word.text for word in words) == (
        "one two three four five six seven eight"
    )


def test_monologue_in_the_overlap_still_reuses_the_other_known_person():
    # The overlap only shows person 2. The other number is probably person 1.
    assert _speaker_map(
        votes=Counter({(1, 2): 50}),
        local_speakers=[1, 2],
        recent=[2, 1],
    ) == {1: 2, 2: 1}


def test_new_person_gets_a_new_number_when_all_known_people_are_taken():
    assert _speaker_map(
        votes=Counter({(1, 1): 9, (2, 2): 9}),
        local_speakers=[1, 2, 3],
        recent=[1, 2],
    ) == {1: 1, 2: 2, 3: 3}


def test_window_without_overlap_text_is_appended():
    done = _window([(1, "one two three")])

    words = _stitch_window(done, _window([(1, "four five six")]))

    assert len(words) == 6


def test_timed_words_take_the_speaker_of_the_matching_word():
    timed = _timed_words("one [T:100] two [T:200] extra [T:300] four [T:400]", 0.0, 9.0)
    speakers = _window([(1, "one two"), (2, "four")])

    # `extra` has no match, so it takes the speaker of the word before it.
    assert _speakers_of_timed_words(timed, speakers) == [1, 1, 1, 2]


def test_speaker_only_mode_gives_one_item_for_each_turn():
    items = _items_from_speaker_words(_window([(1, "a b"), (2, "c")]), 10.0)

    assert [(i.text, i.speaker, i.start_time, i.end_time) for i in items] == [
        ("a b", "Speaker 1", 0.0, 2.0),
        ("c", "Speaker 2", 2.0, 10.0),
    ]


# ---------------------------------------------------------------- the two passes


class _Passes:
    def __init__(self) -> None:
        self.tasks: list[str] = []

    def __call__(self, waveform, task) -> str:
        self.tasks.append(task)
        return {
            "timestamps": "_ [T:10] hello [T:50] _ [T:70] world [T:90]",
            "saa": "[Speaker 1]: hello [Speaker 2]: world",
            "asr": "hello world",
        }[task]


@pytest.mark.parametrize(
    ("timestamps", "speakers", "expected_tasks", "expected"),
    [
        (
            True,
            True,
            ["saa", "timestamps"],
            [("hello", "Speaker 1"), ("world", "Speaker 2")],
        ),
        (True, False, ["timestamps"], [("hello", None), ("world", None)]),
        (False, True, ["saa"], [("hello", "Speaker 1"), ("world", "Speaker 2")]),
        (False, False, ["asr"], [("hello world", None)]),
    ],
)
def test_options_select_the_passes(timestamps, speakers, expected_tasks, expected):
    decode = _Passes()

    items = transcribe_plus(
        np.full(10 * SAMPLE_RATE, 0.2, dtype=np.float32),
        SAMPLE_RATE,
        30.0,
        10.0,
        timestamps,
        speakers,
        decode,
    )

    assert decode.tasks == expected_tasks
    assert [(item.text, item.speaker) for item in items] == expected


# ---------------------------------------------------------------- options


@pytest.mark.parametrize(
    "model",
    [
        AsrModelType.GRANITE_SPEECH_4_1_2B_PLUS_MLX_AUDIO,
        AsrModelType.GRANITE_SPEECH_4_1_2B_PLUS_TRANSFORMERS,
    ],
)
def test_presets_enable_both_modes_and_keep_the_pinned_commit(model):
    options = _resolve_asr_options(model)

    assert options.model_family == "granite_plus"
    assert options.word_timestamps is True
    assert options.speaker_attribution is True
    assert options.revision == GRANITE_SPEECH_4_1_2B_PLUS_PRESET.model_spec.revision


def test_plus_preset_supports_both_backends():
    assert GRANITE_SPEECH_4_1_2B_PLUS_PRESET.model_spec.supported_engines == {
        InferenceAsrFramework.TRANSFORMERS,
        InferenceAsrFramework.MLX_AUDIO,
    }


@pytest.mark.parametrize(
    "options_cls", [InlineAsrTransformersOptions, InlineAsrMlxAudioOptions]
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"task": "translate"},
        {"prompt": "keywords"},
        {"prompt": "keywords", "word_timestamps": False},
    ],
)
def test_options_that_the_modes_replace_are_rejected(options_cls, kwargs):
    with pytest.raises(ValueError, match="granite_plus"):
        options_cls(
            repo_id="r/m", revision="a" * 40, model_family="granite_plus", **kwargs
        )


def test_fallback_to_transformers_keeps_the_switches():
    options = GRANITE_SPEECH_4_1_2B_PLUS_PRESET.to_auto_options()
    if not isinstance(options, InlineAsrMlxAudioOptions):
        options = GRANITE_SPEECH_4_1_2B_PLUS_PRESET.to_options(
            InferenceAsrFramework.MLX_AUDIO
        ).model_copy(
            update={
                "transformers_fallback": GRANITE_SPEECH_4_1_2B_PLUS_PRESET.to_options(
                    InferenceAsrFramework.TRANSFORMERS
                )
            }
        )

    fallback = options.model_copy(
        update={"speaker_attribution": False, "word_timestamps": False}
    ).transformers_fallback_options()

    assert fallback is not None
    assert fallback.speaker_attribution is False
    assert fallback.word_timestamps is False


def test_prompt_works_when_both_modes_are_off():
    options = InlineAsrTransformersOptions(
        repo_id="r/m",
        model_family="granite_plus",
        prompt="keywords",
        word_timestamps=False,
        speaker_attribution=False,
    )

    assert options.prompt == "keywords"


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        ({}, True),
        ({"speaker_attribution": False}, True),
        ({"word_timestamps": False}, True),
        # Plain output has one item for each window, like the other Granite models.
        ({"word_timestamps": False, "speaker_attribution": False}, False),
    ],
)
def test_video_pipeline_does_not_merge_the_phrases_of_plus(update, expected):
    options = GRANITE_SPEECH_4_1_2B_PLUS_PRESET.to_options(
        InferenceAsrFramework.TRANSFORMERS
    ).model_copy(update=update)

    assert _gives_finished_phrases(options) is expected


def test_video_pipeline_still_merges_whisper_fragments():
    assert (
        _gives_finished_phrases(_resolve_asr_options(AsrModelType.WHISPER_TINY))
        is False
    )
