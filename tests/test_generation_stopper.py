# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the DocTags repetition stopper.

This is the guard that aborts a VLM generation stuck in a loop, so both
directions matter: failing to trip wastes a full generation budget on
repeated output, and tripping too eagerly truncates a legitimate document
whose items happen to be evenly spaced.

The logic is subtle -- a run must be *consecutive*, share both tag and inner
text, and show either duplicate boxes or one stable axis with regular
progression on the other -- and it is pure string handling, so it needs no
model to exercise.
"""

from __future__ import annotations

import sys

import pytest

from docling.models.utils.generation_utils import (
    DocTagsRepetitionStopper,
    GenerationStopper,
)


def _block(tag: str, box: tuple[int, int, int, int], text: str = "same") -> str:
    x, y, w, h = box
    return f"<{tag}><loc_{x}><loc_{y}><loc_{w}><loc_{h}>{text}</{tag}>"


def _doc(*blocks: str) -> str:
    return "".join(blocks)


@pytest.fixture
def stopper() -> DocTagsRepetitionStopper:
    return DocTagsRepetitionStopper()


# -- when it must trip ---------------------------------------------------


def test_identical_boxes_repeated_three_times_stop_generation(stopper):
    box = (10, 20, 30, 40)
    assert stopper.should_stop(
        _doc(_block("text", box), _block("text", box), _block("text", box))
    )


def test_a_column_walking_down_the_page_stops_generation(stopper):
    """Stable x and w with an evenly spaced y is the classic repetition loop."""
    blocks = [_block("text", (10, y, 30, 40)) for y in (100, 200, 300, 400)]
    assert stopper.should_stop(_doc(*blocks))


def test_a_row_walking_across_the_page_stops_generation(stopper):
    blocks = [_block("text", (x, 50, 30, 40)) for x in (100, 200, 300)]
    assert stopper.should_stop(_doc(*blocks))


def test_spacing_within_the_tolerance_still_stops(stopper):
    """Progression is allowed to wobble by up to 20% of the mean step."""
    # steps of 100 and 110: mean 105, tolerance 21, both within it.
    blocks = [_block("text", (10, y, 30, 40)) for y in (100, 200, 310)]
    assert stopper.should_stop(_doc(*blocks))


# -- when it must not trip -----------------------------------------------


def test_empty_output_does_not_stop(stopper):
    assert not stopper.should_stop("")


def test_text_without_any_doctags_does_not_stop(stopper):
    assert not stopper.should_stop("just some prose with no tags at all")


def test_two_repeats_are_not_enough(stopper):
    box = (10, 20, 30, 40)
    assert not stopper.should_stop(_doc(_block("text", box), _block("text", box)))


def test_irregular_spacing_does_not_stop(stopper):
    """A real document's items are rarely evenly spaced."""
    blocks = [_block("text", (10, y, 30, 40)) for y in (100, 200, 500)]
    assert not stopper.should_stop(_doc(*blocks))


def test_a_run_broken_by_different_inner_text_does_not_stop(stopper):
    """Three evenly spaced items saying different things is a normal page."""
    blocks = [
        _block("text", (10, y, 30, 40), text=f"line {i}")
        for i, y in enumerate((100, 200, 300))
    ]
    assert not stopper.should_stop(_doc(*blocks))


def test_a_run_broken_by_a_different_tag_does_not_stop(stopper):
    """The run must be consecutive; an interleaved tag resets it."""
    blocks = _doc(
        _block("text", (10, 100, 30, 40)),
        _block("caption", (10, 150, 30, 40)),
        _block("text", (10, 200, 30, 40)),
        _block("caption", (10, 250, 30, 40)),
        _block("text", (10, 300, 30, 40)),
    )
    assert not stopper.should_stop(blocks)


def test_boxes_that_change_size_and_position_do_not_stop(stopper):
    """No axis is stable and no dimension repeats, so this is a real page."""
    blocks = [
        _block("text", box)
        for box in ((10, 100, 30, 40), (60, 200, 35, 45), (110, 300, 40, 50))
    ]
    assert not stopper.should_stop(_doc(*blocks))


def test_a_constant_box_size_is_enough_to_trip_even_when_x_moves(stopper):
    """Stability of *either* x or w satisfies the first arm of the check.

    Items marching diagonally but keeping an identical width still read as a
    repetition loop. Pinned because it is not obvious from the docstring,
    which describes the case as "stable X/W with regular Y progression".
    """
    blocks = [
        _block("text", (x, y, 30, 40)) for x, y in ((10, 100), (60, 200), (110, 300))
    ]
    assert stopper.should_stop(_doc(*blocks))


def test_a_non_consecutive_repeat_at_the_end_is_still_evaluated(stopper):
    """The final run is checked after the loop, not only on a tag change."""
    blocks = _doc(
        _block("caption", (5, 5, 5, 5), text="intro"),
        *[_block("text", (10, y, 30, 40)) for y in (100, 200, 300)],
    )
    assert stopper.should_stop(blocks)


# -- configuration -------------------------------------------------------


def test_lookback_defaults_and_can_be_overridden():
    assert DocTagsRepetitionStopper().lookback_tokens() == sys.maxsize
    # The constructor argument is clamped to at least 1.
    assert DocTagsRepetitionStopper(lookback_tokens=0)._lookback_tokens == 1
    assert DocTagsRepetitionStopper(lookback_tokens=50)._lookback_tokens == 50


def test_the_check_interval_is_clamped_to_at_least_one():
    assert DocTagsRepetitionStopper(N=0).N == 1
    assert DocTagsRepetitionStopper(N=8).N == 8


def test_the_base_interface_requires_a_should_stop_implementation():
    class Custom(GenerationStopper):
        def should_stop(self, s: str) -> bool:
            return "halt" in s

    custom = Custom()
    assert custom.should_stop("please halt")
    assert not custom.should_stop("carry on")
    assert custom.lookback_tokens() == sys.maxsize
