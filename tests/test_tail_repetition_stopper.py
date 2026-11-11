# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the tail-repetition stopper used by code and formula extraction.

CodeFormulaV2 sometimes keeps emitting the same filler after a formula
(``\\ \\ \\ ...``, ``\\quad \\quad ...``) until ``max_new_tokens``, which costs a
full generation budget and leaves kilobytes of noise in the document. The
stopper has to catch those loops without touching LaTeX or code that repeats
legitimately, and when it stops one row of a batch the other rows must keep
generating.
"""

from __future__ import annotations

import pytest
import torch
from transformers import StoppingCriteriaList

from docling.models.utils.generation_utils import TailRepetitionStopper
from docling.models.utils.hf_stopping_criteria import HFStoppingCriteriaWrapper

# Looping outputs reported in docling#4539 (CodeFormulaV2 on arXiv
# cond-mat/0303516), with the repeated filler they ran into.
LOOPS = [
    pytest.param(
        r"A y = \lambda x , \quad A ^ { T } x = \mu y , \quad \ \ \ ( 9 4 ) \quad ",
        r"\ ",
        r"A y = \lambda x , \quad A ^ { T } x = \mu y , \quad \ \ \ ( 9 4 ) \quad",
        id="control-spaces",
    ),
    pytest.param(
        r"H _ { 1 } ( x ) = 1 - T + T x G _ { 1 } ( H _ { 1 } ( x ) ) , \quad ( 8 5 b ) ",
        r"\ ",
        r"H _ { 1 } ( x ) = 1 - T + T x G _ { 1 } ( H _ { 1 } ( x ) ) , \quad ( 8 5 b )",
        id="control-spaces-after-number",
    ),
    pytest.param(
        r"u & = \frac { L i _ { \alpha - 1 } ( u ) } { u \zeta ( \alpha - 1 ) } , ",
        r"& \text { \quad } ",
        r"u & = \frac { L i _ { \alpha - 1 } ( u ) } { u \zeta ( \alpha - 1 ) } ,",
        id="aligned-text-cells",
    ),
    pytest.param(
        r"r = \frac { \text {Tr} e - \| e ^ { 2 } \| } { 1 - \| e ^ { 2 } \| } . ",
        r"\quad ",
        r"r = \frac { \text {Tr} e - \| e ^ { 2 } \| } { 1 - \| e ^ { 2 } \| } .",
        id="quads",
    ),
]

# Output that repeats on purpose and must be left alone.
LEGITIMATE = [
    pytest.param(
        r"a = b \quad \quad \quad \quad \quad \quad \quad \quad c = d", id="quads"
    ),
    pytest.param(
        r"\begin{pmatrix} "
        + r" \\ ".join(" & ".join(["0"] * 8) for _ in range(8))
        + r" \end{pmatrix}",
        id="zero-matrix",
    ),
    pytest.param("=" * 120 + "\nTitle\n" + "=" * 120, id="separator-lines"),
    pytest.param("print('x')\n" * 12, id="repeated-code-lines"),
    pytest.param("", id="empty"),
]


@pytest.fixture
def stopper() -> TailRepetitionStopper:
    return TailRepetitionStopper()


@pytest.mark.parametrize(("content", "filler", "expected"), LOOPS)
def test_a_looping_formula_trips_the_stopper(stopper, content, filler, expected):
    assert stopper.should_stop(content + filler * 200)


@pytest.mark.parametrize(("content", "filler", "expected"), LOOPS)
def test_the_loop_is_cut_back_to_the_formula(stopper, content, filler, expected):
    assert stopper.strip(content + filler * 200) == expected


def test_a_loop_interrupted_mid_unit_is_still_caught(stopper):
    """The generation stops wherever max_new_tokens or the check lands."""
    looped = r"x = 1 , \quad ( 1 0 ) " + r"\emph { \ } " * 60 + r"\emph { "
    assert stopper.should_stop(looped)
    assert stopper.strip(looped) == r"x = 1 , \quad ( 1 0 )"


def test_a_short_repeat_does_not_trip_before_the_span_threshold(stopper):
    """Sixteen control spaces are only 32 characters, far from a loop."""
    assert not stopper.should_stop("x = 1 " + r"\ " * 16)


@pytest.mark.parametrize("text", LEGITIMATE)
def test_legitimate_repetition_is_left_alone(stopper, text):
    assert not stopper.should_stop(text)
    assert stopper.strip(text) == text


def test_stripping_never_leaves_a_dangling_backslash(stopper):
    """The escaped character of a trailing backslash can be part of the loop."""
    looped = "x = 1 \\" + " \\" * 400
    assert stopper.strip(looped) == "x = 1"


# -- the transformers adapter ------------------------------------------------


class _CharTokenizer:
    """One token per character, so token ids decode to known text."""

    def __init__(self, texts: list[str]):
        alphabet = sorted({ch for text in texts for ch in text})
        self._to_char = dict(enumerate(alphabet))
        self._to_id = {ch: ix for ix, ch in self._to_char.items()}

    def encode(self, text: str) -> list[int]:
        return [self._to_id[ch] for ch in text]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(self._to_char[int(ix)] for ix in ids)


def test_only_the_looping_row_of_a_batch_is_stopped(stopper):
    """transformers stops every row when a criterion returns a single True."""
    healthy = "S = " + " + ".join(f"a _ {{ {i} }}" for i in range(60))
    looping = r"y = a x + b , \quad ( 3 ) " + r"\ " * 200
    tokenizer = _CharTokenizer([healthy, looping])
    width = min(len(healthy), len(looping))
    input_ids = torch.tensor(
        [tokenizer.encode(healthy)[:width], tokenizer.encode(looping)[:width]]
    )
    criteria = StoppingCriteriaList([HFStoppingCriteriaWrapper(tokenizer, stopper)])

    is_done = criteria(input_ids, scores=None)

    assert is_done.tolist() == [False, True]
