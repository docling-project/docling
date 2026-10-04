# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import itertools
import logging
import re
import sys
from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, List

from typing_extensions import override

if TYPE_CHECKING:
    from transformers import GenerationConfig

_log = logging.getLogger(__name__)


class GenerationStopper:
    """
    Base interface for stopping logic.
    - should_stop(s): True to stop given the current decoded text window.
    - lookback_tokens(): how many tokens should be considered (default: sys.maxsize).
    """

    @abstractmethod
    def should_stop(self, s: str) -> bool:
        pass

    def lookback_tokens(self) -> int:
        return sys.maxsize


def build_generation_config(
    base_config: "GenerationConfig | None",
    *,
    overrides: dict[str, Any] | None = None,
    max_new_tokens: int | None = None,
    use_cache: bool | None = None,
    do_sample: bool | None = None,
    temperature: float | None = None,
    pad_token_id: int | list[int] | None = None,
    eos_token_id: int | list[int] | None = None,
) -> "GenerationConfig":
    from transformers import GenerationConfig

    config = deepcopy(base_config) if base_config is not None else GenerationConfig()

    # Model-level defaults, kept at the lowest precedence so caller overrides
    # still win over them -- matching the previous behavior where these were
    # spread as loose generate() kwargs *before* extra_generation_config.
    if max_new_tokens is not None:
        config.max_new_tokens = max_new_tokens
    if use_cache is not None:
        config.use_cache = use_cache

    # Default pad_token_id to the tokenizer's, falling back to eos_token_id, to
    # silence the transformers pad_token_id warning. This only fills a gap, so a
    # caller override below can still replace it.
    if pad_token_id is not None:
        config.pad_token_id = pad_token_id
    elif config.pad_token_id is None and eos_token_id is not None:
        config.pad_token_id = eos_token_id

    # Caller overrides (e.g. vlm_options.extra_generation_config) win over the
    # model defaults above. update() validates and preserves custom entries such
    # as num_logits_to_keep, unlike a raw setattr loop.
    if overrides:
        config.update(allow_custom_entries=True, **overrides)

    # The explicit sampling decision has the final say, matching the old ordering
    # where do_sample/temperature were set after the override spread.
    if do_sample is not None:
        config.do_sample = do_sample
    if temperature is not None:
        config.temperature = temperature

    return config


def find_repeated_tail(
    text: str, *, min_repeats: int, min_span: int, max_unit: int
) -> int | None:
    """Length of the unit that `text` keeps repeating at its end, if any.

    The tail counts as repeated when its last ``max(min_span, unit * min_repeats)``
    characters are periodic with period ``unit``, for some ``unit`` of at most
    ``max_unit`` characters. Periodicity is checked with a single slice
    comparison per candidate unit, and does not need the text to end on a unit
    boundary, so a generation interrupted mid-unit is still recognised.

    Returns:
        The shortest such unit length, or None if the tail does not repeat.
    """
    for unit in range(1, max_unit + 1):
        span = max(min_span, unit * min_repeats)
        if span > len(text):
            break
        tail = text[-span:]
        if tail[unit:] == tail[:-unit]:
            return unit
    return None


def strip_repeated_tail(text: str, unit: int) -> str:
    """Remove the tail of `text` that repeats every `unit` characters.

    The periodic run is extended as far back as it goes, so every copy of the
    unit goes, not only those in the detection window. Trailing whitespace is
    dropped, and so is a lone backslash whose escaped character was part of the
    removed run.
    """
    start = len(text) - unit
    while start > 0 and text[start - 1] == text[start - 1 + unit]:
        start -= 1
    kept = text[:start].rstrip()
    while kept.endswith("\\") and (len(kept) - len(kept.rstrip("\\"))) % 2 == 1:
        kept = kept[:-1].rstrip()
    return kept


class TailRepetitionStopper(GenerationStopper):
    """Stops a generation whose output ends in the same short unit, over and over.

    Code and formula models occasionally fall into a loop after the content they
    were asked for, e.g. emitting ``\\quad \\quad \\quad ...`` or ``\\ \\ \\ ...``
    until ``max_new_tokens``. Legitimate LaTeX and code repeat too (runs of
    ``\\quad``, matrix cells, separator lines), so a loop only counts once the
    same unit of at most `max_unit` characters has been repeated at least
    `min_repeats` times and covers at least `min_span` characters.

    The check is a handful of slice comparisons on the decoded tail, cheap
    enough to run at every generation step.
    """

    def __init__(
        self,
        *,
        min_repeats: int = 16,
        min_span: int = 320,
        max_unit: int = 64,
        lookback_tokens: int = 640,
    ):
        self.min_repeats = max(2, int(min_repeats))
        self.min_span = max(1, int(min_span))
        self.max_unit = max(1, int(max_unit))
        self._lookback_tokens = max(1, int(lookback_tokens))

    @override
    def lookback_tokens(self) -> int:
        return self._lookback_tokens

    def repeated_unit(self, text: str) -> int | None:
        """Length of the unit `text` keeps repeating at its end, if any."""
        return find_repeated_tail(
            text,
            min_repeats=self.min_repeats,
            min_span=self.min_span,
            max_unit=self.max_unit,
        )

    @override
    def should_stop(self, s: str) -> bool:
        return self.repeated_unit(s) is not None

    def strip(self, text: str) -> str:
        """`text` without its repeated tail; unchanged if the tail does not repeat."""
        unit = self.repeated_unit(text)
        if unit is None:
            return text
        return strip_repeated_tail(text, unit)


class DocTagsRepetitionStopper(GenerationStopper):
    """
    Detects repetitive <tag>...<loc_x><loc_y><loc_w><loc_h>text</tag> blocks,
    but only when repeats are **consecutive** and both tag & inner text are identical.

    Performance:
    - Heavy check runs every N calls (default 32).
    - Only decodes the last LOOKBACK_TOKENS tokens per sequence (default 200).
    """

    def __init__(self, *, N: int = 32, lookback_tokens: int = 200):
        self.N = max(1, int(N))
        self._lookback_tokens = max(1, int(lookback_tokens))
        self._call_count = 0

        # <tag> ... <loc_x><loc_y><loc_w><loc_h> text ... </tag>
        self._PATTERN = re.compile(
            r"""
            <(?P<tag>[a-zA-Z0-9_]+)>\s*
            (?P<prefix>.*?)?
            <loc_(?P<x>\d+)><loc_(?P<y>\d+)><loc_(?P<w>\d+)><loc_(?P<h>\d+)>
            (?P<text>.*?)
            </(?P=tag)>
            """,
            re.DOTALL | re.VERBOSE,
        )

    # --- small helper ---
    def _regular(self, vals: List[int]) -> bool:
        """3+ strictly increasing values with ~regular spacing (±20%)."""
        if len(vals) < 3:
            return False
        diffs = [b - a for a, b in itertools.pairwise(vals)]
        if any(d <= 0 for d in diffs):
            return False
        mean = sum(diffs) / len(diffs)
        tol = 0.2 * mean
        return all(abs(d - mean) <= tol for d in diffs)

    @override
    def should_stop(self, s: str) -> bool:
        """
        Trip only on **consecutive** runs (no other matched blocks between) of ≥3 items
        with the same <tag> and identical inner text, where within that run we see:
          - any exact duplicate (x,y,w,h), or
          - stable X/W with regular Y progression, or
          - stable Y/H with regular X progression.
        """
        # Stream matches and evaluate runs on-the-fly to stay compact and fast.
        prev_tag = prev_text = None
        run = []  # list of (x,y,w,h)

        def run_repetitive(boxes: List[tuple[int, int, int, int]]) -> bool:
            if len(boxes) < 3:
                return False
            # duplicates?
            if len(set(boxes)) < len(boxes):
                return True
            xs, ys, ws, hs = zip(*boxes)
            x_stable = all(x == xs[0] for x in xs)
            y_stable = all(y == ys[0] for y in ys)
            w_stable = all(w == ws[0] for w in ws)
            h_stable = all(h == hs[0] for h in hs)
            # horizontal (down the page): X/W stable, Y regular
            if (x_stable or w_stable) and self._regular(list(ys)):
                return True
            # vertical (across): Y/H stable, X regular
            if (y_stable or h_stable) and self._regular(list(xs)):
                return True
            return False

        for m in self._PATTERN.finditer(s):
            tag, text = m.group("tag"), m.group("text")
            box = (
                int(m.group("x")),
                int(m.group("y")),
                int(m.group("w")),
                int(m.group("h")),
            )

            if prev_tag == tag and prev_text == text:
                run.append(box)  # consecutive same-tag+text
            else:
                # evaluate previous run before starting a new one
                if run_repetitive(run):
                    return True
                prev_tag, prev_text = tag, text
                run = [box]

        # check the last run
        return run_repetitive(run)
