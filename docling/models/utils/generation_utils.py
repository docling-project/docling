import itertools
import logging
import re
import sys
from abc import abstractmethod
from typing import List

from transformers import StoppingCriteria

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

        def run_repetitive(boxes: List[tuple]) -> bool:
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


class HFStoppingCriteriaWrapper(StoppingCriteria):
    """
    Adapts any GenerationStopper to HuggingFace Transformers.
    Decodes exactly min(seq_len, stopper.lookback_tokens()) tokens from the end.
    """

    def __init__(
        self,
        tokenizer,
        stopper: GenerationStopper,
        *,
        skip_special_tokens: bool = False,
    ):
        self.tokenizer = tokenizer
        self.stopper = stopper
        self.skip_special_tokens = skip_special_tokens

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        lb = max(1, int(self.stopper.lookback_tokens()))
        for seq in input_ids:  # (batch, seq_len)
            window = seq[-lb:]  # slicing handles lb > len(seq)
            try:
                text = self.tokenizer.decode(
                    window, skip_special_tokens=self.skip_special_tokens
                )
            except Exception as e:
                _log.info(f"Decoding failed for stopping check: {e}")
                continue

            try:
                if self.stopper.should_stop(text):
                    _log.info(
                        "HF wrapper: stopping due to TextStopper.should_stop==True"
                    )
                    return True
            except Exception as e:
                _log.info(f"Error in TextStopper.should_stop: {e}")
                continue
        return False
