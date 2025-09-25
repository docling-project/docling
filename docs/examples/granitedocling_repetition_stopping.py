# %% [markdown]
# Experimental VLM pipeline with custom repetition stopping criteria.
#
# This script demonstrates the use of custom stopping criteria that detect
# repetitive location coordinate patterns in generated text and stop generation
# when such patterns are found.
#
# What this example does
# - Uses the GraniteDocling model with custom repetition stopping criteria injected
# - Processes a PDF document or image and monitors for repetitive coordinate patterns
# - Stops generation early when repetitive patterns are detected


# %%

import logging
import re
from collections import defaultdict
from typing import List, Optional

import torch
from transformers import StoppingCriteria

from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


class RepetitionStoppingCriteria(StoppingCriteria):
    """
    Detects repetitive <tag>...<loc_x><loc_y><loc_w><loc_h>text</tag> blocks,
    but only when repeats are **consecutive** and both tag & inner text are identical.

    Performance:
    - Heavy check runs every N calls (default 32).
    - Only decodes the last LOOKBACK_TOKENS tokens per sequence (default 200).
    """

    N: int = 32
    LOOKBACK_TOKENS: int = 200

    def __init__(
        self,
        tokenizer,
        *,
        N: Optional[int] = None,
        lookback_tokens: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        if N is not None:
            self.N = int(N)
        if lookback_tokens is not None:
            self.LOOKBACK_TOKENS = max(1, int(lookback_tokens))

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

        self._call_count = 0

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        self._call_count += 1
        if self._call_count % self.N != 0:
            return False

        for seq in input_ids:
            try:
                text = self.tokenizer.decode(
                    seq[-self.LOOKBACK_TOKENS :], skip_special_tokens=False
                )
                if self.should_stop(text):
                    _log.info(
                        "Stopping generation early due to consecutive repetitive element blocks"
                    )
                    return True
            except Exception as e:
                _log.warning(f"Error decoding sequence for repetition check: {e}")
                continue
        return False

    # --- small helper ---
    def _regular(self, vals: List[int]) -> bool:
        """3+ strictly increasing values with ~regular spacing (±20%)."""
        if len(vals) < 3:
            return False
        diffs = [b - a for a, b in zip(vals, vals[1:])]
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


# Set up logging to see when repetition stopping is triggered
logging.basicConfig(level=logging.INFO)

# Replace with a local path if preferred.
# source = "https://ibm.biz/docling-page-with-table" # Example that shows no repetitions.
source = "tests/data_scanned/old_newspaper.png"  # Example that creates repetitions.

print(f"📄 Processing document: {source}")

###### USING GRANITEDOCLING WITH CUSTOM REPETITION STOPPING

# Create custom VLM options with repetition stopping criteria
custom_vlm_options = vlm_model_specs.GRANITEDOCLING_TRANSFORMERS.model_copy()

# Inject the repetition stopping criteria factory
# The factory will be called with the tokenizer when the model is initialized
custom_vlm_options.custom_stopping_criteria = [RepetitionStoppingCriteria]

pipeline_options = VlmPipelineOptions(
    vlm_options=custom_vlm_options,
)

converter = DocumentConverter(
    format_options={
        InputFormat.IMAGE: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        ),
    }
)

doc = converter.convert(source=source).document

print(doc.export_to_markdown())
