# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging

import torch
from transformers import StoppingCriteria

from docling.models.utils.generation_utils import GenerationStopper

_log = logging.getLogger(__name__)


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

    def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
        """Flag the rows whose decoded tail trips the stopper.

        One flag per row, so transformers finishes only those rows and lets the
        rest of the batch run on. A single ``True`` would stop every row.
        """
        lb = max(1, int(self.stopper.lookback_tokens()))
        is_done = torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )
        for row, seq in enumerate(input_ids):  # (batch, seq_len)
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
                        "HF wrapper: stopping row %s due to %s.should_stop==True",
                        row,
                        type(self.stopper).__name__,
                    )
                    is_done[row] = True
            except Exception as e:
                _log.info(f"Error in TextStopper.should_stop: {e}")
                continue
        return is_done  # type: ignore[return-value]
