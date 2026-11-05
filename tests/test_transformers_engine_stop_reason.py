# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The stop reason that the Transformers engine reports for a GenerationStopper.

A stopper declares with ``is_valid_end()`` whether its stop marks the valid end
of the output or cuts it. The engine reports the reason of the stopper that
ended each row of the batch, also when transformers keeps calling the stoppers
on the padding of rows that already finished.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import VlmStopReason
from docling.datamodel.pipeline_options_vlm_model import TransformersPromptStyle
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.models.inference_engines.vlm.base import VlmEngineInput
from docling.models.inference_engines.vlm.transformers_engine import (
    TransformersVlmEngine,
)
from docling.models.utils.generation_utils import (
    GenerationStopper,
    TailRepetitionStopper,
)

_PAD = "_"


class _ClosingTagStopper(GenerationStopper):
    """Stops when the model closes the document: a valid end of the output."""

    def should_stop(self, s: str) -> bool:
        return "</doctag>" in s

    def is_valid_end(self) -> bool:
        return True


class _CharProcessor:
    """Processor and tokenizer in one: one token per character."""

    pad_token = _PAD

    def __init__(self, texts: list[str]):
        alphabet = sorted({ch for text in texts for ch in text} | {_PAD, "P"})
        self._to_char = dict(enumerate(alphabet))
        self._to_id = {ch: ix for ix, ch in self._to_char.items()}
        self.pad_token_id = self._to_id[_PAD]

    def encode(self, text: str) -> list[int]:
        return [self._to_id[ch] for ch in text]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(self._to_char[int(ix)] for ix in ids)

    def batch_decode(self, sequences, **kwargs) -> list[str]:
        return [self.decode(seq).rstrip(_PAD) for seq in sequences]

    def __call__(self, text, images, **kwargs):
        return {"input_ids": torch.tensor([self.encode("P") for _ in text])}


class _ScriptedModel:
    """Emits a fixed continuation per row, the way transformers runs the loop.

    A row that ran out of text emits padding, as after an end-of-sequence
    token. The stopping criteria run on every row at every step, also on rows
    that already finished.
    """

    def __init__(self, processor: _CharProcessor, rows: list[str]):
        self._processor = processor
        self._rows = [processor.encode(row) for row in rows]

    def generate(self, input_ids, generation_config, stopping_criteria=None):
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool)
        for step in range(max(len(row) for row in self._rows) + 32):
            next_tokens = torch.tensor(
                [
                    self._processor.pad_token_id
                    if finished[ix] or step >= len(row)
                    else row[step]
                    for ix, row in enumerate(self._rows)
                ]
            )
            finished |= torch.tensor([step >= len(row) for row in self._rows])
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if stopping_criteria is not None:
                finished |= stopping_criteria(input_ids, scores=None)
            if bool(finished.all()):
                break
        return input_ids


def _run(
    rows: list[str], stoppers: list[GenerationStopper], *, max_new_tokens: int = 128
):
    processor = _CharProcessor(rows)
    engine = TransformersVlmEngine(
        options=TransformersVlmEngineOptions(),
        accelerator_options=AcceleratorOptions(device="cpu"),
        artifacts_path=None,
    )
    engine._initialized = True
    engine.device = "cpu"
    engine.processor = processor
    engine.vlm_model = _ScriptedModel(processor, rows)  # type: ignore[assignment]
    image = Image.new("RGB", (8, 8), "white")
    return engine.predict_batch(
        [
            VlmEngineInput(
                image=image,
                prompt="Prompt",
                max_new_tokens=max_new_tokens,
                extra_generation_config={
                    "transformers_prompt_style": TransformersPromptStyle.RAW,
                    "custom_stopping_criteria": stoppers,
                },
            )
            for _ in rows
        ]
    )


@pytest.fixture
def repetition_stopper() -> TailRepetitionStopper:
    return TailRepetitionStopper(
        min_repeats=4, min_span=8, max_unit=4, lookback_tokens=64
    )


def test_each_row_reports_the_reason_of_the_stopper_that_ended_it(
    repetition_stopper,
) -> None:
    """The closing tag is a valid end, the loop is a cut output.

    The last row ends on its own and then gets padded while the other rows
    run on. That padding repeats, so the repetition stopper fires on it, but
    the row had already ended: it must not be reported as a cut output.
    """
    rows = [
        "<doctag>text</doctag>more text that the stopper never lets through",
        "<doctag>" + "ab" * 30,
        "plain",
    ]

    outputs = _run(rows, [_ClosingTagStopper(), repetition_stopper])

    assert outputs[0].text == "<doctag>text</doctag>"
    assert outputs[0].stop_reason == VlmStopReason.END_OF_SEQUENCE.value
    assert outputs[1].text.startswith("<doctag>abab")
    assert len(outputs[1].text) < len(rows[1])
    assert outputs[1].stop_reason == VlmStopReason.STOP_SEQUENCE.value
    # The repetition stopper fired only on the padding after this row ended.
    # No stopper ended it, so the engine does not know why it ended.
    assert outputs[2].text == "plain"
    assert outputs[2].stop_reason == VlmStopReason.UNSPECIFIED.value


@pytest.mark.parametrize("valid_end", [False, True])
def test_stopper_at_token_limit_keeps_declared_reason(
    repetition_stopper, valid_end: bool
) -> None:
    text = "</doctag>" if valid_end else "abababab"
    stopper = _ClosingTagStopper() if valid_end else repetition_stopper

    (output,) = _run([text], [stopper], max_new_tokens=len(text))

    assert output.text == text
    assert output.stop_reason == (
        VlmStopReason.END_OF_SEQUENCE.value
        if valid_end
        else VlmStopReason.STOP_SEQUENCE.value
    )
