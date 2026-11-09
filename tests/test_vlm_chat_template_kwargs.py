# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT
# Typhoon OCR V2 by the Typhoon team: https://huggingface.co/typhoon-ai/typhoon-ocr2-2b

"""The in-process engines pass the chat template kwargs of the model spec.

_TEMPLATE keeps the generation prompt of the Typhoon OCR V2 chat template
(the Qwen3.5 template, chat_template.jinja lines 142-149). It opens a thinking
block unless enable_thinking is False. A real transformers tokenizer renders it
with Jinja. The processor around that tokenizer, the model and the vLLM runtime
are stand-ins.
"""

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.models.inference_engines.vlm.base import VlmEngineInput
from docling.models.inference_engines.vlm.transformers_engine import (
    TransformersVlmEngine,
)
from docling.models.inference_engines.vlm.vllm_engine import VllmVlmEngine

_TEMPLATE = r"""
{%- for message in messages %}
    {{- '<|im_start|>' + message.role + '\n' }}
    {%- for content in message.content %}
        {%- if content.type == 'image' %}
            {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
        {%- else %}
            {{- content.text }}
        {%- endif %}
    {%- endfor %}
    {{- '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n\n' }}
    {%- endif %}
{%- endif %}"""

_PROMPT = "Parse the page."

# The image comes first, then the text, in one user message.
_USER_TURN = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    f"{_PROMPT}<|im_end|>\n<|im_start|>assistant\n"
)

_THINKING_CASES = [
    pytest.param({}, "<think>\n\n", id="template-default"),
    pytest.param(
        {"enable_thinking": False}, "<think>\n\n</think>\n\n", id="thinking-off"
    ),
]


class _ChatTemplateProcessor:
    """Processor around a real tokenizer that renders _TEMPLATE.

    Like a transformers processor, apply_chat_template returns text unless the
    caller asks for tokens. The processor records the text that it tokenizes.
    """

    def __init__(self) -> None:
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(
                models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")
            ),
            chat_template=_TEMPLATE,
        )
        self.tokenized_text: list[str] = []

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("tokenize", False)
        return self.tokenizer.apply_chat_template(conversation, **kwargs)

    def __call__(self, text: list[str], images: Any, **kwargs: Any) -> dict:
        self.tokenized_text.extend(text)
        return {"input_ids": torch.zeros((len(text), 1), dtype=torch.long)}

    def batch_decode(self, sequences: torch.Tensor, **kwargs: Any) -> list[str]:
        return ["[]"] * len(sequences)


class _OneTokenModel:
    """Stand-in for a transformers model: it adds one token to each row."""

    def generate(self, input_ids: torch.Tensor, generation_config: Any):
        return torch.cat([input_ids, torch.ones_like(input_ids[:, :1])], dim=-1)


class _RecordingLlm:
    """Stand-in for vllm.LLM: it records the prompts that it gets."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, llm_inputs: list[dict], sampling_params: Any = None):
        self.prompts.extend(item["prompt"] for item in llm_inputs)
        completion = SimpleNamespace(
            text="[]", token_ids=[1], finish_reason="stop", stop_reason=None
        )
        return [SimpleNamespace(outputs=[completion]) for _ in llm_inputs]


def _engine_input(chat_template_kwargs: dict[str, Any]) -> VlmEngineInput:
    return VlmEngineInput(
        image=Image.new("RGB", (8, 8), "white"),
        prompt=_PROMPT,
        chat_template_kwargs=chat_template_kwargs,
    )


@pytest.mark.parametrize(("chat_template_kwargs", "generation_prompt"), _THINKING_CASES)
def test_transformers_engine_passes_chat_template_kwargs(
    chat_template_kwargs: dict[str, Any], generation_prompt: str
) -> None:
    processor = _ChatTemplateProcessor()
    engine = TransformersVlmEngine(
        options=TransformersVlmEngineOptions(),
        accelerator_options=AcceleratorOptions(device="cpu"),
        artifacts_path=None,
    )
    engine._initialized = True
    engine.device = "cpu"
    engine.processor = processor
    engine.vlm_model = _OneTokenModel()  # type: ignore[assignment]

    engine.predict_batch([_engine_input(chat_template_kwargs)])

    assert processor.tokenized_text == [_USER_TURN + generation_prompt]


@pytest.mark.parametrize(("chat_template_kwargs", "generation_prompt"), _THINKING_CASES)
def test_vllm_engine_passes_chat_template_kwargs(
    chat_template_kwargs: dict[str, Any], generation_prompt: str
) -> None:
    llm = _RecordingLlm()
    engine = VllmVlmEngine.__new__(VllmVlmEngine)
    engine._initialized = True
    engine.llm = llm
    engine.processor = _ChatTemplateProcessor()
    engine.sampling_params = SimpleNamespace()
    engine.model_config = SimpleNamespace(repo_id="org/model")

    # vLLM is optional. predict_batch only imports SamplingParams from it.
    vllm = SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs))
    with patch.dict(sys.modules, {"vllm": vllm}):
        engine.predict_batch([_engine_input(chat_template_kwargs)])

    assert llm.prompts == [_USER_TURN + generation_prompt]
