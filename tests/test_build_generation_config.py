# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from transformers import GenerationConfig

from docling.models.utils.generation_utils import build_generation_config


def test_overrides_win_over_model_defaults() -> None:
    # extra_generation_config must override the model-level max_new_tokens,
    # matching the pre-refactor loose-kwarg ordering.
    config = build_generation_config(
        GenerationConfig(max_new_tokens=10),
        overrides={"max_new_tokens": 999, "num_logits_to_keep": 0},
        max_new_tokens=17,
    )
    assert config.max_new_tokens == 999
    assert config.num_logits_to_keep == 0  # custom entry preserved, not dropped


def test_explicit_sampling_wins_over_overrides() -> None:
    config = build_generation_config(
        GenerationConfig(),
        overrides={"do_sample": True, "temperature": 5.0},
        do_sample=False,
    )
    assert config.do_sample is False


def test_pad_token_falls_back_to_eos_only_when_unset() -> None:
    config = build_generation_config(GenerationConfig(), eos_token_id=123)
    assert config.pad_token_id == 123

    config = build_generation_config(
        GenerationConfig(), pad_token_id=7, eos_token_id=123
    )
    assert config.pad_token_id == 7
