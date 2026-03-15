from pathlib import Path

import pytest
import torch
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import PictureDescriptionVlmOptions
from docling.models.stages.picture_description.picture_description_vlm_model import (
    PictureDescriptionVlmModel,
)


class _DummyBatch(dict):
    def to(self, device):
        self["device"] = device
        return self


class _DummyProcessor:
    def __init__(self) -> None:
        self.template_calls = 0
        self.process_calls = []
        self.decode_calls = 0

    def apply_chat_template(self, messages, add_generation_prompt=True):
        self.template_calls += 1
        self.messages = messages
        self.add_generation_prompt = add_generation_prompt
        return "formatted prompt"

    def __call__(self, *, text, images, return_tensors, padding):
        self.process_calls.append(
            {
                "text": text,
                "images": images,
                "return_tensors": return_tensors,
                "padding": padding,
            }
        )
        return _DummyBatch(
            {
                "input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            }
        )

    def batch_decode(self, token_ids, *, skip_special_tokens):
        self.decode_calls += 1
        self.token_ids = token_ids
        self.skip_special_tokens = skip_special_tokens
        return ["first description", "second description"]


class _DummyModel:
    def __init__(self) -> None:
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return torch.tensor(
            [
                [1, 2, 3, 10, 11],
                [1, 2, 3, 20, 21],
            ]
        )


class _DummyTokenizer:
    def __init__(self) -> None:
        self.padding_side = "left"


class _InitDummyProcessor:
    def __init__(self) -> None:
        self.tokenizer = _DummyTokenizer()


class _InitDummyModel:
    def eval(self):
        return self


def test_legacy_picture_description_vlm_batches_generation() -> None:
    model = PictureDescriptionVlmModel.__new__(PictureDescriptionVlmModel)
    model.processor = _DummyProcessor()
    model.model = _DummyModel()
    model.device = "cpu"
    model.options = PictureDescriptionVlmOptions(
        repo_id="org/model",
        prompt="Describe this image in a few sentences.",
        generation_config={"max_new_tokens": 17, "do_sample": False},
    )

    images = [
        Image.new("RGB", (8, 8), "white"),
        Image.new("RGB", (10, 10), "black"),
    ]

    outputs = list(model._annotate_images(images))

    assert outputs == ["first description", "second description"]
    assert model.processor.template_calls == 1
    assert len(model.processor.process_calls) == 1
    assert model.processor.process_calls[0]["text"] == [
        "formatted prompt",
        "formatted prompt",
    ]
    assert model.processor.process_calls[0]["images"] == images
    assert model.processor.process_calls[0]["return_tensors"] == "pt"
    assert model.processor.process_calls[0]["padding"] is True
    assert model.processor.decode_calls == 1
    assert model.processor.skip_special_tokens is True
    assert len(model.model.generate_calls) == 1
    assert model.model.generate_calls[0]["generation_config"].max_new_tokens == 17


def test_legacy_picture_description_vlm_skips_empty_batch() -> None:
    model = PictureDescriptionVlmModel.__new__(PictureDescriptionVlmModel)
    model.processor = _DummyProcessor()
    model.model = _DummyModel()
    model.device = "cpu"
    model.options = PictureDescriptionVlmOptions(repo_id="org/model")

    assert list(model._annotate_images([])) == []
    assert model.processor.template_calls == 0
    assert model.processor.process_calls == []
    assert model.processor.decode_calls == 0
    assert model.model.generate_calls == []


def test_legacy_picture_description_vlm_init_uses_configured_padding_side(
    monkeypatch,
) -> None:
    transformers = pytest.importorskip("transformers")
    processor = _InitDummyProcessor()
    model = _InitDummyModel()

    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: processor,
    )
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText,
        "from_pretrained",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(torch, "compile", lambda compiled_model: compiled_model)

    picture_description_model = PictureDescriptionVlmModel(
        enabled=True,
        enable_remote_services=False,
        artifacts_path=Path("/tmp"),
        options=PictureDescriptionVlmOptions(
            repo_id="org/model",
            padding_side="right",
        ),
        accelerator_options=AcceleratorOptions(device="cpu"),
    )

    assert processor.tokenizer.padding_side == "right"
    assert picture_description_model.processor is processor
    assert picture_description_model.model is model
