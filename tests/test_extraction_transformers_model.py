# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import GenerationConfig

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import VlmStopReason
from docling.datamodel.extraction_options import (
    GRANITE_VISION_4_1_TRANSFORMERS,
    NU_EXTRACT_2B_TRANSFORMERS,
)
from docling.datamodel.pipeline_options import VlmExtractionPipelineOptions
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.models.extraction import transformers_extraction_model as module
from docling.models.extraction.transformers_extraction_model import (
    TransformersExtractionModel,
)
from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline

# These tests exercise the local Transformers extraction path and need torch and
# transformers importable, so CI runs them in the dedicated `vlm` suite.
pytestmark = pytest.mark.ml_vlm


class _LoadedModel:
    def eval(self) -> None:
        pass


@patch(
    "docling.models.extraction.transformers_extraction_model.TransformersExtractionModel",
)
def test_pipeline_routes_local_spec_to_transformers_model(mock_model_cls: Mock) -> None:
    """The local branch must build a TransformersExtractionModel from the spec
    without loading weights; the API branch is covered in test_extraction_api.py."""
    vlm_options = GRANITE_VISION_4_1_TRANSFORMERS.model_copy(
        update={
            "engine_options": TransformersVlmEngineOptions(
                device=AcceleratorDevice.CPU,
                torch_dtype="float16",
                trust_remote_code=False,
                compile_model=False,
            )
        }
    )
    ExtractionVlmPipeline(VlmExtractionPipelineOptions(vlm_options=vlm_options))

    mock_model_cls.assert_called_once()
    assert mock_model_cls.call_args.kwargs["vlm_options"] is vlm_options


def test_transformers_engine_options_control_model_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    processor_loader = Mock(
        return_value=SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))
    )
    model_loader = Mock(return_value=_LoadedModel())
    monkeypatch.setattr(module.AutoProcessor, "from_pretrained", processor_loader)
    monkeypatch.setattr(
        module.AutoModelForImageTextToText, "from_pretrained", model_loader
    )
    monkeypatch.setattr(module, "decide_device", lambda *_args, **_kwargs: "cpu")
    options = GRANITE_VISION_4_1_TRANSFORMERS.model_copy(
        update={
            "engine_options": TransformersVlmEngineOptions(
                device=AcceleratorDevice.CPU,
                torch_dtype="float16",
                trust_remote_code=False,
                compile_model=False,
            )
        }
    )

    TransformersExtractionModel(
        enabled=True,
        artifacts_path=tmp_path,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.MPS),
        vlm_options=options,
    )

    assert processor_loader.call_args.kwargs["trust_remote_code"] is False
    assert model_loader.call_args.kwargs["device_map"] == "cpu"
    assert model_loader.call_args.kwargs["dtype"] == "float16"
    assert model_loader.call_args.kwargs["trust_remote_code"] is False


@pytest.mark.parametrize(
    ("generated_tokens", "expected_reason"),
    [
        ([3, 4, 5], VlmStopReason.LENGTH),
        ([3, 99, 0], VlmStopReason.END_OF_SEQUENCE),
    ],
)
def test_transformers_reports_generation_stop_reason(
    generated_tokens: list[int], expected_reason: VlmStopReason
) -> None:
    model = TransformersExtractionModel.__new__(TransformersExtractionModel)
    model.model_spec = NU_EXTRACT_2B_TRANSFORMERS.model_spec.model_copy(
        update={"max_new_tokens": 3}
    )
    model.engine_options = TransformersVlmEngineOptions(use_kv_cache=False)
    model.max_new_tokens = 3
    model.temperature = 0.0
    model.generation_config = GenerationConfig(eos_token_id=99, pad_token_id=0)
    model.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(eos_token_id=99, pad_token_id=0),
        batch_decode=lambda *_args, **_kwargs: ["{}"],
    )
    model.vlm_model = SimpleNamespace(
        generate=lambda **_kwargs: torch.tensor([[1, 2, *generated_tokens]])
    )

    prediction = next(
        iter(model._generate_and_decode({"input_ids": torch.tensor([[1, 2]])}))
    )

    assert prediction.num_tokens == 3
    assert prediction.stop_reason == expected_reason


def test_transformers_rejects_input_over_context_limit() -> None:
    model = TransformersExtractionModel.__new__(TransformersExtractionModel)
    model.model_spec = NU_EXTRACT_2B_TRANSFORMERS.model_spec.model_copy(
        update={"max_input_tokens": 4}
    )
    model.engine_options = TransformersVlmEngineOptions(use_kv_cache=False)
    model.max_new_tokens = 3
    model.temperature = 0.0
    model.generation_config = GenerationConfig(eos_token_id=99, pad_token_id=0)

    def _fail_generate(**_kwargs):
        raise AssertionError("generate must not run when input exceeds the limit")

    model.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(eos_token_id=99, pad_token_id=0),
        batch_decode=lambda *_args, **_kwargs: ["{}"],
    )
    model.vlm_model = SimpleNamespace(generate=_fail_generate)

    with pytest.raises(ValueError, match="exceeding the configured context limit"):
        list(model._generate_and_decode({"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}))


@pytest.mark.parametrize("preparation", ["generic_chat", "nuextract", "nuextract_3"])
def test_ordered_local_content_routes_chat_and_processor_options(
    monkeypatch, preparation
):
    from copy import deepcopy

    from PIL import Image

    from docling.datamodel.extraction import (
        ExtractionTarget,
        ExtractionTemplate,
        ImageContentItem,
        TextContentItem,
    )
    from docling.datamodel.extraction_options import (
        GRANITE_VISION_4_1_SPEC,
        NUEXTRACT_2B_SPEC,
        NUEXTRACT_3_SPEC,
    )
    from docling.models.extraction import prompt_utils
    from docling.models.extraction.prompt_utils import prepare_target

    spec = {
        "generic_chat": GRANITE_VISION_4_1_SPEC,
        "nuextract": NUEXTRACT_2B_SPEC,
        "nuextract_3": NUEXTRACT_3_SPEC,
    }[preparation].model_copy(
        update={
            "extra_chat_template_kwargs": {
                "enable_thinking": False,
                "nested": {"call": True},
            },
            "extra_processor_kwargs": {"max_soft_tokens": 32},
        }
    )
    model = TransformersExtractionModel(
        False,
        None,
        AcceleratorOptions(),
        module.ExtractionVlmOptions(
            model_spec=spec, engine_options=TransformersVlmEngineOptions()
        ),
    )
    render = Mock(return_value="rendered")
    preprocess = Mock(return_value={"input_ids": torch.tensor([[1, 2]])})

    class Processor:
        tokenizer = SimpleNamespace(
            apply_chat_template=render
            if preparation == "nuextract"
            else Mock(side_effect=AssertionError("must render through processor"))
        )
        apply_chat_template = render

        def __call__(self, **kwargs):
            return preprocess(**kwargs)

    model.processor = Processor()
    model.device = "cpu"
    model._generate_and_decode = Mock(return_value=[SimpleNamespace(text="{}")])
    vision = Mock(return_value=["vision"])
    monkeypatch.setattr(prompt_utils, "_process_all_vision_info", vision)
    with Image.new("RGB", (3, 2)) as image:
        for field in ("total", "date"):
            target = prepare_target(
                ExtractionTarget(
                    template=ExtractionTemplate(
                        format="nuextract"
                        if preparation != "generic_chat"
                        else "example_json",
                        value={field: "verbatim-string"},
                    ),
                    instructions=f"Extract {field}",
                ),
                spec,
            )
            saved = deepcopy(target.chat_template_kwargs)
            content = [
                TextContentItem(text="before"),
                ImageContentItem(image=image),
                TextContentItem(text="after"),
            ]
            list(model.process([content], target))
            messages = render.call_args.args[0][0]["content"]
            assert [item["type"] for item in messages[:3]] == ["text", "image", "text"]
            assert messages[1]["image"] is image and messages[2]["text"] == "after"
            kwargs = render.call_args.kwargs
            assert kwargs["enable_thinking"] is False
            assert (
                kwargs["tokenize"] is False and kwargs["add_generation_prompt"] is True
            )
            assert "max_soft_tokens" not in kwargs
            if preparation != "generic_chat":
                assert (
                    f'"{field}"' in kwargs["template"]
                    and '"verbatim-string"' in kwargs["template"]
                    and f"Extract {field}" in kwargs["instructions"]
                )
                assert kwargs["template"] == target.chat_template_kwargs["template"]
                assert len(messages) == 3
            else:
                assert messages[3]["text"] == target.prompt and "template" not in kwargs
            assert preprocess.call_args.kwargs == {
                "text": ["rendered"],
                "images": ["vision"] if preparation == "nuextract" else [image],
                "padding": True,
                "return_tensors": "pt",
                "max_soft_tokens": 32,
            }
            kwargs["nested"]["call"] = False
            assert target.chat_template_kwargs == saved
        if preparation == "nuextract_3":
            vision.assert_not_called()
            list(model.process([[ImageContentItem(image=image)]], target))
            assert render.call_args.args[0][0]["content"] == [
                {"type": "image", "image": image}
            ]
            assert preprocess.call_args.kwargs["images"] == [image]
        # Text-only requests do not import or invoke vision preprocessing.
        vision.reset_mock()
        list(model.process([[TextContentItem(text="only text")]], target))
        assert preprocess.call_args.kwargs["images"] is None
        vision.assert_not_called()
    assert spec.extra_chat_template_kwargs == {
        "enable_thinking": False,
        "nested": {"call": True},
    }


@pytest.mark.parametrize(
    "style",
    [
        module.ExtractionPromptStyle.NUEXTRACT,
        module.ExtractionPromptStyle.GRANITE_VISION,
    ],
)
def test_main_inline_constructor_and_image_wrapper(style):
    import numpy as np

    from docling.datamodel.pipeline_options_vlm_model import (
        InferenceFramework,
        InlineVlmOptions,
        ResponseFormat,
    )
    from docling.models.extraction.nuextract_transformers_model import (
        NuExtractTransformersModel,
    )
    from docling.models.extraction.prompt_utils import _PreparedTarget

    inline = InlineVlmOptions(
        repo_id="test/model",
        prompt="",
        inference_framework=InferenceFramework.TRANSFORMERS,
        response_format=ResponseFormat.PLAINTEXT,
        extra_processor_kwargs={"size": 12},
    )
    model = TransformersExtractionModel(
        False, None, AcceleratorOptions(), inline, style
    )
    assert model.vlm_options is inline
    captured = []

    def process(requests, targets):
        captured.append((requests, targets))
        return [SimpleNamespace(text="{}") for _ in requests]

    model._process_prepared = process
    result = list(
        model.process_images(
            [np.zeros((3, 2, 4), dtype=np.uint8), np.zeros((3, 2), dtype=np.uint8)],
            ["final one", "final two"],
        )
    )
    assert len(result) == 2
    requests, targets = captured[0]
    assert all(req[0].image.mode == "RGB" for req in requests)
    assert all(isinstance(target, _PreparedTarget) for target in targets)
    if style == module.ExtractionPromptStyle.NUEXTRACT:
        assert [target.chat_template_kwargs["template"] for target in targets] == [
            "final one",
            "final two",
        ]
    else:
        assert [target.prompt for target in targets] == ["final one", "final two"]
        assert targets[0].processor_kwargs == {"do_pad": True, "size": 12}
    assert (
        NuExtractTransformersModel(
            False, None, AcceleratorOptions(), inline
        ).model_spec.preparation
        == "nuextract"
    )
    with pytest.raises(ValueError, match="must match"):
        list(model.process_images([np.zeros((2, 2))], ["one", "two"]))
    with pytest.raises(ValueError, match="Unsupported numpy"):
        list(model.process_images([np.zeros((2, 2, 5))], "prompt"))


@pytest.mark.parametrize(
    "key", ["text", "images", "padding", "return_tensors", "template", "instructions"]
)
def test_local_processor_request_collisions_fail_before_preprocessing(key):
    from docling.datamodel.extraction import TextContentItem
    from docling.models.extraction.prompt_utils import prepare_legacy_target

    options = GRANITE_VISION_4_1_TRANSFORMERS.model_copy(deep=True)
    options.model_spec.extra_processor_kwargs[key] = "static"
    model = TransformersExtractionModel(False, None, AcceleratorOptions(), options)
    model.device = "cpu"
    processor = Mock()
    processor.apply_chat_template.return_value = "rendered"
    model.processor = processor
    with pytest.raises(ValueError, match="request-owned"):
        list(
            model.process(
                [[TextContentItem(text="text")]],
                prepare_legacy_target("{}", options.model_spec),
            )
        )
    processor.assert_not_called()
