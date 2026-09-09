# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from docling.backend.md_backend import MarkdownDocumentBackend
from docling.backend.msword_backend import MsWordDocumentBackend
from docling.datamodel.base_models import (
    ApiImageRequestResult,
    ConversionStatus,
    InputFormat,
    VlmStopReason,
)
from docling.datamodel.document import InputDocument
from docling.datamodel.extraction import TextContentItem
from docling.datamodel.extraction_options import ChannelSelection, ExtractionPromptStyle
from docling.datamodel.pipeline_options import VlmExtractionPipelineOptions
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.datamodel.vlm_model_specs import (
    GRANITE_VISION_4_1_API,
    GRANITE_VISION_4_1_TRANSFORMERS,
    NU_EXTRACT_2B_TRANSFORMERS,
    NU_EXTRACT_API,
)
from docling.exceptions import OperationNotAllowed
from docling.models.base_model import BaseVlmModel
from docling.models.extraction.api_extraction_model import ApiExtractionVlmModel
from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline

_MD_FIXTURE = Path("tests/data/md/sources/blocks.md")
_DOCX_FIXTURE = Path("tests/data/docx/sources/word_tables.docx")


def _pipeline_shell(
    spec, channel: ChannelSelection = ChannelSelection.AUTO, markdown_params=None
) -> ExtractionVlmPipeline:
    """A pipeline whose channel/text logic can run without loading a model."""
    pipeline = ExtractionVlmPipeline.__new__(ExtractionVlmPipeline)
    pipeline.pipeline_options = cast(
        VlmExtractionPipelineOptions,
        SimpleNamespace(
            vlm_options=spec,
            input_channels=channel,
            markdown_params=markdown_params,
        ),
    )
    return pipeline


def _input(path: Path, fmt: InputFormat, backend) -> InputDocument:
    return InputDocument(path_or_stream=path, format=fmt, backend=backend)


def test_nuextract_api_dispatches_to_extraction_model() -> None:
    pipeline = ExtractionVlmPipeline(
        VlmExtractionPipelineOptions(
            vlm_options=NU_EXTRACT_API, enable_remote_services=True
        )
    )
    assert isinstance(pipeline.vlm_model, ApiExtractionVlmModel)


def test_nuextract_api_requires_enable_remote_services() -> None:
    with pytest.raises(OperationNotAllowed):
        ExtractionVlmPipeline(
            VlmExtractionPipelineOptions(
                vlm_options=NU_EXTRACT_API, enable_remote_services=False
            )
        )


def test_auto_resolves_to_text_for_markdown() -> None:
    pipeline = _pipeline_shell(NU_EXTRACT_2B_TRANSFORMERS)
    in_doc = _input(_MD_FIXTURE, InputFormat.MD, MarkdownDocumentBackend)
    assert pipeline._resolve_channel(in_doc) == ChannelSelection.TEXT


def test_image_channel_on_text_format_is_loud_error() -> None:
    pipeline = _pipeline_shell(NU_EXTRACT_2B_TRANSFORMERS, ChannelSelection.IMAGE)
    in_doc = _input(_MD_FIXTURE, InputFormat.MD, MarkdownDocumentBackend)
    with pytest.raises(ValueError, match="does not offer page images"):
        pipeline._resolve_channel(in_doc)


def test_text_channel_on_granite_style_is_loud_error() -> None:
    pipeline = _pipeline_shell(GRANITE_VISION_4_1_API, ChannelSelection.TEXT)
    in_doc = _input(_MD_FIXTURE, InputFormat.MD, MarkdownDocumentBackend)
    with pytest.raises(ValueError, match="does not accept a text payload"):
        pipeline._resolve_channel(in_doc)


def test_image_and_text_on_text_only_format_is_loud_error() -> None:
    pipeline = _pipeline_shell(
        NU_EXTRACT_2B_TRANSFORMERS, ChannelSelection.IMAGE_AND_TEXT
    )
    in_doc = _input(_MD_FIXTURE, InputFormat.MD, MarkdownDocumentBackend)
    with pytest.raises(ValueError, match="does not offer page images"):
        pipeline._resolve_channel(in_doc)


def test_static_channel_capability_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="does not accept a text payload"):
        VlmExtractionPipelineOptions(
            vlm_options=GRANITE_VISION_4_1_TRANSFORMERS,
            input_channels=ChannelSelection.TEXT,
        )


def test_markdown_uses_normalized_source_text() -> None:
    pipeline = _pipeline_shell(NU_EXTRACT_2B_TRANSFORMERS)
    in_doc = _input(_MD_FIXTURE, InputFormat.MD, MarkdownDocumentBackend)
    assert pipeline._get_text_from_input(in_doc) == _MD_FIXTURE.read_text(
        encoding="utf-8"
    )


def test_docx_serialized_to_markdown() -> None:
    pipeline = _pipeline_shell(NU_EXTRACT_2B_TRANSFORMERS)
    in_doc = _input(_DOCX_FIXTURE, InputFormat.DOCX, MsWordDocumentBackend)
    text = pipeline._get_text_from_input(in_doc)
    assert text.strip()


def test_nuextract_request_carries_template_out_of_band(monkeypatch) -> None:
    from docling.utils import api_nuextract_request as mod

    captured: dict = {}

    def _post(**kwargs):
        captured.update(kwargs)
        return ApiImageRequestResult("{}", 0, VlmStopReason.END_OF_SEQUENCE)

    monkeypatch.setattr(mod, "_post_openai_chat_completion", _post)

    mod.api_nuextract_request(
        content_items=[TextContentItem(text="hello doc")],
        template='{"title": "string"}',
        url=cast(ApiVlmEngineOptions, NU_EXTRACT_API.engine_options).url,
        model="numind/NuExtract-2.0-8B",
    )

    payload = captured["payload"]
    assert payload["chat_template_kwargs"] == {"template": '{"title": "string"}'}
    content = payload["messages"][0]["content"]
    assert content == [{"type": "text", "text": "hello doc"}]
    assert all(item["type"] != "image_url" for item in content)
    assert payload["model"] == "numind/NuExtract-2.0-8B"


def test_text_extraction_maps_to_single_page() -> None:
    from docling.datamodel.base_models import VlmPrediction, VlmStopReason
    from docling.datamodel.extraction import ExtractionResult

    pipeline = _pipeline_shell(NU_EXTRACT_2B_TRANSFORMERS)
    seen: dict = {}

    class _StubModel:
        def process(self, requests, template):
            reqs = [list(r) for r in requests]
            seen["requests"] = reqs
            seen["template"] = template
            return [
                VlmPrediction(
                    text='{"title": "Duck"}', stop_reason=VlmStopReason.END_OF_SEQUENCE
                )
            ]

    pipeline.vlm_model = cast(BaseVlmModel, _StubModel())
    in_doc = _input(_MD_FIXTURE, InputFormat.MD, MarkdownDocumentBackend)
    ext_res = ExtractionResult(input=in_doc)

    pipeline._extract_via_text(ext_res, prompt='{"title": "string"}')

    assert len(ext_res.pages) == 1
    page = ext_res.pages[0]
    assert page.page_no == 1
    assert page.extracted_data == {"title": "Duck"}
    assert page.raw_text == '{"title": "Duck"}'
    assert len(seen["requests"][0]) == 1
    assert isinstance(seen["requests"][0][0], TextContentItem)


def test_api_failure_makes_text_extraction_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from docling.datamodel.extraction import ExtractionResult
    from docling.models.extraction import api_extraction_model as mod

    def _fail(**_kwargs):
        raise RuntimeError("service unavailable")

    monkeypatch.setattr(mod, "api_nuextract_request", _fail)
    pipeline = ExtractionVlmPipeline(
        VlmExtractionPipelineOptions(
            vlm_options=NU_EXTRACT_API,
            enable_remote_services=True,
        )
    )
    result = ExtractionResult(
        input=_input(_MD_FIXTURE, InputFormat.MD, MarkdownDocumentBackend)
    )

    pipeline._extract_via_text(result, prompt="{}")

    assert result.pages[0].errors == ["service unavailable"]
    assert pipeline._determine_status(result) == ConversionStatus.FAILURE
