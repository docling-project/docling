# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Source-to-item acceptance gates, without weights or external services."""

import json
import warnings
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    ImageRef,
    PageItem,
    ProvenanceItem,
    Size,
)
from PIL import Image
from pydantic import BaseModel, Field

from docling.datamodel.base_models import (
    ApiImageRequestResult,
    ConversionStatus,
    InputFormat,
    VlmStopReason,
)
from docling.datamodel.extraction import (
    DocumentExtractionResult,
    DocumentScope,
    ExtractedPageData,
    ExtractionResult,
    ExtractionTarget,
    ExtractionTemplate,
    PageScope,
)
from docling.datamodel.extraction_options import (
    GRANITE_VISION_4_1_API,
    NU_EXTRACT_API,
    ChannelSelection,
    ExtractionVlmOptions,
)
from docling.datamodel.pipeline_options import VlmExtractionPipelineOptions
from docling.datamodel.settings import DEFAULT_PAGE_RANGE
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.document_extractor import DocumentExtractor, ExtractionFormatOption
from docling.exceptions import ConversionError
from docling.models.extraction import api_extraction_model, prompt_utils
from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline
from docling.utils import api_image_request


class _Output(BaseModel):
    total: float = Field(description="Final total", examples=[13.0])
    note: str | None = None


def _target() -> ExtractionTarget:
    return ExtractionTarget.from_pydantic(_Output)


def _archive(tmp_path: Path, *, missing_image=False) -> Path:
    doc = DoclingDocument(name="source")
    for n in (1, 2, 3):
        image = Image.new("RGB", (32, 24), (n, 0, 0))
        doc.pages[n] = PageItem(
            page_no=n,
            size=Size(width=32, height=24),
            image=None
            if missing_image and n == 3
            else ImageRef.from_pil(image, dpi=72),
        )
        text = f"Page {n} total {n * 10}"
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=text,
            prov=ProvenanceItem(
                page_no=n,
                bbox=BoundingBox(l=0, t=0, r=32, b=24),
                charspan=(0, len(text)),
            ),
        )
    path = tmp_path / "source.dclx"
    doc.save_as_doclang_archive(path)
    return path


def _extractor(channel=ChannelSelection.AUTO, options=NU_EXTRACT_API, timeout=None):
    pipeline_options = VlmExtractionPipelineOptions(
        vlm_options=options,
        input_channels=channel,
        enable_remote_services=True,
        document_timeout=timeout,
    )
    return DocumentExtractor(
        extraction_format_options={
            fmt: ExtractionFormatOption(
                pipeline_cls=ExtractionVlmPipeline, pipeline_options=pipeline_options
            )
            for fmt in (
                InputFormat.DCLX,
                InputFormat.MD,
                InputFormat.PDF,
                InputFormat.IMAGE,
            )
        }
    )


def _responses(monkeypatch, responses=None):
    calls = []
    answers = iter(responses) if responses is not None else None

    def request(**kwargs):
        calls.append(kwargs)
        answer = next(answers) if answers is not None else '{"total": 42}'
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, tuple):
            text, stop = answer
        else:
            text, stop = answer, VlmStopReason.END_OF_SEQUENCE
        return ApiImageRequestResult(text, 7, stop, {"completion_tokens": 7})

    monkeypatch.setattr(api_extraction_model, "api_extraction_request", request)
    return calls


@pytest.mark.parametrize("channel", list(ChannelSelection))
def test_absolute_pages_are_independent_for_every_channel(
    tmp_path, monkeypatch, channel
):
    source = _archive(tmp_path)
    calls = _responses(monkeypatch)
    result = _extractor(channel).extract(source, target=_target(), page_range=(2, 3))
    assert isinstance(result, DocumentExtractionResult)
    assert result.status == ConversionStatus.SUCCESS
    assert [item.scope for item in result.items] == [
        PageScope(page_no=2),
        PageScope(page_no=3),
    ]
    assert len(calls) == 2
    for n, call, item in zip((2, 3), calls, result.items):
        content = call["content_items"]
        expected = (
            ["text"]
            if channel == ChannelSelection.TEXT
            else ["image", "text"]
            if channel == ChannelSelection.IMAGE_AND_TEXT
            else ["image"]
        )
        assert [part.type for part in content] == expected
        if expected[-1] == "text":
            assert f"Page {n} total" in content[-1].text
            assert f"Page {5 - n} total" not in content[-1].text
        assert item.validation_status == "passed"
        assert item.num_tokens == 7 and item.usage == {"completion_tokens": 7}
        assert item.stop_reason == VlmStopReason.END_OF_SEQUENCE
        assert "image" not in item.model_dump_json()


@pytest.mark.parametrize("options", [NU_EXTRACT_API, GRANITE_VISION_4_1_API])
def test_existing_api_models_through_sdk(tmp_path, monkeypatch, options):
    calls = _responses(monkeypatch)
    result = _extractor(options=options).extract(
        _archive(tmp_path), target=_target(), page_range=(3, 3)
    )
    assert [item.scope.page_no for item in result.items] == [3]
    assert result.status == ConversionStatus.SUCCESS
    if options is NU_EXTRACT_API:
        assert json.loads(calls[0]["chat_template_kwargs"]["template"]) == {
            "total": "number",
            "note": "string",
        }
    else:
        assert "Final total" in calls[0]["prompt"]


@pytest.mark.parametrize("channel", [ChannelSelection.AUTO, ChannelSelection.TEXT])
def test_unpaginated_source_is_one_text_document(tmp_path, monkeypatch, channel):
    source = tmp_path / "source.md"
    source.write_text("# Invoice\n\nTotal 42")
    calls = _responses(monkeypatch)
    result = _extractor(channel).extract(source, target=_target())
    assert result.status == ConversionStatus.SUCCESS
    assert [item.scope for item in result.items] == [DocumentScope()]
    assert len(calls) == 1
    assert [part.type for part in calls[0]["content_items"]] == ["text"]
    assert calls[0]["content_items"][0].text == source.read_text()


@pytest.mark.parametrize("channel", list(ChannelSelection))
def test_nuextract3_templates_reach_http_from_cached_sdk(
    tmp_path, monkeypatch, channel
):
    source = _archive(tmp_path)
    session = MagicMock()
    post = session.__enter__.return_value.post
    response = post.return_value
    response.ok = True
    monkeypatch.setattr(api_image_request, "_make_retry_session", lambda: session)
    options = ExtractionVlmOptions.from_preset(
        "nuextract_3", engine_options=ApiVlmEngineOptions()
    )
    extractor = _extractor(channel, options)
    first = _target().model_copy(
        update={
            "template": ExtractionTemplate(
                format="nuextract",
                value={"total": "number", "note": "verbatim-string"},
            ),
            "instructions": "Copy note exactly; first template",
        }
    )
    buyer_schema = {
        "type": "object",
        "properties": {"buyer": {"type": "string"}},
        "required": ["buyer"],
    }
    targets = [
        first,
        ExtractionTarget(output_schema=deepcopy(first.output_schema)),
        _target(),
        ExtractionTarget(
            output_schema=buyer_schema,
            template=ExtractionTemplate(
                format="nuextract", value={"buyer": "verbatim-string"}
            ),
            instructions="Copy buyer exactly; second template",
        ),
    ]
    for index, target in enumerate(targets):
        # The last syntactically valid answer deliberately violates its original schema.
        answer = '{"buyer": 17}' if index == 3 else '{"total": 42}'
        response.text = json.dumps(
            {
                "id": "offline",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 7,
                    "total_tokens": 10,
                },
            }
        )
        saved = target.model_dump(mode="json")
        post.reset_mock()
        result = extractor.extract(
            source, target=target, page_range=(2, 3), raises_on_error=False
        )
        assert [item.scope for item in result.items] == [
            PageScope(page_no=2),
            PageScope(page_no=3),
        ]
        assert post.call_count == 2
        for n, call, item in zip((2, 3), post.call_args_list, result.items):
            payload = call.kwargs["json"]
            chat = payload["chat_template_kwargs"]
            expected = (
                target.template.value
                if target.template is not None
                else {"total": "number", "note": "string"}
            )
            assert json.loads(chat["template"]) == expected
            assert chat["enable_thinking"] is False and chat["mode"] == "structured"
            assert payload["model"] == "numind/NuExtract3"
            content = payload["messages"][0]["content"]
            expected_types = (
                ["text"]
                if channel == ChannelSelection.TEXT
                else ["image_url", "text"]
                if channel == ChannelSelection.IMAGE_AND_TEXT
                else ["image_url"]
            )
            assert [part["type"] for part in content] == expected_types
            if "text" in expected_types:
                assert f"Page {n} total" in content[-1]["text"]
                assert f"Page {5 - n} total" not in content[-1]["text"]
            assert item.validation_status == ("failed" if index == 3 else "passed")
            assert item.raw_text == answer
            if index == 3:
                assert "second template" in chat["instructions"]
                assert "first template" not in chat["instructions"]
            elif index == 0:
                assert "first template" in chat["instructions"]
        assert target.model_dump(mode="json") == saved
    assert "template" not in options.model_spec.extra_chat_template_kwargs
    assert "instructions" not in options.engine_options.params
    assert len(extractor._initialized_pipelines) == 1


@pytest.mark.parametrize(
    "channel,page_range,legacy,message",
    [
        (ChannelSelection.IMAGE, DEFAULT_PAGE_RANGE, False, "unpaginated"),
        (ChannelSelection.IMAGE_AND_TEXT, DEFAULT_PAGE_RANGE, False, "unpaginated"),
        (ChannelSelection.AUTO, (2, 3), False, "non-default page range"),
        (ChannelSelection.AUTO, DEFAULT_PAGE_RANGE, True, "use target="),
    ],
)
def test_unpaginated_rejections(
    tmp_path, monkeypatch, channel, page_range, legacy, message
):
    source = tmp_path / "source.md"
    source.write_text("Total 42")
    calls = _responses(monkeypatch)
    entry = {"template": {"total": "number"}} if legacy else {"target": _target()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = _extractor(channel).extract(
            source, page_range=page_range, raises_on_error=False, **entry
        )
    assert result.status == ConversionStatus.FAILURE
    assert message in result.errors[0].error_message
    assert calls == []


@pytest.mark.parametrize(
    "channel", [ChannelSelection.TEXT, ChannelSelection.IMAGE_AND_TEXT]
)
@pytest.mark.parametrize("attribution", ["missing", "multiple_pages", "unknown_page"])
def test_unattributed_text_fails_known_pages_before_inference(
    tmp_path, monkeypatch, channel, attribution
):
    from docling.backend.xml.doclang_archive_backend import DocLangArchiveBackend

    original = DocLangArchiveBackend.convert

    def convert(backend):
        doc = original(backend)
        if attribution == "missing":
            doc.add_text(label=DocItemLabel.TEXT, text="Unattributed amount 99")
        else:
            doc.texts[-1].prov.append(
                doc.texts[-1]
                .prov[0]
                .model_copy(
                    update={"page_no": 2 if attribution == "multiple_pages" else 99}
                )
            )
        return doc

    monkeypatch.setattr(DocLangArchiveBackend, "convert", convert)
    calls = _responses(monkeypatch)
    result = _extractor(channel).extract(
        _archive(tmp_path), target=_target(), page_range=(2, 3), raises_on_error=False
    )
    assert result.status == ConversionStatus.FAILURE
    assert [item.scope.page_no for item in result.items] == [2, 3]
    assert all(
        "page attribution" in item.errors[0] and item.validation_status == "not_run"
        for item in result.items
    )
    assert calls == []


@pytest.mark.parametrize(
    "channel",
    [ChannelSelection.IMAGE, ChannelSelection.IMAGE_AND_TEXT, ChannelSelection.TEXT],
)
def test_image_only_pages_remain_represented(tmp_path, monkeypatch, channel):
    from docling.backend.xml.doclang_archive_backend import DocLangArchiveBackend

    original = DocLangArchiveBackend.convert

    def convert(backend):
        doc = original(backend)
        doc.body.children = []
        doc.texts = []
        return doc

    monkeypatch.setattr(DocLangArchiveBackend, "convert", convert)
    calls = _responses(monkeypatch)
    result = _extractor(channel).extract(
        _archive(tmp_path), target=_target(), page_range=(2, 3), raises_on_error=False
    )
    assert [item.scope.page_no for item in result.items] == [2, 3]
    assert result.status == (
        ConversionStatus.FAILURE
        if channel == ChannelSelection.TEXT
        else ConversionStatus.SUCCESS
    )
    assert len(calls) == (0 if channel == ChannelSelection.TEXT else 2)


def test_missing_page_image_is_scoped_partial_failure(tmp_path, monkeypatch):
    calls = _responses(monkeypatch)
    result = _extractor(ChannelSelection.IMAGE).extract(
        _archive(tmp_path, missing_image=True), target=_target(), page_range=(2, 3)
    )
    assert result.status == ConversionStatus.PARTIAL_SUCCESS
    assert len(calls) == 1
    assert result.items[1].scope == PageScope(page_no=3)
    assert result.items[1].validation_status == "not_run"
    assert "no restored image" in result.items[1].errors[0]


@pytest.mark.parametrize(
    "answer,state,error",
    [
        ('{"total": 42}', "passed", None),
        ('{"total": 42, "note": null}', "passed", None),
        ('{"total": null}', "failed", "$.total"),
        ("{}", "failed", "required property"),
        ('{"total": "42"}', "failed", "$.total"),
        ("not json", "not_run", "invalid JSON"),
        ("null", "not_run", "not an object"),
        ("[]", "not_run", "not an object"),
        ('{"total": NaN}', "not_run", "Nonfinite"),
        ('{"total": 1e999}', "not_run", "Out of range"),
        (RuntimeError("tokenization failed"), "not_run", "tokenization failed"),
        (RuntimeError("inference failed"), "not_run", "inference failed"),
        (RuntimeError("context overflow"), "not_run", "context overflow"),
        (TimeoutError("transport timeout"), "not_run", "transport timeout"),
    ],
)
def test_parse_schema_and_inference_failures_keep_raw_metadata(
    tmp_path, monkeypatch, answer, state, error
):
    calls = _responses(monkeypatch, [answer])
    result = _extractor().extract(
        _archive(tmp_path), target=_target(), page_range=(2, 2), raises_on_error=False
    )
    item = result.items[0]
    assert len(calls) == 1
    assert item.scope == PageScope(page_no=2)
    assert item.validation_status == state
    assert result.status == (
        ConversionStatus.SUCCESS if error is None else ConversionStatus.FAILURE
    )
    if error:
        assert error in item.errors[0]
        assert item.extracted_data is None
    if isinstance(answer, str):
        assert item.raw_text == answer
        assert item.num_tokens == 7 and item.usage == {"completion_tokens": 7}


@pytest.mark.parametrize(
    "stop",
    [VlmStopReason.LENGTH, VlmStopReason.STOP_SEQUENCE, VlmStopReason.CONTENT_FILTERED],
)
def test_incomplete_stop_is_not_success(tmp_path, monkeypatch, stop):
    _responses(monkeypatch, [('{"total": 42}', stop)])
    result = _extractor().extract(
        _archive(tmp_path), target=_target(), page_range=(2, 2), raises_on_error=False
    )
    assert result.status == (
        ConversionStatus.FAILURE
        if stop == VlmStopReason.CONTENT_FILTERED
        else ConversionStatus.PARTIAL_SUCCESS
    )
    assert result.items[0].stop_reason == stop
    assert result.items[0].raw_text == '{"total": 42}'


def test_raises_on_error_and_partial_keeps_successful_items(tmp_path, monkeypatch):
    source = _archive(tmp_path)
    _responses(monkeypatch, [RuntimeError("failed"), '{"total": 42}'])
    result = _extractor().extract(source, target=_target(), page_range=(2, 3))
    assert result.status == ConversionStatus.PARTIAL_SUCCESS
    assert result.items[1].extracted_data == {"total": 42}
    _responses(monkeypatch, ["{}", "{}"])
    with pytest.raises(ConversionError, match=r"Page 2.*required property"):
        _extractor().extract(source, target=_target(), page_range=(2, 3))


@pytest.mark.parametrize("method", ["extract", "extract_all"])
def test_exactly_one_entry_path_before_loading(monkeypatch, method):
    extractor = _extractor()
    monkeypatch.setattr(
        extractor, "_extract", lambda *args, **kwargs: pytest.fail("Source was opened")
    )
    source = Path("missing.pdf") if method == "extract" else [Path("missing.pdf")]
    with pytest.raises(ValueError, match="exactly one"):
        getattr(extractor, method)(source)
    with pytest.raises(ValueError, match="exactly one"):
        getattr(extractor, method)(source, "{}", target=_target())


@pytest.mark.parametrize("method", ["extract", "extract_all"])
def test_legacy_positional_call_warns_once_and_dto_roundtrips(
    tmp_path, monkeypatch, method
):
    source = _archive(tmp_path)
    calls = _responses(monkeypatch)
    extractor = _extractor()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        result = getattr(extractor, method)(
            source if method == "extract" else [source, source],
            {"total": "number"},
            None,
            True,
            100,
            1000000,
            (2, 3),
        )
        results = [result] if method == "extract" else list(result)
    assert sum("template= extraction" in str(w.message) for w in captured) == 1
    assert len(calls) == len(results) * 2
    for result in results:
        assert isinstance(result, ExtractionResult)
        assert [page.page_no for page in result.pages] == [2, 3]
        assert result.model_dump()["pages"][0]["extracted_data"] == {"total": 42}
        page = ExtractedPageData.model_validate_json(result.pages[0].model_dump_json())
        assert page == result.pages[0]
        rebuilt = ExtractionResult(
            input=result.input, status=result.status, pages=[page]
        )
        assert rebuilt.model_dump()["pages"][0]["page_no"] == 2


def test_legacy_class_uses_main_sample_semantics(tmp_path, monkeypatch):
    class Sample(BaseModel):
        total: float = 13.0

    calls = _responses(monkeypatch)
    with pytest.warns(DeprecationWarning, match="template= extraction"):
        _extractor().extract(_archive(tmp_path), Sample)
    assert all(
        json.loads(call["chat_template_kwargs"]["template"]) == {"total": 13.0}
        for call in calls
    )


def test_normalization_and_cached_call_isolation(tmp_path, monkeypatch):
    source = _archive(tmp_path)
    calls = _responses(monkeypatch)
    extractor = _extractor()
    normalizations = []
    original = prompt_utils.normalize_target

    def normalize(target):
        normalizations.append(target)
        return original(target)

    monkeypatch.setattr(prompt_utils, "normalize_target", normalize)
    first = _target().model_copy(update={"instructions": "First call"})
    before = deepcopy(first.model_dump())
    results = extractor.extract_all([source, source], target=first, page_range=(2, 3))
    first.output_schema["required"] = []
    first.output_schema["properties"]["total"]["description"] = "Caller mutation"
    assert all(result.status == ConversionStatus.SUCCESS for result in results)
    assert len(normalizations) == 1
    assert all(
        "Final total" in call["chat_template_kwargs"]["instructions"] for call in calls
    )
    second = ExtractionTarget(
        output_schema=before["output_schema"], instructions="Second call"
    )
    extractor.extract(source, target=second, page_range=(2, 3))
    assert len(normalizations) == 2
    assert all(
        "First call" in call["chat_template_kwargs"]["instructions"]
        for call in calls[:4]
    )
    assert all(
        "Second call" in call["chat_template_kwargs"]["instructions"]
        and "First call" not in call["chat_template_kwargs"]["instructions"]
        for call in calls[4:]
    )
    assert len(extractor._initialized_pipelines) == 1
    pipeline = next(iter(extractor._initialized_pipelines.values()))
    assert (
        not {"target", "validator", "chunks", "items", "constraint_schema"}
        & vars(pipeline).keys()
    )
    assert (
        not {"target", "validator", "chunks", "constraint_schema"}
        & vars(pipeline.vlm_model).keys()
    )


@pytest.mark.parametrize(
    "name,style",
    [
        ("NU_EXTRACT_2B_TRANSFORMERS", "nuextract"),
        ("GRANITE_VISION_4_1_TRANSFORMERS", "granite_vision"),
    ],
)
def test_main_inline_constants_support_overrides_and_serialization(name, style):
    from docling.datamodel import vlm_model_specs
    from docling.datamodel.pipeline_options_vlm_model import InlineVlmOptions

    original = getattr(vlm_model_specs, name)
    assert isinstance(original, InlineVlmOptions)
    inline = original.model_copy(
        update={
            "repo_id": "custom/model",
            "scale": 1.25,
            "max_size": 512,
            "extra_processor_kwargs": {"custom_option": True},
        }
    )
    with pytest.warns(DeprecationWarning, match="InlineVlmOptions"):
        options = VlmExtractionPipelineOptions(
            vlm_options=inline.model_dump(mode="json"), extraction_prompt_style=style
        )
    assert options.vlm_options.model_spec.default_repo_id == "custom/model"
    assert (
        options.vlm_options.model_spec.extra_processor_kwargs["custom_option"] is True
    )
    assert options.vlm_options.scale == 1.25 and options.vlm_options.max_size == 512
    assert options.vlm_options.model_spec.preparation == (
        "nuextract" if style == "nuextract" else "generic_chat"
    )


def test_validation_runs_once_without_changing_original_schema(tmp_path, monkeypatch):
    from docling.models.extraction import template_utils

    checks = []
    original = template_utils.schema_validator

    class Validator:
        def __init__(self, schema):
            self.inner = original(schema)

        def iter_errors(self, data):
            checks.append(data)
            return self.inner.iter_errors(data)

    monkeypatch.setattr(prompt_utils, "schema_validator", Validator)
    target = _target()
    before = deepcopy(target.model_dump())
    _responses(monkeypatch)
    result = _extractor().extract(_archive(tmp_path), target=target, page_range=(2, 3))
    assert result.status == ConversionStatus.SUCCESS
    assert checks == [{"total": 42}, {"total": 42}]
    assert target.model_dump() == before


def test_template_only_target_does_not_infer_schema(tmp_path, monkeypatch):
    from docling.datamodel.extraction import ExtractionTemplate

    _responses(monkeypatch, ["{}"])
    result = _extractor().extract(
        _archive(tmp_path),
        page_range=(2, 2),
        target=ExtractionTarget(
            template=ExtractionTemplate(format="nuextract", value={"total": "number"})
        ),
    )
    assert result.status == ConversionStatus.SUCCESS
    assert result.items[0].validation_status == "not_requested"


def test_wire_and_pydantic_targets_validate_identically(tmp_path, monkeypatch):
    source = _archive(tmp_path)
    targets = [
        _target(),
        ExtractionTarget.model_validate_json(_target().model_dump_json()),
    ]
    results = []
    for target in targets:
        _responses(monkeypatch, ['{"total": "42"}'])
        results.append(
            _extractor().extract(
                source, target=target, page_range=(2, 2), raises_on_error=False
            )
        )
    assert results[0].items == results[1].items
    assert results[0].status == results[1].status == ConversionStatus.FAILURE


def test_remaining_timeout_reaches_api_and_unprocessed_pages_are_kept(
    tmp_path, monkeypatch
):
    from docling.pipeline import extraction_vlm_pipeline as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        clock[0] = 2.0
        return ApiImageRequestResult('{"total": 42}', 7, VlmStopReason.END_OF_SEQUENCE)

    monkeypatch.setattr(api_extraction_model, "api_extraction_request", request)
    result = _extractor(timeout=1.0).extract(
        _archive(tmp_path), target=_target(), page_range=(2, 3)
    )
    assert result.status == ConversionStatus.PARTIAL_SUCCESS
    assert len(calls) == 1 and calls[0]["timeout"] == 1.0
    assert result.items[0].extracted_data == {"total": 42}
    assert result.items[1].scope == PageScope(page_no=3)
    assert result.items[1].validation_status == "not_run"
    assert "timeout" in result.items[1].errors[0]
    assert result.errors[0].category.value == "timeout"


@pytest.mark.parametrize("preset", ["nuextract_2b", "granite_vision_4_1"])
@pytest.mark.parametrize("failure", [None, "tokenization", "context", "inference"])
def test_existing_local_models_through_completed_sdk(
    tmp_path, monkeypatch, preset, failure
):
    import torch
    from transformers import GenerationConfig

    from docling.datamodel.extraction_options import ExtractionVlmOptions
    from docling.models.extraction import transformers_extraction_model as module

    calls = []

    class Processor:
        pad_token_id = 0
        eos_token_id = 9
        padding_side = "right"

        @property
        def tokenizer(self):
            return self

        def apply_chat_template(self, conversation, **kwargs):
            calls.append((conversation, kwargs))
            return "serialized input"

        def __call__(self, **kwargs):
            if failure == "tokenization":
                raise RuntimeError("tokenization failed")
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def batch_decode(self, sequences, **kwargs):
            return ['{"total": 42}']

    class Model:
        def eval(self):
            pass

        def generate(self, **kwargs):
            if failure == "inference":
                raise RuntimeError("inference failed")
            return torch.tensor([[1, 2, 3, 8, 9]])

    monkeypatch.setattr(
        module.AutoProcessor, "from_pretrained", lambda *args, **kwargs: Processor()
    )
    monkeypatch.setattr(
        module.AutoModelForImageTextToText,
        "from_pretrained",
        lambda *args, **kwargs: Model(),
    )
    monkeypatch.setattr(
        module.GenerationConfig,
        "from_pretrained",
        lambda *args, **kwargs: GenerationConfig(),
    )
    monkeypatch.setattr(module, "decide_device", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(
        prompt_utils,
        "_process_all_vision_info",
        lambda messages: [messages[0][0]["content"][0]["image"]],
    )
    options = ExtractionVlmOptions.from_preset(preset)
    options = options.model_copy(
        update={
            "model_spec": options.model_spec.model_copy(
                update={"extra_generation_config": {"eos_token_id": 9}}
            )
        }
    )
    if failure == "context":
        options = options.model_copy(
            update={
                "model_spec": options.model_spec.model_copy(
                    update={"max_input_tokens": 2}
                )
            }
        )
    source = _archive(tmp_path)
    extractor = _extractor(options=options)
    for fopt in extractor.extraction_format_to_options.values():
        fopt.pipeline_options.artifacts_path = tmp_path
    result = extractor.extract(
        source, target=_target(), page_range=(2, 3), raises_on_error=False
    )
    assert len(calls) == 2 and [item.scope.page_no for item in result.items] == [2, 3]
    if failure:
        assert result.status == ConversionStatus.FAILURE
        assert all(item.validation_status == "not_run" for item in result.items)
        message = "context limit" if failure == "context" else failure
        assert all(message in item.errors[0] for item in result.items)
    else:
        assert result.status == ConversionStatus.SUCCESS
        assert all(
            item.extracted_data == {"total": 42}
            and item.validation_status == "passed"
            and item.num_tokens == 2
            and item.generation_time >= 0
            and item.stop_reason == VlmStopReason.END_OF_SEQUENCE
            for item in result.items
        ), result.items


@pytest.mark.parametrize("kind", ["limit", "pipeline_init", "no_pipeline"])
def test_unexecuted_sources_release_backend(tmp_path, monkeypatch, kind):
    from docling.backend.image_backend import ImageDocumentBackend

    source = tmp_path / "source.png"
    Image.new("RGB", (32, 24)).save(source)
    unloaded = []
    original = ImageDocumentBackend.unload

    def unload(backend):
        original(backend)
        unloaded.append(backend)

    monkeypatch.setattr(ImageDocumentBackend, "unload", unload)
    extractor = _extractor()
    if kind == "pipeline_init":

        def fail(*args):
            raise RuntimeError("pipeline init failed")

        monkeypatch.setattr(extractor, "_get_pipeline", fail)
        with pytest.raises(RuntimeError, match="pipeline init failed"):
            extractor.extract(source, target=_target())
    else:
        kwargs = {"max_num_pages": 0} if kind == "limit" else {}
        if kind == "no_pipeline":
            monkeypatch.setattr(extractor, "_get_pipeline", lambda *args: None)
        result = extractor.extract(
            source, target=_target(), raises_on_error=False, **kwargs
        )
        assert result.status == ConversionStatus.FAILURE
    assert len(unloaded) == 1 and unloaded[0]._frames == []


def test_pipeline_target_boundary_and_invalid_call_cleanup(tmp_path, monkeypatch):
    from docling.backend.xml.doclang_archive_backend import DocLangArchiveBackend
    from docling.datamodel.document import InputDocument

    source = _archive(tmp_path)
    calls = _responses(monkeypatch)
    pipeline = ExtractionVlmPipeline(
        VlmExtractionPipelineOptions(
            vlm_options=NU_EXTRACT_API, enable_remote_services=True
        )
    )

    def input_doc():
        return InputDocument(
            path_or_stream=source,
            format=InputFormat.DCLX,
            backend=DocLangArchiveBackend,
        )

    with pytest.warns(DeprecationWarning, match="template= extraction"):
        legacy = pipeline.execute(input_doc(), False, template="{}")
    assert isinstance(legacy, ExtractionResult) and len(legacy.pages) == 3
    modern = pipeline.execute(input_doc(), False, target=_target())
    assert isinstance(modern, DocumentExtractionResult) and len(modern.items) == 3
    assert len(calls) == 6
    doc = input_doc()
    unloaded = []
    monkeypatch.setattr(doc._backend, "unload", lambda: unloaded.append(True))
    with pytest.raises(ValueError, match="exactly one"):
        pipeline.execute(doc, False, template="{}", target=_target())
    assert unloaded == [True]


def test_constrained_sdk_call_still_validates_original_schema(tmp_path, monkeypatch):
    options = NU_EXTRACT_API.model_copy(update={"output_mode": "schema_constrained"})
    target = _target()
    before = deepcopy(target.model_dump())
    calls = _responses(monkeypatch, ['{"total": "42"}'])
    result = _extractor(options=options).extract(
        _archive(tmp_path), target=target, page_range=(2, 2), raises_on_error=False
    )
    assert result.status == ConversionStatus.FAILURE
    assert result.items[0].validation_status == "failed"
    assert "$.total" in result.items[0].errors[0]
    assert calls[0]["constraint_schema"] is not None
    assert target.model_dump() == before


def test_schema_preparation_failure_makes_no_inference_requests(tmp_path, monkeypatch):
    calls = _responses(monkeypatch)
    target = ExtractionTarget(
        output_schema={"type": "object", "properties": {"total": {"type": "invalid"}}}
    )
    result = _extractor().extract(
        _archive(tmp_path), target=target, raises_on_error=False
    )
    assert result.status == ConversionStatus.FAILURE
    assert result.errors and "invalid JSON Schema" in result.errors[0].error_message
    assert calls == []


def test_schema_runtime_failure_retains_raw_output(tmp_path, monkeypatch):
    class Validator:
        def __init__(self, schema):
            pass

        def iter_errors(self, data):
            raise RuntimeError("validator failed")

    monkeypatch.setattr(prompt_utils, "schema_validator", Validator)
    _responses(monkeypatch, ['{"total": 42}'])
    result = _extractor().extract(
        _archive(tmp_path), target=_target(), page_range=(2, 2), raises_on_error=False
    )
    item = result.items[0]
    assert result.status == ConversionStatus.FAILURE
    assert item.validation_status == "not_run"
    assert item.raw_text == '{"total": 42}' and item.num_tokens == 7
    assert item.errors == ["Schema validation could not run: validator failed"]


def test_converted_document_page_limit_is_enforced(tmp_path, monkeypatch):
    calls = _responses(monkeypatch)
    result = _extractor().extract(
        _archive(tmp_path), target=_target(), max_num_pages=2, raises_on_error=False
    )
    assert result.status == ConversionStatus.FAILURE
    assert "max_num_pages" in result.errors[0].error_message
    assert calls == []
