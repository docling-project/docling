# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Stage 1 target preparation and durable-result contract gates (no weights)."""

import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image
from pydantic import BaseModel, Field, ValidationError, field_validator

from docling.datamodel.base_models import InputFormat, VlmStopReason
from docling.datamodel.document import InputDocument
from docling.datamodel.extraction import (
    DocumentExtractionResult,
    DocumentScope,
    ExtractedPageData,
    ExtractionItem,
    ExtractionResult,
    ExtractionTarget,
    ExtractionTemplate,
    PageScope,
)
from docling.datamodel.extraction_options import (
    GRANITE_VISION_4_1_SPEC,
    NUEXTRACT_2B_SPEC,
)
from docling.models.extraction.prompt_utils import prepare_legacy_target, prepare_target

_FIXTURES = json.loads(
    (Path(__file__).parent / "data/extraction/template_conversion.json").read_text()
)


@pytest.mark.parametrize("case", _FIXTURES["supported"], ids=lambda c: c["name"])
def test_supported_conversion_preserves_schema_and_caller(case: dict) -> None:
    schema = deepcopy(case["schema"])
    target = ExtractionTarget(output_schema=schema)
    before = target.model_dump()
    prepared = prepare_target(target, NUEXTRACT_2B_SPEC)
    again = prepare_target(target, NUEXTRACT_2B_SPEC)

    assert json.loads(prepared.chat_template_kwargs["template"]) == case["template"]
    assert prepared.chat_template_kwargs == again.chat_template_kwargs
    assert prepared.validator.schema == schema
    assert prepared.target.output_schema == schema
    assert target.model_dump() == before and schema == case["schema"]
    assert prepared.validator is not again.validator
    prepared.target.output_schema.clear()
    assert target.model_dump() == before


@pytest.mark.parametrize("case", _FIXTURES["unsupported"], ids=lambda c: c["name"])
def test_conversion_loss_fails_with_path(case: dict) -> None:
    target = ExtractionTarget(
        output_schema={
            "type": "object",
            "properties": {"value": case["node"]},
        }
    )
    with pytest.raises(ValueError, match=re.escape(case["path"])):
        prepare_target(target, NUEXTRACT_2B_SPEC)
    # Validation supports the full draft; an explicit native template bypasses conversion.
    explicit = target.model_copy(
        update={
            "template": ExtractionTemplate(
                format="nuextract", value={"value": "string"}
            ),
        }
    )
    assert (
        prepare_target(explicit, NUEXTRACT_2B_SPEC).validator.schema
        == target.output_schema
    )
    assert (
        prepare_target(target, GRANITE_VISION_4_1_SPEC).validator.schema
        == target.output_schema
    )


class Address(BaseModel):
    city: str = Field(description="Postal city")


class Invoice(BaseModel):
    address: Address
    total: float | None = Field(default=None, description="Final payable total")
    tags: list[str] = Field(default_factory=list)

    @field_validator("total")
    @classmethod
    def _python_only(cls, value):
        if value == 123:
            raise ValueError("SDK-only constraint")
        return value


def test_pydantic_shorthand_is_schema_and_matches_wire_validation() -> None:
    target = ExtractionTarget.from_pydantic(Invoice)
    wire = ExtractionTarget.model_validate_json(target.model_dump_json())
    sdk = prepare_target(target, NUEXTRACT_2B_SPEC)
    remote = prepare_target(wire, NUEXTRACT_2B_SPEC)
    assert sdk.chat_template_kwargs == remote.chat_template_kwargs
    assert json.loads(sdk.chat_template_kwargs["template"]) == {
        "address": {"city": "string"},
        "total": "number",
        "tags": ["string"],
    }
    assert "Postal city" in sdk.chat_template_kwargs["instructions"]
    for data in [
        {"address": {"city": "Paris"}, "total": 123, "tags": []},
        {"address": {"city": "Paris"}, "total": None},
        {"address": {"city": None}},
        {},
        {"address": {"city": "Paris"}, "total": "12"},
    ]:
        assert sdk.validator.is_valid(data) == remote.validator.is_valid(data)
    assert sdk.validator.is_valid({"address": {"city": "Paris"}, "total": 123})
    assert not sdk.validator.is_valid({"address": {"city": None}})


def test_examples_are_values_without_inferred_contract() -> None:
    example = Invoice(address=Address(city="Example city"))
    target = ExtractionTarget(
        template=ExtractionTemplate(format="example_json", value=example)
    )
    prepared = prepare_target(target, GRANITE_VISION_4_1_SPEC)
    assert prepared.validator is None
    assert prepared.constraint_schema is None
    assert '"tags": []' in prepared.prompt
    assert "illustration only" in prepared.prompt
    assert "Output contract" not in prepared.prompt
    assert "use null" not in prepared.prompt
    with pytest.raises(ValidationError, match="example_json"):
        ExtractionTemplate(format="nuextract", value=example)
    with pytest.raises(ValidationError):
        ExtractionTemplate(format="example_json", value=Invoice)


def test_schema_and_example_remain_distinct_in_generic_prompt() -> None:
    schema = {
        "type": "object",
        "properties": {"total": {"type": "number"}},
        "required": ["total"],
    }
    target = ExtractionTarget(
        output_schema=schema,
        template=ExtractionTemplate(
            format="example_json",
            value={"total": 12.3, "items": []},
        ),
        instructions="Extract the tax-inclusive total.",
    )
    prepared = prepare_target(
        target,
        GRANITE_VISION_4_1_SPEC.model_copy(
            update={"prompt": "Read the page carefully."}
        ),
    )
    assert "Read the page carefully." in prepared.prompt
    assert "Extract the tax-inclusive total." in prepared.prompt
    assert (
        "schema permits null" in prepared.prompt
        and "Optional properties may be omitted" in prepared.prompt
    )
    assert '"total": 12.3' in prepared.prompt
    assert not prepared.validator.is_valid({"total": None})
    assert not prepared.validator.is_valid({})
    assert not prepared.validator.is_valid({"total": "12.3"})
    assert prepared.validator.is_valid({"total": 12.3})


def test_native_template_preserves_semantics_without_schema_conversion() -> None:
    native = {"text": "verbatim-string", "date": "date", "choices": [["yes", "no"]]}
    target = ExtractionTarget(
        template=ExtractionTemplate(format="nuextract", value=native)
    )
    prepared = prepare_target(target, NUEXTRACT_2B_SPEC)
    assert json.loads(prepared.chat_template_kwargs["template"]) == native
    assert prepared.validator is None
    with pytest.raises(ValueError, match="native dialect"):
        prepare_target(target, GRANITE_VISION_4_1_SPEC)
    with pytest.raises(ValueError, match="not example_json"):
        prepare_target(
            ExtractionTarget(
                template=ExtractionTemplate(format="example_json", value={})
            ),
            NUEXTRACT_2B_SPEC,
        )


@pytest.mark.parametrize(
    "schema, path",
    [
        (
            {"type": "object", "$schema": "http://json-schema.org/draft-07/schema#"},
            "#/$schema",
        ),
        (
            {"type": "object", "properties": {"value": {"type": 123}}},
            "#/properties/value/type",
        ),
        ({"type": "array"}, "#"),
        ({}, "#"),
        ({"type": ["object", "null"]}, "#"),
        (
            {
                "type": "object",
                "properties": {"value": {"$ref": "https://example.com/schema"}},
            },
            "#/properties/value/$ref",
        ),
        (
            {"type": "object", "properties": {"value": {"$ref": "#/$defs/missing"}}},
            "#/properties/value/$ref",
        ),
        (
            {"type": "object", "properties": {"value": {"$ref": "#"}}},
            "#/properties/value/$ref",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"$id": "other", "type": "string"}},
            },
            "#/properties/value/$id",
        ),
        (
            {"type": "object", "properties": {"value": {"$dynamicRef": "#here"}}},
            "#/properties/value/$dynamicRef",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "value": {"$schema": "http://json-schema.org/draft-07/schema#"}
                },
            },
            "#/properties/value/$schema",
        ),
    ],
)
def test_schema_preflight_fails_before_guidance(schema: dict, path: str) -> None:
    target = ExtractionTarget(output_schema=schema)
    with pytest.raises(ValueError, match=re.escape(path)):
        prepare_target(target, GRANITE_VISION_4_1_SPEC)


def test_local_pointer_escaping_and_annotations_are_not_schemas() -> None:
    schema = {
        "type": "object",
        "$defs": {"a/b~c": {"type": "string"}},
        "properties": {
            "value": {"$ref": "#/$defs/a~1b~0c"},
            "text": {
                "type": "string",
                "description": "verbatim-string",
                "examples": [{"$ref": "https://example.com/not-a-schema"}],
            },
        },
    }
    prepared = prepare_target(ExtractionTarget(output_schema=schema), NUEXTRACT_2B_SPEC)
    assert json.loads(prepared.chat_template_kwargs["template"]) == {
        "value": "string",
        "text": "string",
    }
    assert prepared.validator.is_valid({"value": "ok", "text": "ok"})


def test_ref_assertions_are_validated_and_not_silently_overwritten() -> None:
    schema = {
        "type": "object",
        "$defs": {"Value": {"type": "number"}},
        "properties": {
            "value": {"$ref": "#/$defs/Value", "type": "string"},
        },
    }
    target = ExtractionTarget(output_schema=schema)
    with pytest.raises(ValueError, match=r"#/properties/value/type"):
        prepare_target(target, NUEXTRACT_2B_SPEC)
    validator = prepare_target(target, GRANITE_VISION_4_1_SPEC).validator
    assert not validator.is_valid({"value": "1"}) and not validator.is_valid(
        {"value": 1}
    )


def test_standard_validation_without_coercion_defaults_or_format_assertions() -> None:
    schema = {
        "type": "object",
        "properties": {
            "email": {"type": "string", "format": "email"},
            "count": {"type": "integer", "minimum": 2, "default": 5},
        },
        "required": ["count"],
        "additionalProperties": False,
    }
    prepared = prepare_target(ExtractionTarget(output_schema=schema), NUEXTRACT_2B_SPEC)
    assert prepared.validator.is_valid({"count": 2, "email": "not-an-email"})
    assert not prepared.validator.is_valid({"count": "2"})
    assert not prepared.validator.is_valid({"count": 1})
    data = {}
    errors = list(prepared.validator.iter_errors(data))
    assert errors[0].validator == "required" and data == {}
    errors = list(prepared.validator.iter_errors({"count": 1}))
    assert list(errors[0].absolute_path) == ["count"]


def test_prepared_options_and_dynamic_values_are_call_owned() -> None:
    spec = NUEXTRACT_2B_SPEC.model_copy(
        update={
            "extra_chat_template_kwargs": {
                "enable_thinking": False,
                "options": {"a": 1},
            },
            "extra_processor_kwargs": {"visual": {"budget": 100}},
        }
    )
    first = prepare_target(
        ExtractionTarget(
            template=ExtractionTemplate(format="nuextract", value={"a": "integer"}),
            instructions="First",
        ),
        spec,
    )
    second = prepare_target(
        ExtractionTarget(
            template=ExtractionTemplate(format="nuextract", value={"b": "number"}),
            instructions="Second",
        ),
        spec,
    )
    assert "First" in first.chat_template_kwargs["instructions"]
    assert "First" not in second.chat_template_kwargs["instructions"]
    assert "Second" in second.chat_template_kwargs["instructions"]
    assert not first.chat_template_kwargs["enable_thinking"]
    first.chat_template_kwargs["options"]["a"] = 2
    first.processor_kwargs["visual"]["budget"] = 0
    assert second.processor_kwargs["visual"]["budget"] == 100
    assert spec.extra_chat_template_kwargs["options"]["a"] == 1


@pytest.mark.parametrize(
    "options",
    [
        {"template": "static"},
        {"instructions": "static"},
        {"messages": []},
        {"mode": "markdown"},
    ],
)
def test_request_owned_or_non_extraction_chat_options_fail(options: dict) -> None:
    spec = NUEXTRACT_2B_SPEC.model_copy(update={"extra_chat_template_kwargs": options})
    with pytest.raises(ValueError):
        prepare_target(ExtractionTarget(output_schema={"type": "object"}), spec)


@pytest.mark.parametrize(
    "target",
    [
        {},
        {"instructions": "only"},
        {"template": {"format": "unknown", "value": {}}},
        {"template": {"format": "example_json", "value": "{}"}},
        {"grouping": "page", "output_schema": {"type": "object"}},
    ],
)
def test_malformed_targets_fail(target: dict) -> None:
    with pytest.raises(ValidationError):
        ExtractionTarget.model_validate(target)


@pytest.mark.parametrize(
    "scope",
    [
        {"kind": "page", "page_no": 0},
        {"kind": "page", "page_no": -1},
        {"kind": "page", "page_no": True},
        {"kind": "page", "page_no": "1"},
        {"kind": "page", "page_no": 1.0},
        {"kind": "page"},
        {"kind": "document", "page_no": 1},
        {"kind": "section", "id": "x"},
        {"kind": "page", "page_no": 1, "end": 2},
        {},
    ],
)
def test_malformed_scope_rejected_in_json(scope: dict) -> None:
    with pytest.raises(ValidationError):
        ExtractionItem.model_validate_json(json.dumps({"scope": scope}))


@pytest.mark.parametrize("scope", [PageScope(page_no=7), DocumentScope()])
def test_item_roundtrip_keeps_scope_outcome_and_prediction_metadata(scope) -> None:
    item = ExtractionItem(
        scope=scope,
        extracted_data={"lines": [{"total": 2.3}], "missing": None},
        raw_text='{"total":2.3}',
        errors=["example error"],
        validation_status="failed",
        num_tokens=12,
        usage={"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17},
        stop_reason=VlmStopReason.LENGTH,
    )
    assert ExtractionItem.model_validate_json(item.model_dump_json()) == item
    wire = json.loads(item.model_dump_json())
    assert "content" not in wire and "image" not in wire
    with pytest.raises(ValidationError):
        ExtractionItem(scope=scope, content=[])
    with Image.new("RGB", (1, 1)) as image:
        with pytest.raises(ValidationError):
            ExtractionItem(scope=scope, extracted_data={"image": image})


def test_envelope_and_legacy_dtos_keep_separate_collections() -> None:
    document = InputDocument.create_invalid(
        filename="test.pdf", format=InputFormat.PDF, filesize=0
    )
    result = DocumentExtractionResult(
        input=document, items=[ExtractionItem(scope=DocumentScope())]
    )
    assert json.loads(result.model_dump_json())["items"][0]["scope"] == {
        "kind": "document"
    }
    legacy = ExtractionResult(
        input=document,
        pages=[ExtractedPageData(page_no=3, extracted_data={"ok": True})],
    )
    wire = json.loads(legacy.model_dump_json())
    assert wire["pages"][0]["page_no"] == 3 and "items" not in wire
    assert (
        ExtractedPageData.model_validate_json(legacy.pages[0].model_dump_json())
        == legacy.pages[0]
    )


def test_legacy_sample_class_path_remains_separate() -> None:
    class Legacy(BaseModel):
        count: int = Field(default=4, examples=[4])

    legacy = prepare_legacy_target(Legacy, NUEXTRACT_2B_SPEC)
    modern = prepare_target(ExtractionTarget.from_pydantic(Legacy), NUEXTRACT_2B_SPEC)
    assert json.loads(legacy.chat_template_kwargs["template"]) == {"count": 4}
    assert json.loads(modern.chat_template_kwargs["template"]) == {"count": "integer"}
    assert legacy.validator is None and legacy.target is None
    assert modern.validator is not None
    assert (
        prepare_legacy_target(
            '{"count":"integer"}', NUEXTRACT_2B_SPEC
        ).chat_template_kwargs["template"]
        == '{"count":"integer"}'
    )


def test_preparation_import_does_not_require_optional_validator() -> None:
    script = """
import sys
sys.modules['jsonschema'] = None
from docling.datamodel.extraction import ExtractionTarget, ExtractionTemplate
from docling.datamodel.extraction_options import GRANITE_VISION_4_1_SPEC
from docling.models.extraction.prompt_utils import prepare_target
prepared = prepare_target(ExtractionTarget(template=ExtractionTemplate(format='example_json', value={'items':[]})), GRANITE_VISION_4_1_SPEC)
assert prepared.validator is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_root_local_reference_and_identifier_stay_in_supplied_document() -> None:
    schema = {
        "$id": "https://example.com/local-contract",
        "$defs": {
            "Contract": {"type": "object", "properties": {"name": {"type": "string"}}},
        },
        "$ref": "#/$defs/Contract",
    }
    prepared = prepare_target(ExtractionTarget(output_schema=schema), NUEXTRACT_2B_SPEC)
    assert json.loads(prepared.chat_template_kwargs["template"]) == {"name": "string"}
    assert prepared.validator.is_valid({"name": "ok"})
    assert not prepared.validator.is_valid({"name": 1})


def test_referenced_annotation_is_checked_as_a_schema() -> None:
    schema = {
        "type": "object",
        "examples": [{"type": "not-a-type"}],
        "properties": {
            "value": {"$ref": "#/examples/0"},
        },
    }
    with pytest.raises(ValueError, match=r"#/properties/value/\$ref/type"):
        prepare_target(ExtractionTarget(output_schema=schema), GRANITE_VISION_4_1_SPEC)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_json_values_fail_at_target_and_item_boundaries(value: float) -> None:
    with pytest.raises(ValidationError):
        ExtractionTarget(
            template=ExtractionTemplate(
                format="example_json", value={"nested": [value]}
            )
        )
    with pytest.raises(ValidationError):
        ExtractionItem(scope=DocumentScope(), extracted_data={"nested": [value]})


def test_required_field_without_type_cannot_disappear_from_template() -> None:
    target = ExtractionTarget(output_schema={"type": "object", "required": ["missing"]})
    with pytest.raises(ValueError, match=r"#/required/0"):
        prepare_target(target, NUEXTRACT_2B_SPEC)
    assert prepare_target(target, GRANITE_VISION_4_1_SPEC).validator.is_valid(
        {"missing": 1}
    )


def test_nullable_branch_annotations_keep_native_type_and_validation() -> None:
    target = ExtractionTarget(
        output_schema={
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null", "description": "Missing value"},
                    ]
                },
            },
        }
    )
    prepared = prepare_target(target, NUEXTRACT_2B_SPEC)
    assert json.loads(prepared.chat_template_kwargs["template"]) == {"value": "string"}
    assert "Missing value" in prepared.chat_template_kwargs["instructions"]
    assert prepared.validator.is_valid({"value": None})
    assert not prepared.validator.is_valid({"value": 2})
