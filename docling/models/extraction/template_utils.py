# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Call-local schema preflight and bounded NuExtract template conversion."""

from collections.abc import Iterator
from copy import deepcopy
from typing import TYPE_CHECKING, Any, NoReturn
from urllib.parse import unquote

from docling.datamodel.extraction import ExtractionTarget

if TYPE_CHECKING:
    from jsonschema.protocols import Validator

_SCHEMA_MAPS = {
    "$defs",
    "definitions",
    "properties",
    "patternProperties",
    "dependentSchemas",
}
_SCHEMA_LISTS = {"allOf", "anyOf", "oneOf", "prefixItems"}
_SCHEMA_VALUES = {
    "additionalProperties",
    "unevaluatedProperties",
    "propertyNames",
    "items",
    "contains",
    "unevaluatedItems",
    "not",
    "if",
    "then",
    "else",
    "contentSchema",
}
_ANNOTATIONS = {
    "$schema",
    "$id",
    "$defs",
    "definitions",
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
}
# These constraints stay in the unchanged validator contract and task instructions.
_CONSTRAINTS = {
    "required",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
}


def _pointer(parts: tuple[str | int, ...]) -> str:
    return "#" + "".join(
        "/" + str(p).replace("~", "~0").replace("/", "~1") for p in parts
    )


def _fail(path: tuple[str | int, ...], reason: str) -> NoReturn:
    raise ValueError(f"{_pointer(path)}: {reason}")


def normalize_target(target: ExtractionTarget) -> ExtractionTarget:
    """Own and revalidate all mappings, including model_copy/update inputs."""
    return ExtractionTarget.model_validate(deepcopy(target.model_dump(mode="python")))


def _resolve_ref(root: dict[str, Any], ref: str, path: tuple[str | int, ...]) -> Any:
    if ref == "#":
        return root
    if not ref.startswith("#/"):
        _fail(path, "only local JSON Pointer references are supported")
    node: Any = root
    for part in unquote(ref[2:]).split("/"):
        if "~" in part.replace("~0", "").replace("~1", ""):
            _fail(path, "invalid JSON Pointer escape")
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and key in node:
            node = node[key]
        elif (
            isinstance(node, list)
            and key.isdecimal()
            and str(int(key)) == key
            and int(key) < len(node)
        ):
            node = node[int(key)]
        else:
            _fail(path, f"unresolved reference {ref!r}")
    if not isinstance(node, (dict, bool)):
        _fail(path, f"reference {ref!r} does not identify a schema")
    return node


def _schema_children(
    node: dict[str, Any], path: tuple[str | int, ...]
) -> Iterator[tuple[Any, tuple[str | int, ...]]]:
    for key, value in node.items():
        if key in _SCHEMA_MAPS:
            if not isinstance(value, dict):
                _fail((*path, key), "schema map must be an object")
            for name, child in value.items():
                yield child, (*path, key, name)
        elif key in _SCHEMA_LISTS:
            for index, child in enumerate(value):
                yield child, (*path, key, index)
        elif key in _SCHEMA_VALUES:
            yield value, (*path, key)


def _object_contract(node: Any, root: dict[str, Any]) -> bool:
    if not isinstance(node, dict):
        return False
    if "type" in node:
        return node["type"] == "object" or node["type"] == ["object"]
    if "$ref" in node and _object_contract(
        _resolve_ref(root, node["$ref"], ("$ref",)), root
    ):
        return True
    if any(_object_contract(child, root) for child in node.get("allOf", [])):
        return True
    return any(
        key in node and all(_object_contract(child, root) for child in node[key])
        for key in ("anyOf", "oneOf")
    )


def schema_validator(schema: dict[str, Any]) -> "Validator":
    """Preflight Draft 2020-12 and references without retrieving any resource."""
    from jsonschema import Draft202012Validator, SchemaError
    from referencing import Registry

    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        _fail(tuple(exc.absolute_path), f"invalid JSON Schema: {exc.message}")

    visited: set[int] = set()

    def visit(node: Any, path: tuple[str | int, ...], stack: set[int]) -> None:
        if not isinstance(node, dict):
            return
        if id(node) in stack:
            _fail(path, "recursive schema references are unsupported")
        if id(node) in visited:
            return
        draft = node.get("$schema", "https://json-schema.org/draft/2020-12/schema")
        if draft != "https://json-schema.org/draft/2020-12/schema":
            _fail((*path, "$schema"), "only JSON Schema Draft 2020-12 is supported")
        for key in (
            "$anchor",
            "$dynamicAnchor",
            "$dynamicRef",
            "$recursiveRef",
            "$recursiveAnchor",
            "$vocabulary",
        ):
            if key in node:
                _fail(
                    (*path, key),
                    "schema resources and dynamic/anchor references are unsupported",
                )
        if path and "$id" in node:
            _fail((*path, "$id"), "nested schema resources are unsupported")
        next_stack = stack | {id(node)}
        if "$ref" in node:
            referenced = _resolve_ref(schema, node["$ref"], (*path, "$ref"))
            try:
                Draft202012Validator.check_schema(referenced)
            except SchemaError as exc:
                _fail(
                    (*path, "$ref", *exc.absolute_path),
                    f"invalid referenced schema: {exc.message}",
                )
            visit(
                referenced,
                (*path, "$ref"),
                next_stack,
            )
        for child, child_path in _schema_children(node, path):
            visit(child, child_path, next_stack)
        visited.add(id(node))

    visit(schema, (), set())
    if not _object_contract(schema, schema):
        _fail((), "the output schema must require a top-level object")
    # An empty registry fails closed even if a future schema keyword uses retrieval.
    return Draft202012Validator(schema, registry=Registry())


def _schema_to_nuextract(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert only fixed objects, homogeneous arrays, string enums and nullable types.

    Schema-only annotations never select native semantic types such as verbatim-string.
    Requiredness, nullability and scalar bounds remain authoritative in the validator.
    """

    def convert(node: Any, path: tuple[str | int, ...]) -> Any:
        if not isinstance(node, dict):
            _fail(path, "boolean schemas have no NuExtract template representation")
        allowed = (
            _ANNOTATIONS
            | _CONSTRAINTS
            | {
                "type",
                "properties",
                "items",
                "enum",
                "anyOf",
                "$ref",
                "additionalProperties",
            }
        )
        for key in node:
            if key not in allowed:
                _fail((*path, key), "unsupported NuExtract conversion keyword")
        if "$ref" in node:
            # Draft 2020-12 siblings are conjunctive; only metadata may be overlaid.
            for key in node.keys() - _ANNOTATIONS - {"$ref"}:
                _fail((*path, key), "assertion siblings of $ref cannot be converted")
            referenced = _resolve_ref(schema, node["$ref"], (*path, "$ref"))
            return convert(referenced, path)
        if "anyOf" in node:
            branches = node["anyOf"]
            non_null = [
                b
                for b in branches
                if not (
                    isinstance(b, dict)
                    and b.get("type") == "null"
                    and b.keys() <= _ANNOTATIONS | {"type"}
                )
            ]
            if len(branches) != 2 or len(non_null) != 1:
                _fail((*path, "anyOf"), "only a single type plus null is convertible")
            for key in node.keys() - _ANNOTATIONS - {"anyOf"}:
                _fail((*path, key), "assertion siblings of anyOf cannot be converted")
            return convert(non_null[0], (*path, "anyOf", branches.index(non_null[0])))
        kind = node.get("type")
        if isinstance(kind, list):
            non_null_types = [t for t in kind if t != "null"]
            if len(non_null_types) != 1 or len(kind) > 2:
                _fail((*path, "type"), "only a single type plus null is convertible")
            kind = non_null_types[0]
        if "enum" in node:
            for key in ("properties", "items", "additionalProperties"):
                if key in node:
                    _fail(
                        (*path, key),
                        "structural keyword on an enum cannot be converted",
                    )
            choices = [v for v in node["enum"] if v is not None]
            if (
                len(choices) < 2
                or not all(isinstance(v, str) for v in choices)
                or kind not in (None, "string")
            ):
                _fail(
                    (*path, "enum"),
                    "NuExtract enums require at least two string choices",
                )
            return choices
        if kind == "object":
            if "items" in node:
                _fail(
                    (*path, "items"), "array keyword on an object cannot be converted"
                )
            additional = node.get("additionalProperties")
            if isinstance(additional, dict):
                _fail(
                    (*path, "additionalProperties"),
                    "dynamic object keys cannot be converted",
                )
            properties = node.get("properties", {})
            for index, name in enumerate(node.get("required", [])):
                if name not in properties:
                    _fail(
                        (*path, "required", index),
                        "required property has no declared template type",
                    )
            return {
                name: convert(child, (*path, "properties", name))
                for name, child in properties.items()
            }
        if kind == "array":
            for key in ("properties", "additionalProperties"):
                if key in node:
                    _fail(
                        (*path, key), "object keyword on an array cannot be converted"
                    )
            if "items" not in node:
                _fail((*path, "items"), "array item schema is required")
            return [convert(node["items"], (*path, "items"))]
        if kind in ("string", "integer", "number", "boolean"):
            for key in ("properties", "items", "additionalProperties"):
                if key in node:
                    _fail(
                        (*path, key),
                        "structural keyword on a primitive cannot be converted",
                    )
            return kind
        _fail((*path, "type"), "a supported non-null type is required")

    return convert(schema, ())
