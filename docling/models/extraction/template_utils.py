# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Call-local schema preflight and bounded NuExtract template conversion."""

from collections.abc import Iterator
from copy import deepcopy
from typing import TYPE_CHECKING, Any, NoReturn
from urllib.parse import unquote

from docling.datamodel.extraction import ExtractionSpec

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


def normalize_spec(spec: ExtractionSpec) -> ExtractionSpec:
    """Own and revalidate all mappings, including model_copy/update inputs."""
    return ExtractionSpec.model_validate(deepcopy(spec.model_dump(mode="python")))


# Annotation keywords that carry free caller text into a generic prompt.
_PROMPT_ANNOTATIONS = {"title", "description", "examples", "default", "$comment"}


def strip_schema_annotations(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``schema`` without free-text annotation keywords.

    Only keywords at schema positions are removed. A property that is named
    ``title`` or ``description`` stays, because property names are keys of a
    schema map, not schema keywords. The cost is linear in the schema size.
    """

    def strip(node: Any) -> Any:
        if not isinstance(node, dict):
            return deepcopy(node)
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _PROMPT_ANNOTATIONS:
                continue
            if key in _SCHEMA_MAPS and isinstance(value, dict):
                out[key] = {name: strip(child) for name, child in value.items()}
            elif key in _SCHEMA_LISTS and isinstance(value, list):
                out[key] = [strip(child) for child in value]
            elif key in _SCHEMA_VALUES:
                out[key] = strip(value)
            else:
                out[key] = deepcopy(value)
        return out

    return strip(schema)


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
        _fail((), "the json_schema must require a top-level object")
    # An empty registry fails closed even if a future schema keyword uses retrieval.
    return Draft202012Validator(schema, registry=Registry())


_NUEXTRACT_HINT = (
    " The json_schema has no equivalent NuExtract template. Add a native_template "
    "for the model input; the json_schema stays the validation contract."
)

_SCALAR_TYPES: tuple[tuple[type, str], ...] = (
    (bool, "boolean"),
    (int, "integer"),
    (float, "number"),
    (str, "string"),
)


def _is_null_branch(branch: Any) -> bool:
    return (
        isinstance(branch, dict)
        and branch.get("type") == "null"
        and branch.keys() <= _ANNOTATIONS | {"type"}
    )


def _scalar_type(value: Any) -> str | None:
    # bool is a subclass of int, so it is checked first.
    for py_type, name in _SCALAR_TYPES:
        if isinstance(value, py_type):
            return name
    return None


def _nullable_branch(
    node: dict[str, Any], path: tuple[str | int, ...]
) -> tuple[Any, tuple[str | int, ...]]:
    """Return the one non-null branch of a nullable ``anyOf``/``oneOf``."""
    unions = [key for key in ("anyOf", "oneOf") if key in node]
    if len(unions) > 1:
        _fail((*path, "oneOf"), "anyOf and oneOf together cannot be converted")
    union = unions[0]
    branches = node[union]
    non_null = [b for b in branches if not _is_null_branch(b)]
    if len(branches) != 2 or len(non_null) != 1:
        _fail((*path, union), "only a single type plus null is convertible")
    for key in node.keys() - _ANNOTATIONS - {union}:
        _fail((*path, key), f"assertion siblings of {union} cannot be converted")
    return non_null[0], (*path, union, branches.index(non_null[0]))


def _enum_template(
    node: dict[str, Any], kind: Any, path: tuple[str | int, ...]
) -> str | list[str]:
    """Convert ``enum``/``const`` to a NuExtract enum or a scalar type name."""
    for key in ("properties", "items", "additionalProperties"):
        if key in node:
            _fail((*path, key), "structural keyword on an enum cannot be converted")
    if "const" in node and "enum" in node:
        _fail((*path, "const"), "const and enum together cannot be converted")
    keyword = "const" if "const" in node else "enum"
    values = [node["const"]] if keyword == "const" else node["enum"]
    choices = [v for v in values if v is not None]
    if len(choices) == 1:
        # A one-value enum is a fixed scalar. NuExtract has no one-value enum,
        # so the guidance is the scalar type; the validator keeps the value.
        scalar = _scalar_type(choices[0])
        if scalar is None or kind not in (None, scalar):
            _fail((*path, keyword), "only a scalar const is convertible")
        return scalar
    if (
        len(choices) < 2
        or not all(isinstance(v, str) for v in choices)
        or kind not in (None, "string")
    ):
        _fail((*path, "enum"), "NuExtract enums require string choices")
    return choices


def _schema_to_nuextract(schema: dict[str, Any]) -> dict[str, Any]:
    """Best-effort conversion of a JSON Schema to a NuExtract native template.

    Converts fixed objects, homogeneous arrays, scalar types, string enums,
    ``const`` and one-value enums (as their scalar type), and nullable unions
    (``type: [T, "null"]``, or ``anyOf``/``oneOf`` with one branch plus null).
    Annotations and constraints are dropped from the guidance; they stay
    authoritative in the validator. Schema-only annotations never select native
    semantic types such as ``verbatim-string``.

    A structure with no equivalent raises ``ValueError``. The message tells the
    caller to add a ``native_template``.
    """
    try:
        return _convert_to_nuextract(schema)
    except ValueError as exc:
        raise ValueError(f"{exc}.{_NUEXTRACT_HINT}") from exc


def _convert_to_nuextract(schema: dict[str, Any]) -> dict[str, Any]:
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
                "const",
                "anyOf",
                "oneOf",
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
        if "anyOf" in node or "oneOf" in node:
            return convert(*_nullable_branch(node, path))
        kind = node.get("type")
        if isinstance(kind, list):
            non_null_types = [t for t in kind if t != "null"]
            if len(non_null_types) != 1 or len(kind) > 2:
                _fail((*path, "type"), "only a single type plus null is convertible")
            kind = non_null_types[0]
        if "const" in node or "enum" in node:
            return _enum_template(node, kind, path)
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


def _vllm_constraint_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Bounded vLLM response_format subset: typed objects/arrays/scalars, enums, nullability.

    Inline non-recursive local references; reject other assertions rather than dropping
    them. Original validation and guidance keep the unchanged Draft 2020-12 contract.
    """
    schema_validator(schema)
    allowed = _ANNOTATIONS | {
        "type",
        "properties",
        "required",
        "items",
        "enum",
        "anyOf",
        "$ref",
        "additionalProperties",
    }

    def convert(node: Any, path: tuple[str | int, ...]) -> Any:
        if isinstance(node, bool):
            return node
        for key in node:
            if key not in allowed:
                _fail((*path, key), "unsupported vLLM constrained-output keyword")
        if "$ref" in node:
            for key in node.keys() - _ANNOTATIONS - {"$ref"}:
                _fail(
                    (*path, key),
                    "assertion siblings of $ref are outside the vLLM subset",
                )
            return convert(_resolve_ref(schema, node["$ref"], (*path, "$ref")), path)
        kind = node.get("type")
        if isinstance(kind, list) and (
            len(kind) > 2 or (len(kind) == 2 and "null" not in kind)
        ):
            _fail((*path, "type"), "only a single type plus null is in the vLLM subset")
        if kind is None and not ({"enum", "anyOf"} & node.keys()):
            _fail(
                (*path, "type"),
                "a type, enum or nullable union is required by the vLLM subset",
            )
        if kind == "array" and "items" not in node:
            _fail(
                (*path, "items"),
                "homogeneous array items are required by the vLLM subset",
            )
        for index, name in enumerate(node.get("required", [])):
            if name not in node.get("properties", {}):
                _fail(
                    (*path, "required", index),
                    "required property has no declared type in the vLLM subset",
                )
        result = {
            key: deepcopy(value)
            for key, value in node.items()
            if key not in {"$schema", "$id", "$defs", "definitions", "$comment"}
        }
        if "properties" in node:
            result["properties"] = {
                name: convert(child, (*path, "properties", name))
                for name, child in node["properties"].items()
            }
        if "items" in node:
            result["items"] = convert(node["items"], (*path, "items"))
        if "additionalProperties" in node:
            if not isinstance(node["additionalProperties"], bool):
                _fail(
                    (*path, "additionalProperties"),
                    "dynamic keys are outside the vLLM subset",
                )
        if "anyOf" in node:
            branches = node["anyOf"]
            if len(branches) != 2 or not any(
                isinstance(branch, dict)
                and branch.get("type") == "null"
                and branch.keys() <= _ANNOTATIONS | {"type"}
                for branch in branches
            ):
                _fail((*path, "anyOf"), "only nullable unions are in the vLLM subset")
            result["anyOf"] = [
                convert(child, (*path, "anyOf", index))
                for index, child in enumerate(branches)
            ]
        return result

    return convert(schema, ())
