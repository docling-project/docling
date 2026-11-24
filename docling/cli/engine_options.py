# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Generic ``KEY=VALUE`` options for the engines that the CLI selects by name."""

import json
from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin

import typer
from pydantic import ValidationError
from pydantic_core import PydanticUndefined

if TYPE_CHECKING:
    from docling.datamodel.pipeline_options import BaseOptions
    from docling.models.factories.base_factory import BaseFactory


def parse_engine_option_pairs(
    values: list[str] | None, param_hint: str
) -> dict[str, Any]:
    """Parse repeated ``KEY=VALUE`` items into keyword arguments.

    The value is decoded as JSON when possible (``0.6``, ``false``,
    ``["en","de"]``); otherwise the raw string is kept.
    """
    parsed: dict[str, Any] = {}
    for item in values or []:
        key, separator, raw_value = item.partition("=")
        key = key.strip()
        if not separator or not key:
            raise typer.BadParameter(
                f"expected KEY=VALUE, got {item!r}", param_hint=param_hint
            )
        if key in parsed:
            raise typer.BadParameter(
                f"key {key!r} is given more than once", param_hint=param_hint
            )
        try:
            parsed[key] = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed[key] = raw_value
    return parsed


def find_options_class(factory: "BaseFactory", kind: str) -> type["BaseOptions"]:
    for options_cls in factory.classes:
        if options_cls.kind == kind:
            return options_cls
    known = ", ".join(sorted(factory.registered_kind))
    raise typer.BadParameter(f"unknown engine {kind!r}; known engines: {known}")


def create_engine_options(
    factory: "BaseFactory",
    kind: str,
    *,
    flag_kwargs: Mapping[str, Any],
    flag_names: Mapping[str, str],
    explicit_flag_fields: set[str],
    option_values: list[str] | None,
    param_hint: str,
) -> "BaseOptions":
    """Create the options of engine ``kind`` from dedicated flags and ``KEY=VALUE`` items.

    Args:
        factory: The factory that registers the engine.
        kind: The selected engine.
        flag_kwargs: Fields set from dedicated CLI flags, such as ``--ocr-lang``.
        flag_names: The CLI flag for each field in ``flag_kwargs``.
        explicit_flag_fields: Fields in ``flag_kwargs`` that the user set
            explicitly. A ``KEY=VALUE`` item for one of these fields is an error.
        option_values: The raw ``KEY=VALUE`` items.
        param_hint: The repeatable option that takes the items, for error messages.
    """
    options_cls = find_options_class(factory, kind)
    extra_kwargs = parse_engine_option_pairs(option_values, param_hint)

    valid_keys = sorted(options_cls.model_fields)
    for key in extra_kwargs:
        if key not in options_cls.model_fields:
            raise typer.BadParameter(
                f"unknown key {key!r} for engine {kind!r}; "
                f"valid keys: {', '.join(valid_keys)}",
                param_hint=param_hint,
            )
        if key in explicit_flag_fields:
            raise typer.BadParameter(
                f"key {key!r} is also set by {flag_names[key]}; use only one of them",
                param_hint=param_hint,
            )

    kwargs = {**flag_kwargs, **extra_kwargs}
    try:
        return factory.create_options(kind=kind, **kwargs)
    except ValidationError as err:
        error = err.errors()[0]
        field = str(error["loc"][0]) if error["loc"] else ""
        message = error["msg"].removeprefix("Value error, ")
        if field in extra_kwargs:
            raise typer.BadParameter(
                f"{field}: {message}", param_hint=param_hint
            ) from err
        raise typer.BadParameter(
            message, param_hint=flag_names.get(field, param_hint)
        ) from err


def describe_options_class(
    options_cls: type["BaseOptions"],
) -> list[tuple[str, str, str, str]]:
    """Return ``(name, type, default, description)`` for each field of ``options_cls``."""
    rows = []
    for name, field in options_cls.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            type_name = f"[{'|'.join(str(member.value) for member in annotation)}]"
        elif get_origin(annotation) is Literal:
            type_name = f"[{'|'.join(str(value) for value in get_args(annotation))}]"
        elif isinstance(annotation, type) and get_origin(annotation) is None:
            type_name = annotation.__name__
        else:
            type_name = str(annotation).replace("typing.", "")
        if field.is_required():
            default = "required"
        else:
            default_value = field.get_default(call_default_factory=True)
            default = (
                ""
                if default_value is PydanticUndefined
                else _format_default(default_value)
            )
        rows.append((name, type_name, default, field.description or ""))
    return rows


def _format_default(value: Any) -> str:
    """Show a default as the JSON that ``KEY=VALUE`` accepts, when it can."""
    try:
        return json.dumps(value, default=lambda obj: obj.model_dump(mode="json"))
    except (TypeError, ValueError, AttributeError):
        return repr(value)
