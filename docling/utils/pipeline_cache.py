# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import hashlib
import json

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from docling.datamodel.pipeline_options import PipelineOptions


def _qualified_type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _opaque_value(value: object) -> dict[str, str | int]:
    return {
        "type": _qualified_type_name(value),
        "identity": id(value),
    }


def _canonicalize(value: object, active: set[int] | None = None) -> object:
    active = active if active is not None else set()
    value_type = _qualified_type_name(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": value_type, "value": bytes(value).hex()}
    if isinstance(value, (BaseModel, dict, list, tuple, set, frozenset)):
        value_id = id(value)
        if value_id in active:
            raise ValueError("Pipeline options cannot contain cyclic values")
        active.add(value_id)
        try:
            if isinstance(value, BaseModel):
                return {
                    "type": value_type,
                    "fields": _canonicalize(dict(value), active),
                }
            if isinstance(value, dict):
                items = [
                    [_canonicalize(key, active), _canonicalize(item, active)]
                    for key, item in value.items()
                ]
                items.sort(
                    key=lambda item: json.dumps(
                        item[0], sort_keys=True, separators=(",", ":")
                    )
                )
                return {"type": value_type, "items": items}
            items = [_canonicalize(item, active) for item in value]
            if isinstance(value, (set, frozenset)):
                items.sort(
                    key=lambda item: json.dumps(
                        item, sort_keys=True, separators=(",", ":")
                    )
                )
            return {"type": value_type, "items": items}
        finally:
            active.remove(value_id)
    return {
        "type": value_type,
        "value": to_jsonable_python(value, fallback=_opaque_value),
    }


def create_pipeline_options_hash(pipeline_options: PipelineOptions) -> str:
    """Hash public option values and their concrete runtime types.

    Values without a serialization contract are keyed by object identity. This
    prevents unsafe reuse without inspecting private or arbitrary object state.
    """
    payload = json.dumps(
        _canonicalize(pipeline_options),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()
