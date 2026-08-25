# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import hashlib
import json
import re
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import (
    Path,
    PosixPath,
    PurePath,
    PurePosixPath,
    PureWindowsPath,
    WindowsPath,
)
from uuid import UUID

from pydantic import AnyUrl, BaseModel
from pydantic_core import MultiHostUrl, Url

from docling.datamodel.pipeline_options import PipelineOptions

_CONTAINER_TYPES = (dict, list, tuple, set, frozenset)
_LOSSLESS_STRING_TYPES = (
    PurePath,
    PurePosixPath,
    PureWindowsPath,
    Path,
    PosixPath,
    WindowsPath,
    AnyUrl,
    Url,
    MultiHostUrl,
    Decimal,
    UUID,
)


def _qualified_type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _canonicalize(value: object, active: set[int] | None = None) -> object:
    active = active if active is not None else set()
    value_type = _qualified_type_name(value)
    if type(value) in (bytes, bytearray, memoryview):
        return {"type": value_type, "value": bytes(value).hex()}
    if isinstance(value, BaseModel) or type(value) in _CONTAINER_TYPES:
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
            if type(value) is dict:
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
            if type(value) in (set, frozenset):
                items.sort(
                    key=lambda item: json.dumps(
                        item, sort_keys=True, separators=(",", ":")
                    )
                )
            return {"type": value_type, "items": items}
        finally:
            active.remove(value_id)
    if value is None or type(value) in (bool, int, str):
        return {"type": value_type, "value": value}
    if type(value) is float:
        return {"type": value_type, "value": value.hex()}
    if isinstance(value, Enum):
        return {"type": value_type, "value": value.name}
    if type(value) in _LOSSLESS_STRING_TYPES:
        return {"type": value_type, "value": str(value)}
    if type(value) in (datetime, date, time):
        return {"type": value_type, "value": value.isoformat()}
    if type(value) is timedelta:
        return {
            "type": value_type,
            "value": [value.days, value.seconds, value.microseconds],
        }
    if type(value) is re.Pattern:
        return {
            "type": value_type,
            "pattern": _canonicalize(value.pattern, active),
            "flags": value.flags,
        }
    return {
        "type": value_type,
        "identity": id(value),
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
