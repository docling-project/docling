# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import hashlib
import json
from typing import Any

from pydantic import BaseModel
from pydantic_core import PydanticSerializationError

from docling.datamodel.pipeline_options import PipelineOptions


def _dump_pipeline_options_fallback(pipeline_options: PipelineOptions) -> str:
    """Fallback serialization when serialize_as_any fails with circular reference.

    In pydantic < 2.12.0, serialize_as_any=True can raise a false-positive
    PydanticSerializationError: ValueError: Circular reference detected (id repeated)
    on shared default objects. This fallback serializes standard model data and
    explicitly captures concrete types and schemas of nested BaseModel fields
    to avoid collisions without triggering cycle detection.
    """
    try:
        base_dump = pipeline_options.model_dump(mode="json")
    except Exception:
        base_dump = {}

    sub_dumps: dict[str, Any] = {}
    for k, v in sorted(pipeline_options.__dict__.items()):
        if not k.startswith("_") and isinstance(v, BaseModel):
            try:
                sub_dumps[k] = (type(v).__qualname__, v.model_dump(mode="json"))
            except Exception:
                sub_dumps[k] = type(v).__qualname__

    combined = {
        "base": base_dump,
        "sub": sub_dumps,
    }
    return json.dumps(combined, sort_keys=True, default=str)


def create_pipeline_options_hash(pipeline_options: PipelineOptions) -> str:
    """Build a stable cache key from the public option values.

    ``serialize_as_any=True`` serializes every value by its concrete runtime
    type, so a concrete subtype assigned to a base-typed field (e.g. an API vs.
    a local VLM picture-description backend) keeps its distinguishing fields
    instead of being truncated to the base schema and colliding. The options
    class name guards against two option classes that dump identically.

    # ponytail: serialize_as_any masks SecretStr/opaque objects to a constant;
    # none are reachable from PipelineOptions today. If such a field is added,
    # key it explicitly here rather than reintroducing a full type walker.
    """
    try:
        json_data = pipeline_options.model_dump_json(serialize_as_any=True)
    except (PydanticSerializationError, ValueError):
        json_data = _dump_pipeline_options_fallback(pipeline_options)

    payload = type(pipeline_options).__qualname__ + json_data
    return hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()
