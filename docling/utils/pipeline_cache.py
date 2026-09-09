# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import hashlib

from docling.datamodel.pipeline_options import PipelineOptions


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
    payload = type(pipeline_options).__qualname__ + pipeline_options.model_dump_json(
        serialize_as_any=True
    )
    return hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()
