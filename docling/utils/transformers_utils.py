# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Compatibility helpers for loading HuggingFace Transformers models."""

import importlib.metadata
from typing import Any

from packaging import version


def transformers_dtype_kwargs(
    torch_dtype: Any,
    *,
    transformers_version: str | None = None,
) -> dict[str, Any]:
    """Return the version-appropriate dtype keyword argument for ``from_pretrained``.

    ``transformers<5`` expects ``torch_dtype`` (passing ``dtype`` raises
    ``TypeError`` on ``transformers<4.56`` and on remote-code v4 models),
    whereas ``transformers>=5`` expects ``dtype``. When ``torch_dtype`` is
    ``None``, return an empty dict so neither keyword is passed.
    """
    if torch_dtype is None:
        return {}

    resolved_version = (
        transformers_version
        if transformers_version is not None
        else importlib.metadata.version("transformers")
    )
    kwarg_name = (
        "dtype" if version.parse(resolved_version).major >= 5 else "torch_dtype"
    )
    return {kwarg_name: torch_dtype}
