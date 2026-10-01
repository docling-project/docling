# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Loading helpers for ibm-granite/granite-vision-4.1-4b."""

from packaging import version

GRANITE_VISION_4_REPO_ID = "ibm-granite/granite-vision-4.1-4b"

# First transformers release with the native granite4_vision model type.
_GRANITE_VISION_4_NATIVE_TRANSFORMERS = version.parse("5.8.0")


def granite_vision_4_needs_remote_code(transformers_version: str) -> bool:
    """Return whether granite-vision-4.1-4b has to load through its bundled code.

    The modeling code bundled with the pinned checkpoint fails on
    transformers>=5.9, and trust_remote_code=True makes transformers prefer it
    over the native granite4_vision implementation. The bundled code is only
    needed where the native implementation does not exist.

    Args:
        transformers_version: The installed transformers version.

    Returns:
        True if the installed transformers has no native granite4_vision
        implementation, False otherwise.
    """
    return version.parse(transformers_version) < _GRANITE_VISION_4_NATIVE_TRANSFORMERS
