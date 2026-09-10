# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pytest

from docling.models.stages.chart_extraction.granite_vision import (
    _granite_vision_4_trust_remote_code,
)


@pytest.mark.parametrize(
    ("installed_transformers", "expected"),
    [
        # No native granite4_vision yet: the bundled remote code is the only option.
        ("4.57.3", True),
        ("5.7.0", True),
        # Native implementation available; the remote code breaks on >=5.9
        # (create_causal_mask lost `cache_position`), so it must not be preferred.
        ("5.8.0", False),
        ("5.8.1", False),
        ("5.9.0", False),
        ("5.16.1", False),
        ("5.17.0", False),
    ],
)
def test_granite_vision_4_only_trusts_remote_code_without_native_support(
    installed_transformers: str, expected: bool
) -> None:
    assert _granite_vision_4_trust_remote_code(installed_transformers) is expected
