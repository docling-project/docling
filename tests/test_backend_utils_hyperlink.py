# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
from pydantic import AnyUrl

from docling.backend.utils.hyperlink import parse_hyperlink


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("https://example.com/a", AnyUrl("https://example.com/a")),
        ("sub/page.html", Path("sub/page.html")),
        # A Windows drive is not a URL scheme.
        (r"C:\docs\page.html", Path(r"C:\docs\page.html")),
        ("https://api.example.com:{port}/v1", None),
        ("https://[broken", None),
        ("", None),
    ],
)
def test_parse_hyperlink(address: str, expected: AnyUrl | Path | None) -> None:
    assert parse_hyperlink(address) == expected
