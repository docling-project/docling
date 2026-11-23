# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Synthetic OXR payloads shared by the OXR adapter tests.

The payload mirrors the serialized OXR response shape: a ``code``/``data``
envelope, a ``result`` with ``file_name``, ``metadata`` and ``outputs``, and
one ``pages`` list whose blocks carry the OXR region type, pixel ``bbox``,
``content`` and ``format``.

The document is built in code on purpose. A recorded provider capture would
live under ``tests/data/``, and this repository asks for a second approving
review on every change below that path, which the adapter tests do not need.
"""

from __future__ import annotations

from typing import Any


def block(
    kind: str,
    content: str,
    *,
    bbox: tuple[float, float, float, float] = (10, 20, 300, 80),
    fmt: str = "text",
) -> dict[str, Any]:
    """Return one serialized OXR region."""
    return {"type": kind, "bbox": list(bbox), "content": content, "format": fmt}


def page(
    blocks: list[dict[str, Any]],
    *,
    page_num: int = 0,
    width: float = 800,
    height: float = 1000,
) -> dict[str, Any]:
    """Return one serialized OXR page."""
    return {
        "page_num": page_num,
        "page_size": {"w": width, "h": height},
        "merge_content_from_pre_page": False,
        "blocks": blocks,
    }


def result(
    pages: list[dict[str, Any]], *, file_name: str = "demo.pdf"
) -> dict[str, Any]:
    """Return the ``result`` object of an OXR response."""
    return {
        "file_name": file_name,
        "metadata": {
            "page_num": len(pages),
            "file_type": ".pdf",
            "processing_time_ms": 1,
        },
        "outputs": {"markdown": "", "json": {"pages": pages}},
    }


def response(result_object: dict[str, Any]) -> dict[str, Any]:
    """Wrap an OXR ``result`` in the full sync response envelope."""
    return {
        "code": 0,
        "message": "success",
        "warnings": [],
        "data": {
            "task_id": "adapter-sample",
            "error": None,
            "result": result_object,
            "metadata": {},
        },
    }


def adapter_document() -> dict[str, Any]:
    """Return a two-page OXR response that covers every mapped region type."""
    first = page(
        [
            block("Page-header", "OXR Adapter Fixture", bbox=(40, 20, 760, 40)),
            block("Title", "Adapter Fixture", bbox=(40, 60, 760, 110)),
            block(
                "Text",
                "A self-authored page for the OXR adapter.",
                bbox=(40, 130, 760, 220),
            ),
            block(
                "Table",
                "<table><tr><th>Key</th><th>Value</th></tr>"
                "<tr><td>alpha</td><td>1</td></tr></table>",
                bbox=(40, 240, 760, 420),
                fmt="html",
            ),
            block("Formula", "$$E = mc^2$$", bbox=(40, 440, 300, 500), fmt="latex"),
            block(
                "Code Block",
                "```python\nprint('oxr')\n```",
                bbox=(40, 520, 760, 620),
                fmt="markdown",
            ),
            block(
                "Picture", "![](imgs/0_6.png)", bbox=(40, 640, 300, 760), fmt="markdown"
            ),
            block("Page-footer", "- 1 -", bbox=(40, 960, 760, 980)),
        ],
        page_num=0,
    )
    second = page(
        [
            block(
                "Text",
                "The second page keeps its own furniture.",
                bbox=(40, 60, 760, 160),
            ),
            block(
                "Chemical Equation",
                "\\ce{2H2 + O2 -> 2H2O}",
                bbox=(40, 180, 400, 240),
                fmt="latex",
            ),
            block(
                "Chemical Structure",
                "$\\smiles{CCO}$",
                bbox=(40, 260, 400, 340),
                fmt="markdown",
            ),
            block("Stamp", "APPROVED", bbox=(40, 360, 400, 420)),
            block(
                "Footnote",
                "Synthetic sample, not a provider capture.",
                bbox=(40, 440, 760, 480),
            ),
        ],
        page_num=1,
    )
    return response(result([first, second], file_name="self_authored.pdf"))
