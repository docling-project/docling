# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

from docling.datamodel.base_models import DocItemLabel
from docling.datamodel.vlm_prompts import DOCLING_BASE_PAGE_PROMPT
from docling.experimental.pipeline.threaded_layout_vlm_pipeline import (
    _build_layout_aware_prompt,
)


class _BBox:
    def as_tuple(self):
        return (0.0, 0.0, 50.0, 20.0)


def test_layout_vlm_prompt_discards_base_prompt():
    page = SimpleNamespace(
        page_no=1,
        size=SimpleNamespace(width=100.0, height=200.0),
        predictions=SimpleNamespace(
            layout=SimpleNamespace(
                clusters=[SimpleNamespace(label=DocItemLabel.TEXT, bbox=_BBox())]
            )
        ),
    )

    prompt = _build_layout_aware_prompt(DOCLING_BASE_PAGE_PROMPT, page)

    assert prompt.startswith("<layout>\n")
    assert DOCLING_BASE_PAGE_PROMPT not in prompt


def test_layout_vlm_prompt_without_layout_is_empty():
    page = SimpleNamespace(
        page_no=1,
        size=SimpleNamespace(width=100.0, height=200.0),
        predictions=SimpleNamespace(layout=None),
    )

    assert _build_layout_aware_prompt(DOCLING_BASE_PAGE_PROMPT, page) == ""


def test_layout_vlm_custom_prompt_is_unchanged():
    assert _build_layout_aware_prompt("custom prompt", None) == "custom prompt"
