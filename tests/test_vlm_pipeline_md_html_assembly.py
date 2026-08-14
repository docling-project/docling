"""Regression tests for VLM pipeline MARKDOWN/HTML document assembly.

The legacy assembly path appended per-page items with append_child_item, which
raises ValueError when items already have children (nested lists, tables, etc.).
Assembly must merge page documents via DoclingDocument.concatenate instead.

Related: https://github.com/docling-project/docling/issues/2476
"""

from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage

from docling.backend.html_backend import HTMLDocumentBackend
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.datamodel.base_models import InputFormat, Page, PagePredictions, VlmPrediction
from docling.datamodel.document import ConversionResult
from docling.pipeline.vlm_pipeline import VlmPipeline

pytestmark = pytest.mark.ml_vlm


def _make_conv_res(page_texts: list[str]) -> ConversionResult:
    conv_res = MagicMock(spec=ConversionResult)
    conv_res.input = MagicMock()
    conv_res.input.file = MagicMock()
    conv_res.input.file.name = "test.pdf"

    page_image = PILImage.new("RGB", (100, 100), "white")
    pages: list[Page] = []
    for page_no, text in enumerate(page_texts, start=1):
        page = Page(page_no=page_no)
        page.predictions = PagePredictions(vlm_response=VlmPrediction(text=text))
        page._image_cache = {1.0: page_image}
        pages.append(page)

    conv_res.pages = pages
    return conv_res


@pytest.fixture
def pipeline() -> VlmPipeline:
    return VlmPipeline.__new__(VlmPipeline)


@pytest.mark.parametrize(
    "page_texts",
    [
        ["# Title\n\nParagraph on page one."],
        [
            "# Page 1\n\n- item 1\n- item 2\n  - nested item",
            "# Page 2\n\n- item 1\n- item 2\n  - sub item 1 <tag>\n  - sub item 2",
        ],
        [
            "| A | B |\n|---|---|\n| 1 | 2 |",
            "```markdown\n# Page two\n\nMore text.\n```",
        ],
    ],
)
def test_vlm_markdown_assembly_concatenates_nested_pages(
    pipeline: VlmPipeline, page_texts: list[str]
) -> None:
    conv_res = _make_conv_res(page_texts)

    document = pipeline._convert_text_with_backend(
        conv_res, InputFormat.MD, MarkdownDocumentBackend
    )

    assert len(document.pages) == len(page_texts)
    assert sum(1 for _ in document.iterate_items()) > 0


@pytest.mark.parametrize(
    "page_texts",
    [
        ["<html><body><h1>Title</h1><p>Text</p></body></html>"],
        [
            "<html><body><h1>Page 1</h1><ul><li>one</li></ul></body></html>",
            "<html><body><h1>Page 2</h1><ul><li>two<ul><li>nested</li></ul></li></ul></body></html>",
        ],
    ],
)
def test_vlm_html_assembly_concatenates_nested_pages(
    pipeline: VlmPipeline, page_texts: list[str]
) -> None:
    conv_res = _make_conv_res(page_texts)

    document = pipeline._convert_text_with_backend(
        conv_res, InputFormat.HTML, HTMLDocumentBackend
    )

    assert len(document.pages) == len(page_texts)
    assert sum(1 for _ in document.iterate_items()) > 0
