# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Heading depth must survive VLM response parsing and Markdown export."""

import json
from pathlib import PurePath
from types import SimpleNamespace

import pytest
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    SectionHeaderItem,
    Size,
    TitleItem,
)

from docling.datamodel.base_models import Page, PagePredictions, VlmPrediction
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.utils.chandra_utils import _add_html_fragment, parse_chandra_html
from docling.utils.deepseekocr_utils import (
    parse_deepseekocr_markdown,
    parse_unlimited_ocr_markdown,
)
from docling.utils.dots_utils import parse_dots_json
from docling.utils.lightonocr_utils import parse_lightonocr_markdown
from docling.utils.nemotron_parse_utils import parse_nemotron_parse_v2
from docling.utils.nemotron_parse_v3_5_utils import parse_nemotron_parse_v3_5
from docling.utils.paddleocr_vl_utils import parse_paddleocr_vl_result
from docling.utils.unlimited_ocr2_utils import parse_unlimited_ocr2

_SIZE = Size(width=600, height=800)
# A block is (is_title, markup depth or None for no heading markup).
_Block = tuple[bool, int | None]
_FORMATS = {
    "chandra": ResponseFormat.CHANDRA_HTML,
    "dots": ResponseFormat.DOTS_JSON,
    "nemotron_v2": ResponseFormat.NEMOTRON_PARSE_V2,
    "nemotron_v3_5": ResponseFormat.NEMOTRON_PARSE_V3_5,
    "deepseek": ResponseFormat.DEEPSEEKOCR_MARKDOWN,
    "unlimited": ResponseFormat.UNLIMITED_OCR_MARKDOWN,
    "unlimited2": ResponseFormat.UNLIMITED_OCR2,
    "lighton": ResponseFormat.LIGHTONOCR_MARKDOWN,
}
_PARSERS = [*_FORMATS, "paddle"]


def _markdown(index: int, depth: int | None) -> str:
    return f"{'#' * depth} H{index}" if depth is not None else f"H{index}"


def _response(parser: str, blocks: list[_Block]) -> str:
    """Render one page of model output with one block per heading."""
    parts = []
    for index, (is_title, depth) in enumerate(blocks):
        y = 100 + 50 * index
        text = _markdown(index, depth)
        if parser == "chandra":
            label = "Title" if is_title else "Section-Header"
            body = f"<h{depth}>H{index}</h{depth}>" if depth is not None else text
            parts.append(
                f'<div data-bbox="100 {y} 900 {y + 40}" data-label="{label}">'
                f"{body}</div>"
            )
        elif parser == "dots":
            category = "Title" if is_title else "Section-header"
            parts.append(
                json.dumps(
                    {"category": category, "bbox": [60, y, 540, y + 40], "text": text}
                )
            )
        elif parser in {"nemotron_v2", "nemotron_v3_5"}:
            label = "Title" if is_title else "Section-header"
            parts.append(
                f"<x_0.1><y_{y / 1000}>{text}<x_0.9><y_{(y + 40) / 1000}>"
                f"<class_{label}>\n\n"
            )
        elif parser == "deepseek":
            label = "title" if is_title else "sub_title"
            parts.append(
                f"<|ref|>{label}<|/ref|><|det|>[[100, {y}, 900, {y + 40}]]<|/det|>\n"
                f"{text}\n\n"
            )
        elif parser == "unlimited":
            label = "title" if is_title else "sub_title"
            parts.append(f"<|det|>{label} [100, {y}, 900, {y + 40}]<|/det|>{text}\n")
        elif parser == "unlimited2":
            # This format has one label: a level is what marks a section header.
            level = "" if is_title else f"<LEVEL{depth or 2}>"
            parts.append(
                f"<BLOCK>title{level}[100, {y}, 900, {y + 40}]<FORK>H{index}\n"
            )
        elif parser == "lighton":
            label = "title" if is_title else "section"
            parts.append(f"![{label}](100,{y},900,{y + 40}) {text}\n")
        else:
            raise AssertionError(parser)
    if parser == "dots":
        return f"[{', '.join(parts)}]"
    return "".join(parts)


def _parse(parser: str, blocks: list[_Block], page_no: int = 1) -> DoclingDocument:
    """Map one page with the standalone mapper and its default normalization."""
    if parser == "paddle":
        payload = {
            "width": _SIZE.width,
            "height": _SIZE.height,
            "model_settings": {"format_block_content": True},
            "parsing_res_list": [
                {
                    "block_label": "doc_title" if is_title else "paragraph_title",
                    "block_content": _markdown(index, depth),
                    "block_bbox": [60, 100 + 50 * index, 540, 140 + 50 * index],
                }
                for index, (is_title, depth) in enumerate(blocks)
            ],
        }
        return parse_paddleocr_vl_result(payload, filename="page.pdf", page_no=page_no)
    content = _response(parser, blocks)
    if parser == "chandra":
        return parse_chandra_html(content, _SIZE, page_no=page_no)
    if parser == "dots":
        return parse_dots_json(content, _SIZE, page_no=page_no)
    if parser == "nemotron_v2":
        return parse_nemotron_parse_v2(content, _SIZE, _SIZE, page_no=page_no)
    if parser == "nemotron_v3_5":
        return parse_nemotron_parse_v3_5(content, _SIZE, page_no=page_no)
    if parser == "deepseek":
        return parse_deepseekocr_markdown(content, _SIZE, page_no=page_no)
    if parser == "unlimited2":
        return parse_unlimited_ocr2(content, _SIZE, page_no=page_no)
    if parser == "lighton":
        return parse_lightonocr_markdown(content, _SIZE, page_no=page_no)
    return parse_unlimited_ocr_markdown(content, _SIZE, page_no=page_no)


def _levels(doc: DoclingDocument) -> list[int]:
    return [item.level for item in doc.texts if isinstance(item, SectionHeaderItem)]


@pytest.mark.parametrize("parser", _PARSERS)
def test_depth_one_section_keeps_every_depth_distinct(parser: str) -> None:
    doc = _parse(parser, [(False, 1), (False, 2), (False, 3)])

    assert all(isinstance(item, SectionHeaderItem) for item in doc.texts)
    assert _levels(doc) == [1, 2, 3]
    assert doc.export_to_markdown() == "## H0\n\n### H1\n\n#### H2"


@pytest.mark.parametrize("parser", _PARSERS)
def test_source_depth_survives_export_without_depth_one(parser: str) -> None:
    doc = _parse(parser, [(False, 2), (False, 3), (False, 6)], page_no=2)

    assert _levels(doc) == [1, 2, 5]
    assert all(item.prov[0].page_no == 2 for item in doc.texts)
    assert doc.export_to_markdown() == "## H0\n\n### H1\n\n###### H2"


@pytest.mark.parametrize("parser", _PARSERS)
def test_section_without_markup_is_depth_two(parser: str) -> None:
    assert _levels(_parse(parser, [(False, None), (False, 3)])) == [1, 2]
    assert _levels(_parse(parser, [(False, 1), (False, None)])) == [1, 2]


# LightOnOCR-3 uses `title` for all headings, so its depth decides instead.
@pytest.mark.parametrize("parser", [p for p in _PARSERS if p != "lighton"])
def test_layout_label_alone_decides_title(parser: str) -> None:
    doc = _parse(parser, [(True, 2), (False, 1), (False, 2)])

    assert isinstance(doc.texts[0], TitleItem)
    assert doc.texts[0].text == "H0"
    assert _levels(doc) == [1, 2]
    assert doc.export_to_markdown() == "# H0\n\n## H1\n\n### H2"


def test_lighton_title_depth_decides_title() -> None:
    doc = _parse("lighton", [(True, 1), (True, None), (True, 2), (False, 3)])

    assert [type(item) for item in doc.texts] == [
        TitleItem,
        TitleItem,
        SectionHeaderItem,
        SectionHeaderItem,
    ]
    assert doc.export_to_markdown() == "# H0\n\n# H1\n\n## H2\n\n### H3"


def test_html_fragment_records_raw_depth() -> None:
    doc = DoclingDocument(name="fragment")
    doc.add_page(page_no=2, size=_SIZE)
    for depth in (2, 3):
        _add_html_fragment(doc, f"<h{depth}>Method</h{depth}>", label=DocItemLabel.TEXT)

    # The calling mapper normalizes; the shared fragment helper must not.
    assert _levels(doc) == [2, 3]


@pytest.mark.parametrize("parser", list(_FORMATS))
def test_pipeline_normalizes_joined_pages_once(parser: str) -> None:
    pipeline = VlmPipeline.__new__(VlmPipeline)
    pipeline.pipeline_options = SimpleNamespace(
        generate_page_images=False,
        generate_picture_images=False,
        images_scale=1.0,
        vlm_options=InlineVlmOptions(
            prompt="",
            repo_id="test",
            response_format=_FORMATS[parser],
            inference_framework=InferenceFramework.TRANSFORMERS,
            scale=1.0,
        ),
    )
    pipeline.force_backend_text = False
    conv_res = SimpleNamespace(
        input=SimpleNamespace(file=PurePath("test.pdf")), errors=[]
    )

    def join(pages: dict[int, list[_Block]]) -> DoclingDocument:
        page_documents = []
        for page_no, blocks in pages.items():
            page = Page(
                page_no=page_no,
                size=_SIZE,
                predictions=PagePredictions(
                    vlm_response=VlmPrediction(text=_response(parser, blocks))
                ),
            )
            page_documents.append(
                (page_no, pipeline._finalize_page_document(conv_res, page))
            )
        return pipeline._join_page_documents(page_documents)

    # Only page 1 has a depth-1 section, yet "##" has one level on both pages.
    doc = join({1: [(False, 1), (False, 2)], 2: [(False, 2)]})
    assert [(item.prov[0].page_no, item.level) for item in doc.texts] == [
        (1, 1),
        (1, 2),
        (2, 2),
    ]
    # A second normalization would turn "###" into "##".
    doc = join({1: [(False, 3)], 2: [(False, 3)]})
    assert _levels(doc) == [2, 2]
    assert doc.export_to_markdown() == "### H0\n\n### H0"
    assert not conv_res.errors


def test_formatted_html_heading_keeps_one_coordinate_owner() -> None:
    doc = parse_chandra_html(
        '<div data-bbox="100 100 900 200" data-label="Section-Header">'
        "<h3><b>Method</b> and results</h3></div>",
        Size(width=600, height=800),
        page_no=1,
    )
    assert isinstance(doc.texts[0], SectionHeaderItem)
    assert doc.texts[0].level == 2
    assert doc.texts[0].prov
    assert all(not item.prov for item in doc.texts[1:])
    assert doc.export_to_markdown() == "### **Method** and results"
