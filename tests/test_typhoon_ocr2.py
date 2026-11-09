# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT
# Typhoon OCR V2 by the Typhoon team: https://huggingface.co/typhoon-ai/typhoon-ocr2-2b

"""Typhoon OCR V2 presets and layout parsing, without model downloads.

The replies in ``tests/data/json_typhoon_ocr2/sources`` are real model output.
Docling captured them unedited with the ``typhoon_ocr2_2b`` preset on the MLX
engine; each ``.source.txt`` file gives the page, the model revision and the
runtime. ``thai_rainwater.pdf`` is a page written for these tests. Every other
reply below is written by hand from the output format in the model card and the
typhoon-ocr2 reference package. The API tests send their requests to a fake
OpenAI-compatible server on a local socket. Set ``DOCLING_TEST_TYPHOON_OCR2=1`` to
also run a real conversion.
"""

import base64
import hashlib
import io
import json
import logging
import os
import re
import unicodedata
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from docling_core.types.doc import (
    ContentLayer,
    CoordOrigin,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    NodeItem,
    PictureItem,
    RichTableCell,
    SectionHeaderItem,
    Size,
    TableItem,
    TextItem,
)
from PIL import Image
from pydantic import AnyUrl

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    ResponseFormat,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.datamodel.vlm_engine_options import (
    ApiVlmEngineOptions,
    AutoInlineVlmEngineOptions,
)
from docling.datamodel.vlm_prompts import TYPHOON_OCR2_PARSE_PROMPT
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)
from docling.models.inference_engines.vlm.base import VlmEngineType
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.utils.typhoon_ocr2_utils import parse_typhoon_ocr2_json
from tests.fakes.http_service import FakeService
from tests.fakes.openai_compatible import FakeOpenAiApi

_SIZE = Size(width=612, height=792)
# The prompt block of the 2B and 0.8B model cards.
_PROMPT_SHA256 = "b55c0349b52deac3958a9f356966c99acf50fbb5be25f717f7d2c90cf4dac27a"
_DATA = Path(__file__).parent / "data"
_PDF = _DATA / "pdf" / "sources" / "2305.03393v1-pg9.pdf"
_REPLIES = _DATA / "json_typhoon_ocr2" / "sources"
_THAI_PDF = _REPLIES / "thai_rainwater.pdf"
# The text elements of the Thai page, in reading order. The 2B reply has them
# exactly as the page prints them.
_THAI_TEXTS = [
    "รายงานประจำเดือน ฉบับทดสอบที่ ๑๑/๒๕๖๙",
    "การเก็บน้ำฝนของชุมชนตัวอย่าง",
    "ชุมชนตัวอย่างเริ่มเก็บน้ำฝนไว้ใช้ในช่วงหน้าแล้ง โดยติดตั้งถังเก็บน้ำขนาด 2,000 ลิตร จำนวน ๑๒ ถัง ให้แก่บ้านที่อยู่ห่างจากแหล่งน้ำมากที่สุด",
    "ถังทุกใบตั้งอยู่บนฐานปูนและมีก๊อกน้ำกับป้ายบอกวันที่ล้างถังครั้งล่าสุด",
    "ผลการติดตามพบว่าแต่ละครัวเรือนใช้น้ำฝนได้นานขึ้นเรื่อย ๆ โดยเฉลี่ย 45 วันต่อปี และค่าน้ำประปาลดลงร้อยละ ๑๘ ร้านก๋วยเตี๋ยวในชุมชนก็ใช้น้ำฝนล้างผัก ส่วนเด็ก ๆ ช่วยกันจดระดับน้ำทุกเช้าแล้วนำไปรดแปลงผักที่ใส่ปุ๋ยหมัก",
    "หน้า ๑",
]


def _real_reply(name: str) -> str:
    """Return a captured model reply as it is stored, without a final newline."""
    return (_REPLIES / f"{name}.txt").read_text(encoding="utf-8")


def _element(
    category: str, text: str, bbox: Sequence[int] = (100, 100, 900, 140)
) -> dict[str, Any]:
    return {"bbox": list(bbox), "category": category, "text": text}


def _reply(*elements: dict[str, Any]) -> str:
    """Write the elements as the model does: a JSON list in a ``json`` fence."""
    return f"```json\n{json.dumps(list(elements), ensure_ascii=False)}\n```"


def _parse(content: str, size: Size = _SIZE, **kwargs: Any) -> DoclingDocument:
    return parse_typhoon_ocr2_json(
        content, size, page_no=3, filename="page.pdf", **kwargs
    )


def _blocks(doc: DoclingDocument) -> list[NodeItem]:
    """Return the top-level items, which the parser adds once per element."""
    return [ref.resolve(doc) for ref in doc.body.children]


def _box(item: DocItem) -> tuple[float, float, float, float]:
    bbox = item.prov[0].bbox
    return bbox.l, bbox.t, bbox.r, bbox.b


def _summary(doc: DoclingDocument) -> list[tuple[DocItemLabel, str, tuple]]:
    return [
        (item.label, item.text if isinstance(item, TextItem) else "", _box(item))
        for item in _blocks(doc)
        if isinstance(item, DocItem)
    ]


@pytest.mark.parametrize(
    ("category", "text", "label", "layer"),
    [
        ("header", "Annual Report", DocItemLabel.PAGE_HEADER, ContentLayer.FURNITURE),
        ("footer", "12", DocItemLabel.PAGE_FOOTER, ContentLayer.FURNITURE),
        ("title", "Introduction", DocItemLabel.SECTION_HEADER, ContentLayer.BODY),
        ("text", "Body text.", DocItemLabel.TEXT, ContentLayer.BODY),
        ("figure", "", DocItemLabel.PICTURE, ContentLayer.BODY),
        (
            "table",
            "<table><tr><td>1</td></tr></table>",
            DocItemLabel.TABLE,
            ContentLayer.BODY,
        ),
        ("formula", "$$x^2$$", DocItemLabel.FORMULA, ContentLayer.BODY),
        ("figure_caption", "Figure 1.", DocItemLabel.CAPTION, ContentLayer.BODY),
        ("table_caption", "Table 1.", DocItemLabel.CAPTION, ContentLayer.BODY),
        ("formula_caption", "(1)", DocItemLabel.CAPTION, ContentLayer.BODY),
        (
            "figure_footnote",
            "Source: survey.",
            DocItemLabel.FOOTNOTE,
            ContentLayer.BODY,
        ),
        ("table_footnote", "Values in THB.", DocItemLabel.FOOTNOTE, ContentLayer.BODY),
        ("page_footnote", "1 See page 4.", DocItemLabel.FOOTNOTE, ContentLayer.BODY),
        # The category is read without case and surrounding spaces.
        (" Footer ", "12", DocItemLabel.PAGE_FOOTER, ContentLayer.FURNITURE),
        # The model has no list category; an unknown one stays text.
        ("list", "- one", DocItemLabel.TEXT, ContentLayer.BODY),
    ],
)
def test_categories_map_to_labels_and_layers(
    category: str,
    text: str,
    label: DocItemLabel,
    layer: ContentLayer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        doc = _parse(_reply(_element(category, text)))

    assert [(item.label, item.content_layer) for item in _blocks(doc)] == [
        (label, layer)
    ]
    warnings = [record.getMessage() for record in caplog.records]
    if category == "list":
        assert len(warnings) == 1
        assert "unknown" in warnings[0]
        assert "'list'" in warnings[0]
    else:
        assert warnings == []


_ELEMENTS = [
    _element("title", "Results", [100, 50, 900, 80]),
    _element("text", "The first paragraph.", [100, 90, 900, 200]),
    _element("formula", "$$\\frac{a}{b}$$", [300, 210, 700, 260]),
]
_BARE = json.dumps(_ELEMENTS)
# With thinking on, the chat template ends the prompt with `<think>`, so the reply
# starts with the thinking, which can hold a draft of the answer.
_DRAFT = _reply(_element("text", "A draft of the answer."))
_THINKING = f"The page has a title.\n{_DRAFT}\nOK.\n</think>\n\n"


@pytest.mark.parametrize(
    ("content", "count"),
    [
        (_BARE, 3),
        (f"```json\n{_BARE}", 3),
        (f"{_BARE}<|im_end|>", 3),
        (f"```json\n{_BARE}\n```<|im_end|>", 3),
        (f"Here is the layout.\n```json\n{_BARE}\n```", 3),
        (f"</think>\n\n```json\n{_BARE}\n```", 3),
        (f"{_THINKING}```json\n{_BARE}\n```", 3),
        (f"<think>\n{_DRAFT}\n</think>\n\n```json\n{_BARE}\n```", 3),
        (json.dumps({"elements": _ELEMENTS}), 3),
        (json.dumps(_ELEMENTS[0]), 1),
    ],
    ids=[
        "bare_list",
        "no_closing_fence",
        "end_token",
        "end_token_after_fence",
        "prose_before_fence",
        "think_end_tag",
        "thinking_with_a_draft",
        "think_block_with_a_draft",
        "wrapper_object",
        "single_element_object",
    ],
)
def test_reply_shapes_give_the_same_items(content: str, count: int) -> None:
    expected = _summary(_parse(_reply(*_ELEMENTS)))
    assert [label for label, _, _ in expected] == [
        DocItemLabel.SECTION_HEADER,
        DocItemLabel.TEXT,
        DocItemLabel.FORMULA,
    ]

    assert _summary(_parse(content)) == expected[:count]


_FULL = _reply(*_ELEMENTS)
# A looping reply can nest lists past the recursion limit of the JSON decoder.
_DEEP = "[" * 1_000_000


@pytest.mark.parametrize(
    "content",
    [
        # A cut at the last `}` would split this LaTeX and lose the whole page.
        _FULL[: _FULL.index("{a}{") + len("{a}{")],
        _FULL[: _FULL.index('{"bbox": [300')],
        _FULL[: _FULL.index("[300, 210") + len("[300,")],
        _FULL[: _FULL.index('{"bbox": [300')] + _DEEP,
    ],
    ids=["inside_latex", "after_a_comma", "inside_a_box", "deep_nesting"],
)
def test_cut_reply_keeps_complete_elements(
    content: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        doc = _parse(content)

    assert [item.text for item in doc.texts] == ["Results", "The first paragraph."]
    assert len(caplog.records) == 1
    assert "keeping 2 complete elements" in caplog.text


@pytest.mark.parametrize(
    "content",
    [
        "I cannot read this page.",
        '"x"',
        "[1, 2]",
        '{"a": 1}',
        '{"a": [1], "b": [2]}',
        '```json\n[{"bbox": [100, 200, 9',
        _DEEP,
    ],
    ids=[
        "prose",
        "json_string",
        "list_without_objects",
        "object_without_a_list",
        "object_with_two_lists",
        "cut_in_the_first_element",
        "deep_nesting",
    ],
)
def test_reply_without_elements_raises(content: str) -> None:
    with pytest.raises(ValueError):
        _parse(content)


@pytest.mark.parametrize("content", ["", " \n", "```json\n[]\n```", "</think>\n\n"])
def test_blank_reply_is_an_empty_page(content: str) -> None:
    doc = _parse(content)

    assert _blocks(doc) == []
    assert doc.pages[3].size == _SIZE


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ([100, 200, 900, 400], (61.2, 158.4, 550.8, 316.8)),
        # The grid is 0-1000, not 0-999: the far corner is the page corner.
        ([0, 0, 1000, 1000], (0.0, 0.0, 612.0, 792.0)),
        ([900, 400, 100, 200], (61.2, 158.4, 550.8, 316.8)),
        ([-5, 10, 1200, 30], (0.0, 7.92, 612.0, 23.76)),
    ],
    ids=["inside", "full_page", "swapped_corners", "clamped"],
)
def test_boxes_use_the_page_relative_0_1000_grid(
    bbox: list[int], expected: tuple[float, float, float, float]
) -> None:
    doc = _parse(_reply(_element("text", "Body", bbox)))

    prov = doc.texts[0].prov[0]
    assert _box(doc.texts[0]) == pytest.approx(expected)
    assert prov.bbox.coord_origin == CoordOrigin.TOPLEFT
    assert prov.page_no == 3


def test_elements_without_a_usable_box_keep_their_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # json reads NaN, so the parser must reject it itself.
    boxes = [
        "[1, 2, 3]",
        "[1, 2, 3, true]",
        '["1", "2", "3", "4"]',
        "[NaN, 2, 3, 4]",
        "[5, 5, 5, 9]",
        "null",
    ]
    elements = [
        f'{{"bbox": {box}, "category": "text", "text": "T{index}"}}'
        for index, box in enumerate(boxes)
    ]
    with caplog.at_level(logging.WARNING):
        doc = _parse(f"[{', '.join(elements)}]")

    assert [(item.text, item.prov) for item in doc.texts] == [
        (f"T{index}", []) for index in range(len(boxes))
    ]
    assert [record.getMessage() for record in caplog.records] == [
        "Keeping 6 Typhoon OCR V2 elements without a valid box"
    ]


def test_captions_link_to_one_adjacent_owner() -> None:
    doc = _parse(
        _reply(
            _element("figure", ""),
            _element("figure_caption", "Figure 1. Sales."),
            _element("table_caption", "Table 1. Costs."),
            _element("table", "<table><tr><td>1</td></tr></table>"),
            _element("formula", "$$x$$"),
            _element("formula_caption", "(1)"),
            _element("figure", ""),
            # An element without content still separates its neighbors.
            _element("text", ""),
            _element("figure_caption", "Figure 2. Not linked."),
        )
    )

    first, second = doc.pictures
    assert [ref.resolve(doc).text for ref in first.captions] == ["Figure 1. Sales."]
    assert [ref.resolve(doc).text for ref in doc.tables[0].captions] == [
        "Table 1. Costs."
    ]
    assert second.captions == []
    assert [item.label for item in _blocks(doc)] == [
        DocItemLabel.PICTURE,
        DocItemLabel.CAPTION,
        DocItemLabel.CAPTION,
        DocItemLabel.TABLE,
        DocItemLabel.FORMULA,
        DocItemLabel.CAPTION,
        DocItemLabel.PICTURE,
        DocItemLabel.CAPTION,
    ]


def test_title_is_always_a_section_header() -> None:
    reply = _reply(_element("title", "Intro"), _element("title", "## Method"))

    raw = _parse(reply, normalize_heading_levels=False)
    normalized = _parse(reply)

    assert [type(item) for item in raw.texts] == [SectionHeaderItem] * 2
    headings = [item for item in raw.texts if isinstance(item, SectionHeaderItem)]
    assert [(item.text, item.orig, item.level) for item in headings] == [
        ("Intro", "Intro", 2),
        ("Method", "## Method", 2),
    ]
    assert [
        item.level for item in normalized.texts if isinstance(item, SectionHeaderItem)
    ] == [1, 1]


@pytest.mark.parametrize("latex", ["$$a^2 + 8 = 12$$", "\\[a^2 + 8 = 12\\]"])
def test_formula_exports_its_latex_once(latex: str) -> None:
    doc = _parse(_reply(_element("formula", latex)))

    (formula,) = doc.texts
    assert (formula.label, formula.text, formula.orig) == (
        DocItemLabel.FORMULA,
        "a^2 + 8 = 12",
        latex,
    )
    assert doc.export_to_markdown() == "$$a^2 + 8 = 12$$"


def test_page_furniture_stays_out_of_markdown() -> None:
    doc = _parse(
        _reply(
            _element("header", "Annual **Report**"),
            _element("text", "Body"),
            _element("page_footnote", "1 See the appendix."),
            _element("footer", "Report\\(^{x}\\) 2025"),
        )
    )

    # Rich text gives a parent item, an inline group and runs; all are furniture.
    body = [item.text for item in doc.texts if item.content_layer == ContentLayer.BODY]
    assert body == ["Body", "1 See the appendix."]
    assert doc.groups
    assert all(group.content_layer == ContentLayer.FURNITURE for group in doc.groups)
    assert doc.export_to_markdown() == "Body\n\n1 See the appendix."
    assert doc.export_to_markdown(included_content_layers=set(ContentLayer)) == (
        "Annual **Report**\n\nBody\n\n1 See the appendix.\n\nReport$^{x}$ 2025"
    )


def test_table_fallbacks_keep_content(caplog: pytest.LogCaptureFixture) -> None:
    # 60 x 20 cells exceed the 1000-cell limit of the rich HTML table mapper.
    rows = "".join(
        "<tr>" + "".join(f"<td>{row}.{col}</td>" for col in range(20)) + "</tr>"
        for row in range(60)
    )
    pipe_table = "| a | b |\n|---|---|\n| 1 | 2 |"
    with caplog.at_level(logging.WARNING):
        doc = _parse(
            _reply(
                _element("table", f"<table>{rows}</table>", [100, 200, 900, 400]),
                _element("table", pipe_table),
                _element("table", "<table></table>"),
            )
        )

    (table,) = doc.tables
    assert (table.data.num_rows, table.data.num_cols) == (60, 20)
    assert len(table.data.table_cells) == 1200
    assert _box(table) == pytest.approx((61.2, 158.4, 550.8, 316.8))
    assert [(item.label, item.text) for item in doc.texts] == [
        (DocItemLabel.TEXT, pipe_table)
    ]
    assert _box(doc.texts[0]) == pytest.approx((61.2, 79.2, 550.8, 110.88))
    assert "without spans" in caplog.text
    assert "without cells as text" in caplog.text


def test_huge_table_span_keeps_the_text(caplog: pytest.LogCaptureFixture) -> None:
    # Without a grid limit, the fallback reader builds 2 x 100000 grid cells.
    html = '<table><tr><td colspan="100000">a</td></tr><tr><td>b</td></tr></table>'
    with caplog.at_level(logging.WARNING):
        doc = _parse(_reply(_element("table", html)))

    assert doc.tables == []
    assert [(item.label, item.text) for item in doc.texts] == [
        (DocItemLabel.TEXT, "a\nb")
    ]
    assert "Ignoring an HTML table of 2 x 100000 grid cells" in caplog.text


def test_figure_keeps_box_and_unexpected_text() -> None:
    doc = _parse(
        _reply(
            _element("figure", "", [100, 200, 900, 400]),
            _element("figure", "A bar chart", [100, 500, 900, 700]),
        )
    )

    assert [type(item) for item in _blocks(doc)] == [
        PictureItem,
        PictureItem,
        TextItem,
    ]
    first, second = doc.pictures
    assert _box(first) == pytest.approx((61.2, 158.4, 550.8, 316.8))
    assert [item.text for item in doc.texts] == ["A bar chart"]
    assert _box(doc.texts[0]) == _box(second)


def test_real_table_page_keeps_order_layers_boxes_and_spans() -> None:
    doc = _parse(_real_reply("2305.03393v1-pg9"))

    blocks = _blocks(doc)
    assert [(item.label, item.content_layer) for item in blocks] == [
        (DocItemLabel.PAGE_HEADER, ContentLayer.FURNITURE),
        (DocItemLabel.TEXT, ContentLayer.BODY),
        (DocItemLabel.SECTION_HEADER, ContentLayer.BODY),
        (DocItemLabel.TEXT, ContentLayer.BODY),
        # The model labels the table caption as text, so it stays unlinked.
        (DocItemLabel.TEXT, ContentLayer.BODY),
        (DocItemLabel.TABLE, ContentLayer.BODY),
        (DocItemLabel.SECTION_HEADER, ContentLayer.BODY),
        (DocItemLabel.TEXT, ContentLayer.BODY),
        (DocItemLabel.TEXT, ContentLayer.BODY),
    ]
    header, table = blocks[0], blocks[5]
    assert isinstance(header, DocItem)
    # The reply box [312, 114, 786, 128] on the 612 x 792 pt page.
    assert _box(header) == pytest.approx((190.944, 90.288, 481.032, 101.376))
    assert header.prov[0].bbox.coord_origin == CoordOrigin.TOPLEFT
    assert header.prov[0].page_no == 3
    assert [
        (item.text, item.level)
        for item in blocks
        if isinstance(item, SectionHeaderItem)
    ] == [("5.1 Hyper Parameter Optimization", 1), ("5.2 Quantitative Results", 1)]

    assert isinstance(table, TableItem)
    grid = table.data.grid
    assert (table.data.num_rows, table.data.num_cols) == (10, 8)
    assert (grid[0][0].text, grid[0][0].row_span, grid[0][0].column_header) == (
        "# enc-layers",
        2,
        True,
    )
    assert (grid[0][3].text, grid[0][3].col_span) == ("TEDs", 3)
    assert [cell.text for cell in grid[2]] == [
        "6",
        "6",
        "OTSL",
        "0.965",
        "0.934",
        "0.955",
        "0.88",
        "2.73",
    ]
    # The reply writes <b>0.88</b>: the cell holds one bold run.
    bold = grid[2][6]
    assert isinstance(bold, RichTableCell)
    runs = [ref.resolve(doc) for ref in bold.ref.resolve(doc).children]
    assert [
        (run.text, run.formatting is not None and run.formatting.bold)
        for run in runs
        if isinstance(run, TextItem)
    ] == [("0.88", True)]

    markdown = doc.export_to_markdown()
    with_furniture = doc.export_to_markdown(included_content_layers=set(ContentLayer))
    assert "## 5.1 Hyper Parameter Optimization\n\n" in markdown
    assert "| **0.88** " in markdown
    assert "Optimized Table Tokenization" not in markdown
    header_line = "Optimized Table Tokenization for Table Structure Recognition 9"
    assert with_furniture == f"{header_line}\n\n{markdown}"


def test_real_formula_page_exports_latex_once() -> None:
    doc = _parse(
        _real_reply("code_and_formula-p2"), size=Size(width=595.276, height=841.89)
    )

    blocks = _blocks(doc)
    assert [(item.label, item.content_layer) for item in blocks] == [
        (DocItemLabel.SECTION_HEADER, ContentLayer.BODY),
        (DocItemLabel.TEXT, ContentLayer.BODY),
        (DocItemLabel.TEXT, ContentLayer.BODY),
        (DocItemLabel.FORMULA, ContentLayer.BODY),
        (DocItemLabel.TEXT, ContentLayer.BODY),
        (DocItemLabel.TEXT, ContentLayer.BODY),
        (DocItemLabel.TEXT, ContentLayer.BODY),
        (DocItemLabel.PAGE_FOOTER, ContentLayer.FURNITURE),
    ]
    formula, footer = blocks[3], blocks[-1]
    assert isinstance(formula, TextItem)
    assert (formula.text, formula.orig) == ("a^2 + 8 = 12", "$$a^2 + 8 = 12$$")
    assert isinstance(footer, TextItem)
    assert footer.text == "1"

    markdown = doc.export_to_markdown()
    with_furniture = doc.export_to_markdown(included_content_layers=set(ContentLayer))
    assert markdown.startswith("## Formula\n\n")
    assert markdown.count("$$a^2 + 8 = 12$$") == 1
    assert "$$$$" not in markdown
    assert with_furniture == f"{markdown}\n\n1"


def test_real_thai_page_keeps_text_cells_and_furniture() -> None:
    # This reply is a bare JSON list, without a code fence.
    doc = _parse(_real_reply("thai_rainwater"), size=Size(width=594.96, height=841.92))

    assert [(item.label, item.content_layer) for item in _blocks(doc)] == [
        (DocItemLabel.PAGE_HEADER, ContentLayer.FURNITURE),
        (DocItemLabel.SECTION_HEADER, ContentLayer.BODY),
        (DocItemLabel.TEXT, ContentLayer.BODY),
        (DocItemLabel.TEXT, ContentLayer.BODY),
        (DocItemLabel.TEXT, ContentLayer.BODY),
        (DocItemLabel.TABLE, ContentLayer.BODY),
        (DocItemLabel.PAGE_FOOTER, ContentLayer.FURNITURE),
    ]
    # NFKC splits SARA AM, so the text would show any normalization.
    assert unicodedata.normalize("NFKC", _THAI_TEXTS[1]) != _THAI_TEXTS[1]
    assert [item.text for item in doc.texts] == _THAI_TEXTS
    (table,) = doc.tables
    assert [[cell.text for cell in row] for row in table.data.grid] == [
        # The page prints เต็ม ("full"); the model reads เติม. The cell keeps the reply.
        ["เดือน", "ปริมาณน้ำฝน (มม.)", "จำนวนถังที่เติม"],
        ["มิถุนายน", "๑๖๘", "7"],
        ["กรกฎาคม", "๒๑๒", "10"],
        ["สิงหาคม", "๒๔๕", "12"],
    ]

    markdown = doc.export_to_markdown()
    with_furniture = doc.export_to_markdown(included_content_layers=set(ContentLayer))
    assert markdown.startswith(f"## {_THAI_TEXTS[1]}\n\n{_THAI_TEXTS[2]}\n\n")
    assert with_furniture == f"{_THAI_TEXTS[0]}\n\n{markdown}\n\n{_THAI_TEXTS[-1]}"


@pytest.mark.parametrize(
    ("preset_id", "repo_id"),
    [
        ("typhoon_ocr2_2b", "typhoon-ai/typhoon-ocr2-2b"),
        ("typhoon_ocr2_08b", "typhoon-ai/typhoon-ocr2-0.8b"),
    ],
)
def test_presets_pin_weights_and_engine_floors(preset_id: str, repo_id: str) -> None:
    preset = VlmConvertOptions.get_preset(preset_id)
    spec = preset.model_spec
    transformers = spec.get_engine_config(VlmEngineType.TRANSFORMERS)
    vllm = spec.get_engine_config(VlmEngineType.VLLM)
    mlx = spec.get_engine_config(VlmEngineType.MLX)
    engines = (transformers, vllm, mlx)

    # The models were trained with this prompt, so it must stay byte for byte.
    assert hashlib.sha256(spec.prompt.encode()).hexdigest() == _PROMPT_SHA256
    assert spec.response_format == ResponseFormat.TYPHOON_OCR2_JSON
    assert [config.repo_id for config in engines] == [
        repo_id,
        repo_id,
        f"{repo_id}-MLX-8bit",
    ]
    assert all(
        re.fullmatch(r"[0-9a-f]{40}", config.revision or "") for config in engines
    )
    assert vllm.revision == transformers.revision
    assert mlx.revision != transformers.revision
    assert [config.min_engine_version for config in engines] == [
        "5.15.1",
        "0.33.0",
        "0.7.6",
    ]
    assert transformers.torch_dtype == "bfloat16"
    assert (
        transformers.extra_config["transformers_model_type"]
        == TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT
    )
    assert spec.get_runtime_input_extra_config(VlmEngineType.TRANSFORMERS) == {
        "transformers_prompt_style": TransformersPromptStyle.CHAT,
        "skip_special_tokens": True,
    }
    assert vllm.extra_config == {"max_model_len": 40960}
    assert (spec.temperature, spec.max_new_tokens) == (0.0, 16384)
    assert spec.chat_template_kwargs == {"enable_thinking": False}
    assert not spec.trust_remote_code
    # Ollama ignores chat_template_kwargs, so the thinking block would stay open.
    assert not spec.is_engine_supported(VlmEngineType.API_OLLAMA)
    assert preset.scale * 72 == pytest.approx(2048)
    assert preset.max_size == 2048


def test_changed_api_params_leave_the_preset_unchanged() -> None:
    # The docs ask for chat_template_kwargs in your own params. A deep copy keeps
    # the parts that the preset shares between engines shared.
    spec = VlmConvertOptions.get_preset("typhoon_ocr2_2b").model_spec
    spec = spec.model_copy(deep=True)

    params = spec.get_api_params(VlmEngineType.API_OPENAI)
    params["chat_template_kwargs"]["enable_thinking"] = True

    assert [
        spec.get_api_params(engine_type)["chat_template_kwargs"]
        for engine_type in (VlmEngineType.API, VlmEngineType.API_OPENAI)
    ] == [{"enable_thinking": False}] * 2


# The fake server answers with the real reply for the Thai page.
_PAGE_REPLY = _real_reply("thai_rainwater")
_PAGE_LAYOUT = [
    (DocItemLabel.PAGE_HEADER, ContentLayer.FURNITURE),
    (DocItemLabel.SECTION_HEADER, ContentLayer.BODY),
    (DocItemLabel.TEXT, ContentLayer.BODY),
    (DocItemLabel.TEXT, ContentLayer.BODY),
    (DocItemLabel.TEXT, ContentLayer.BODY),
    (DocItemLabel.TABLE, ContentLayer.BODY),
    (DocItemLabel.PAGE_FOOTER, ContentLayer.FURNITURE),
]


@pytest.fixture
def api() -> Iterator[FakeOpenAiApi]:
    service = FakeService()
    fake = FakeOpenAiApi(completion=_PAGE_REPLY)
    service.include(fake.router)
    service.start()
    fake.service = service
    try:
        yield fake
    finally:
        service.stop()


def _convert(
    api: FakeOpenAiApi,
    source: Path,
    engine_type: VlmEngineType = VlmEngineType.API_OPENAI,
) -> ConversionResult:
    options = VlmConvertOptions.from_preset(
        "typhoon_ocr2_2b",
        engine_options=ApiVlmEngineOptions(
            engine_type=engine_type,
            url=AnyUrl(f"{api.service.base_url}/v1/chat/completions"),
        ),
    )
    pipeline_options = VlmPipelineOptions(
        vlm_options=options, enable_remote_services=True
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline, pipeline_options=pipeline_options
            ),
            InputFormat.IMAGE: ImageFormatOption(
                pipeline_cls=VlmPipeline, pipeline_options=pipeline_options
            ),
        }
    )
    return converter.convert(source)


@pytest.mark.parametrize(
    ("engine_type", "source", "image_size"),
    [
        # The Thai page is 594.96 x 841.92 pt.
        (VlmEngineType.API, "pdf", (1448, 2048)),
        (VlmEngineType.API_OPENAI, "pdf", (1448, 2048)),
        (VlmEngineType.API_OPENAI, "png", (2048, 1536)),
    ],
)
def test_api_request_matches_the_model_card(
    api: FakeOpenAiApi,
    tmp_path: Path,
    engine_type: VlmEngineType,
    source: str,
    image_size: tuple[int, int],
) -> None:
    path = _THAI_PDF
    if source == "png":
        path = tmp_path / "page.png"
        Image.new("RGB", (800, 600), "white").save(path)

    result = _convert(api, path, engine_type)

    (request,) = api.service.requests_for("POST", r"/v1/chat/completions")
    body = request.json()
    (message,) = body.pop("messages")
    # The keys and values that the client of the model card sends.
    assert body == {
        "model": "typhoon-ocr2",
        "max_tokens": 16384,
        "temperature": 0.0,
        "top_p": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert message["role"] == "user"
    image_part, text_part = message["content"]
    assert text_part == {"type": "text", "text": TYPHOON_OCR2_PARSE_PROMPT}
    prefix = "data:image/png;base64,"
    assert image_part["image_url"]["url"].startswith(prefix)
    image_bytes = base64.b64decode(image_part["image_url"]["url"].removeprefix(prefix))
    # A page renders with a 2048-pixel long side. A PDF page can be 1 pixel off.
    assert Image.open(io.BytesIO(image_bytes)).size == pytest.approx(image_size, abs=1)
    assert result.status == ConversionStatus.SUCCESS
    assert [
        (item.label, item.content_layer) for item in _blocks(result.document)
    ] == _PAGE_LAYOUT
    # The Thai text survives the JSON response unchanged.
    assert [item.text for item in result.document.texts] == _THAI_TEXTS


def test_cut_reply_gives_partial_success(api: FakeOpenAiApi) -> None:
    api.completion = _PAGE_REPLY[: _PAGE_REPLY.index("<table>") + len("<table><tr>")]
    api.finish_reason = "length"

    result = _convert(api, _THAI_PDF)

    assert result.status == ConversionStatus.PARTIAL_SUCCESS
    assert [(error.error_message, error.page_no) for error in result.errors] == [
        ("VLM output incomplete (stop_reason=length).", 1)
    ]
    # The five elements before the cut table stay.
    assert [
        (item.label, item.content_layer) for item in _blocks(result.document)
    ] == _PAGE_LAYOUT[:5]


def test_invalid_reply_is_a_page_error(api: FakeOpenAiApi) -> None:
    api.completion = "I cannot read this page."

    result = _convert(api, _THAI_PDF)

    assert result.status == ConversionStatus.PARTIAL_SUCCESS
    (error,) = result.errors
    assert error.page_no == 1
    assert error.error_message.startswith("Invalid Typhoon OCR V2 response: ")
    assert list(result.document.iterate_items()) == []
    assert list(result.document.pages) == [1]


@pytest.mark.skipif(
    os.getenv("DOCLING_TEST_TYPHOON_OCR2") != "1",
    reason="set DOCLING_TEST_TYPHOON_OCR2=1 to run the Typhoon OCR V2 inference test",
)
def test_e2e_typhoon_ocr2() -> None:
    """Convert a page with the 0.8B preset on the local engine that auto-inline picks."""
    options = VlmConvertOptions.from_preset(
        "typhoon_ocr2_08b", engine_options=AutoInlineVlmEngineOptions()
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=VlmPipelineOptions(vlm_options=options),
            )
        }
    )

    result = converter.convert(_PDF)

    # SUCCESS excludes PARTIAL_SUCCESS, which a cut or invalid reply gives.
    assert result.status == ConversionStatus.SUCCESS
    doc = result.document
    assert [table.data.num_cols for table in doc.tables] == [8]
    assert "5.1 Hyper Parameter Optimization" in [
        item.text for item in doc.texts if isinstance(item, SectionHeaderItem)
    ]
    assert any(
        item.label == DocItemLabel.PAGE_HEADER
        and item.content_layer == ContentLayer.FURNITURE
        for item in doc.texts
    )
