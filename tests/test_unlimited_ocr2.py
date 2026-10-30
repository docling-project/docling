# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Unit tests for Unlimited-OCR 2 output parsing.

These operate on plain strings: no model and no network. `REAL_SAMPLE` is the only
real model output, as reported in https://github.com/baidu/Unlimited-OCR/issues/114.
Every other string is hand-written from the grammar in the vendor documentation
and its post-processing script. They are not model output.
"""

import logging

import pytest
from docling_core.types.doc import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    SectionHeaderItem,
    Size,
    TitleItem,
)

from docling.datamodel.pipeline_options import VlmConvertOptions
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.models.inference_engines.vlm.base import VlmEngineType
from docling.utils.unlimited_ocr2_utils import parse_unlimited_ocr2

PAGE_SIZE = Size(width=600, height=800)
REAL_SAMPLE = (
    "<PAGE><BLOCK>aside_text[23, 268, 60, 699]<FORK>"
    "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023\n"
    "<BLOCK>title[343, 186, 655, 209]<FORK>Attention Is All You Need"
)
TABLE = "<table><tr><td>Year</td><td>Total</td></tr><tr><td>2025</td><td>12</td></tr>"


def _parse(content: str) -> DoclingDocument:
    return parse_unlimited_ocr2(content, PAGE_SIZE, page_no=3, filename="doc.pdf")


def test_real_sample_keeps_labels_and_page_relative_boxes() -> None:
    doc = _parse(REAL_SAMPLE)

    stamp, title = doc.texts
    assert stamp.label == DocItemLabel.TEXT
    assert stamp.text == "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023"
    assert isinstance(title, TitleItem)
    assert title.text == "Attention Is All You Need"
    box = title.prov[0].bbox
    assert (box.l, box.t, box.r, box.b) == pytest.approx((205.8, 148.8, 393.0, 167.2))
    assert title.prov[0].page_no == 3


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("text", DocItemLabel.TEXT),
        ("aside_text", DocItemLabel.TEXT),
        ("header", DocItemLabel.PAGE_HEADER),
        ("footer", DocItemLabel.PAGE_FOOTER),
        ("page_number", DocItemLabel.PAGE_FOOTER),
        ("image_caption", DocItemLabel.CAPTION),
        ("table_caption", DocItemLabel.CAPTION),
        ("image_footnote", DocItemLabel.FOOTNOTE),
        ("page_footnote", DocItemLabel.FOOTNOTE),
        ("ref_text", DocItemLabel.REFERENCE),
        ("algorithm", DocItemLabel.CODE),
        ("not_documented", DocItemLabel.TEXT),
    ],
)
def test_text_labels(label: str, expected: DocItemLabel) -> None:
    doc = _parse(f"<PAGE><BLOCK>{label}[10, 20, 30, 40]<FORK>Content")

    assert [(item.label, item.text) for item in doc.texts] == [(expected, "Content")]


def test_page_furniture_is_left_out_of_the_body() -> None:
    doc = _parse(
        "<PAGE><BLOCK>header[10, 20, 30, 40]<FORK>Journal\n"
        "<BLOCK>text[10, 50, 30, 60]<FORK>Body\n"
        "<BLOCK>page_number[10, 70, 30, 80]<FORK>12"
    )

    assert [item.content_layer for item in doc.texts] == [
        ContentLayer.FURNITURE,
        ContentLayer.BODY,
        ContentLayer.FURNITURE,
    ]
    assert doc.export_to_markdown() == "Body"


def test_title_level_decides_section_header() -> None:
    doc = _parse(
        "<PAGE><BLOCK>title<LEVEL3>[10, 20, 30, 40]<FORK>Method\n"
        "<BLOCK>title<LEVEL0>[10, 50, 30, 60]<FORK>Results"
    )

    assert all(isinstance(item, SectionHeaderItem) for item in doc.texts)
    assert [item.level for item in doc.texts] == [3, 1]


def test_equation_removes_math_delimiters() -> None:
    doc = _parse("<PAGE><BLOCK>equation[10, 20, 30, 40]<FORK>$$E = mc^2$$")

    assert doc.texts[0].label == DocItemLabel.FORMULA
    assert doc.texts[0].text == "E = mc^2"


@pytest.mark.parametrize("label", ["image", "figure", "chart", "seal"])
def test_picture_blocks_keep_text_and_remove_noise(label: str) -> None:
    doc = _parse(
        f"<PAGE><BLOCK>{label}[10, 20, 30, 40]<FORK>[non-text]\n"
        f"<BLOCK>{label}[10, 50, 30, 60]<FORK>Text in the picture"
    )

    assert len(doc.pictures) == 2
    assert [(item.label, item.text) for item in doc.texts] == [
        (DocItemLabel.TEXT, "Text in the picture")
    ]


def test_captions_link_to_adjacent_owner() -> None:
    doc = _parse(
        "<PAGE><BLOCK>image[10, 20, 30, 40]<FORK>\n"
        "<BLOCK>image_caption[10, 41, 30, 45]<FORK>Figure 1\n"
        f"<BLOCK>table[10, 50, 30, 60]<FORK>{TABLE}</table>\n"
        "<BLOCK>table_caption[10, 61, 30, 65]<FORK>Table 1"
    )

    assert doc.pictures[0].caption_text(doc) == "Figure 1"
    assert doc.tables[0].caption_text(doc) == "Table 1"


@pytest.mark.parametrize(
    ("second", "joined"),
    [
        ("second part", "first part second part"),
        ("第二部分", "first part第二部分"),
    ],
)
def test_merge_joins_text_and_keeps_both_boxes(second: str, joined: str) -> None:
    doc = _parse(
        "<PAGE><BLOCK>text[10, 20, 30, 40]<FORK>first part\n<MERGE>\n"
        f"<BLOCK>text[60, 20, 90, 40]<FORK>{second}"
    )

    assert [item.text for item in doc.texts] == [joined]
    prov = doc.texts[0].prov
    assert [item.bbox.l for item in prov] == pytest.approx([6.0, 36.0])
    assert joined[prov[1].charspan[0] : prov[1].charspan[1]] == second


def test_merge_does_not_cross_a_non_text_block() -> None:
    doc = _parse(
        "<PAGE><BLOCK>text[10, 20, 30, 40]<FORK>Body\n<MERGE>\n"
        "<BLOCK>equation[10, 50, 30, 60]<FORK>x\n<MERGE>\n"
        "<BLOCK>text[10, 70, 30, 80]<FORK>More"
    )

    assert [item.text for item in doc.texts] == ["Body", "x", "More"]


def test_cross_and_append_prefixes_map_to_the_base_label() -> None:
    doc = _parse(
        "<PAGE><BLOCK>append_text[10, 20, 30, 40]<FORK>End of a paragraph\n"
        f"<BLOCK>cross_table[10, 50, 30, 60]<FORK>{TABLE}</table>"
    )

    assert doc.texts[0].text == "End of a paragraph"
    assert doc.tables[0].data.num_rows == 2


@pytest.mark.parametrize(
    "table",
    [
        f"{TABLE}</table>",
        f"{TABLE}<tr><td>2026</td><td>",
        f"{TABLE}<tr><td colsp",
        TABLE.replace("<td>Total", '<td rowspan="" colspan="x">Total') + "</table>",
    ],
    ids=["complete", "cut_in_a_cell", "cut_in_a_tag", "bad_spans"],
)
def test_table_shapes_keep_their_cells(table: str) -> None:
    doc = _parse(f"<PAGE><BLOCK>table[10, 20, 30, 40]<FORK>{table}")

    cells = [cell.text for cell in doc.tables[0].data.table_cells]
    assert cells[:4] == ["Year", "Total", "2025", "12"]


def test_table_without_html_stays_text() -> None:
    doc = _parse("<PAGE><BLOCK>table[10, 20, 30, 40]<FORK>Year Total")

    assert not doc.tables
    assert doc.texts[0].text == "Year Total"


@pytest.mark.parametrize(
    "content",
    [
        "<PAGE><BLOCK><BLOCK>title<LEVEL2>[10, 20, 30, 40]<FORK>Method",
        "<PAGE><BLOCK>title[10, 20, 30, 40]<LEVEL2><FORK>Method",
        "<PAGE><BLOCK>title</LEVEL2}[10, 20, 30, 40]<FORK>Method",
        "<PAGE><BLOCK>title<LEVEL2>[10\uff0c20\uff0c30\uff0c40]<FORK>Method",
        "<PAGE><BLOCK>title<LEVEL2>[10, 20, 30, 40, 50, 60]<FORK>Method",
        "<PAGE><BLOCK>title<LEVEL2>[10, 20, 30, 40]Method",
        "<PAGE>title<LEVEL2>[10, 20, 30, 40]<FORK>Method",
        "stray text<PAGE><BLOCK>title<LEVEL2>[10, 20, 30, 40]<FORK>Method",
        "<BLOCK>title<LEVEL2>[10, 20, 30, 40]<FORK>Method",
        "<PAGE><BLOCK>title<LEVEL2>[10, 20, 30, 40]<FORK>Method<\uff5cend▁of▁sentence\uff5c>",
    ],
    ids=[
        "doubled_block",
        "level_after_box",
        "broken_level",
        "fullwidth_commas",
        "extra_numbers",
        "missing_fork",
        "missing_block",
        "text_before_page",
        "missing_page",
        "stop_token",
    ],
)
def test_malformed_headers_give_the_same_item(content: str) -> None:
    doc = _parse(content)

    assert [item.text for item in doc.texts] == ["Method"]
    assert isinstance(doc.texts[0], SectionHeaderItem)
    box = doc.texts[0].prov[0].bbox
    assert (box.l, box.t, box.r, box.b) == pytest.approx((6.0, 16.0, 18.0, 32.0))


def test_box_is_clamped_and_ordered() -> None:
    doc = _parse("<PAGE><BLOCK>text[1200, 40, -5, 20]<FORK>Content")

    box = doc.texts[0].prov[0].bbox
    assert (box.l, box.t, box.r, box.b) == pytest.approx((0.0, 16.0, 600.0, 32.0))


@pytest.mark.parametrize(
    "header", ["<BLOCK>text<FORK>", "<BLOCK>text[10, 20, 10, 40]<FORK>"]
)
def test_block_without_a_usable_box_keeps_text_and_warns(
    header: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        doc = _parse(f"<PAGE>{header}Content")

    assert doc.texts[0].text == "Content"
    assert not doc.texts[0].prov
    assert "no valid box" in caplog.text


def test_output_without_blocks_is_one_text_item() -> None:
    doc = _parse("Plain text from a server with the wrong prompt")

    assert [item.text for item in doc.texts] == [
        "Plain text from a server with the wrong prompt"
    ]


@pytest.mark.parametrize("content", ["", "<PAGE>", "<PAGE>\n"])
def test_blank_page_is_an_empty_document(content: str) -> None:
    doc = _parse(content)

    assert not list(doc.iterate_items())
    assert doc.pages[3].size == PAGE_SIZE


def test_only_the_first_page_is_kept(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        doc = _parse(
            f"<PAGE><BLOCK>cross_table[10, 20, 30, 40]<FORK>{TABLE}</table>"
            f"<PAGE><BLOCK>append_table[10, 20, 30, 40]<FORK>{TABLE}</table>"
        )

    assert len(doc.tables) == 1
    assert "more than one page" in caplog.text


def test_repeated_blocks_cut_by_max_tokens_keep_a_valid_document() -> None:
    repeated = "<BLOCK>text[10, 20, 30, 40]<FORK>Loop\n" * 3

    doc = _parse(f"<PAGE>{repeated}<BLOCK>text[10, 20, 30, 40, 10, 20")

    assert [item.text for item in doc.texts] == ["Loop"] * 3


@pytest.mark.parametrize("engine_type", [VlmEngineType.API, VlmEngineType.API_OPENAI])
def test_preset_keeps_table_tokens_on_both_api_engine_types(
    engine_type: VlmEngineType,
) -> None:
    options = VlmConvertOptions.from_preset(
        "unlimited_ocr2",
        engine_options=ApiVlmEngineOptions(
            engine_type=engine_type, url="http://localhost:10000/v1/chat/completions"
        ),
    )

    params = options.model_spec.get_api_params(engine_type)
    assert params["skip_special_tokens"] is False
    assert params["model"] == "Unlimited-OCR-2"
