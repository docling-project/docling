# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Test Surya 2.5 compact HTML parsing in the VLM pipeline."""

import pytest
from docling_core.types.doc import ContentLayer, DocItemLabel, Size, TableItem

from docling.utils.chandra_utils import parse_chandra_html
from docling.utils.surya_utils import parse_surya_html

_SIZE = Size(width=612, height=792)
# Raw datalab-to/surya-ocr-2.5 output for page 9 of arXiv 2305.03393, shortened:
# the table keeps its two header rows and first two body rows, and the last
# paragraph is cut. It ends with the end token, as Transformers decodes it.
_SAMPLE = (
    "<|Page-Header|><|312|><|116|><|786|><|128|>"
    "Optimized Table Tokenization for Table Structure Recognition 9"
    "<|Section-Header|><|215|><|212|><|521|><|226|>"
    "<h2>5.1 Hyper Parameter Optimization</h2>"
    "<|Caption|><|215|><|345|><|787|><|413|>"
    "<p><b>Table 1.</b> HPO performed in OTSL and HTML representation.</p>"
    "<|Table|><|225|><|424|><|770|><|592|><table>"
    '<tr><th data-rs="2" data-kind="parent"># enc-layers'
    '<th data-rs="2" data-kind="parent"># dec-layers'
    '<th data-rs="2" data-kind="parent">Language'
    '<th data-cs="3" data-kind="parent">TEDs'
    '<th data-kind="ghost"><th data-kind="ghost">'
    '<th data-rs="2" data-kind="parent">mAP (0.75)'
    '<th data-rs="2" data-kind="parent">Inference time (secs)'
    '<tr><th data-kind="ghost"><th data-kind="ghost"><th data-kind="ghost">'
    "<th>simple<th>complex<th>all"
    '<th data-kind="ghost"><th data-kind="ghost">'
    '<tr><td data-rs="2" data-kind="parent">6<td data-rs="2" data-kind="parent">6'
    "<td>OTSL<td>0.965<td>0.934<td>0.955<td><b>0.88</b><td><b>2.73</b>"
    '<tr><td data-kind="ghost"><td data-kind="ghost">'
    "<td>HTML<td>0.969<td>0.927<td>0.955<td>0.857<td>5.39</table>"
    "<|Section-Header|><|215|><|640|><|433|><|652|><h2>5.2 Quantitative Results</h2>"
    "<|Text|><|215|><|780|><|787|><|839|>"
    "<p>Finally, OTSL achieves faster inference due to fewer decoding steps which "
    "is a result of the reduced sequence representation.</p><|im_end|>"
)


def _cell(table: TableItem, text: str):
    return next(cell for cell in table.data.table_cells if cell.text == text)


def test_real_page_keeps_layout_labels_and_boxes():
    doc = parse_surya_html(_SAMPLE, _SIZE, page_no=1)

    labels = [item.label for item in doc.texts]
    assert labels.count(DocItemLabel.SECTION_HEADER) == 2
    assert DocItemLabel.CAPTION in labels
    header = next(i for i in doc.texts if i.label == DocItemLabel.PAGE_HEADER)
    assert header.content_layer == ContentLayer.FURNITURE
    assert header.text.startswith("Optimized Table Tokenization")
    # <|Page-Header|><|312|><|116|><|786|><|128|> in 0-1000 page coordinates.
    bbox = header.prov[0].bbox
    assert (bbox.l, bbox.t, bbox.r, bbox.b) == pytest.approx(
        (0.312 * 612, 0.116 * 792, 0.786 * 612, 0.128 * 792)
    )
    assert not any("<|" in item.text for item in doc.texts)
    assert doc.texts[-1].text.endswith("reduced sequence representation.")


def test_real_page_turns_ghost_cells_into_spans():
    doc = parse_surya_html(_SAMPLE, _SIZE, page_no=1)

    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert (table.data.num_rows, table.data.num_cols) == (4, 8)
    teds = _cell(table, "TEDs")
    assert (teds.col_span, teds.start_col_offset_idx) == (3, 3)
    layers = _cell(table, "# enc-layers")
    assert (layers.row_span, layers.col_span) == (2, 1)
    # The HTML row starts after the two body cells that span from the OTSL row.
    assert _cell(table, "HTML").start_col_offset_idx == 2
    # The second header row starts after the three cells spanning from row one.
    assert _cell(table, "simple").start_col_offset_idx == 3
    assert table.prov[0].bbox.t == pytest.approx(0.424 * 792)


def test_cut_output_keeps_the_finished_blocks():
    content = (
        "<|Section-Header|><|100|><|50|><|900|><|80|><h2>Results</h2>"
        "<|Text|><|100|><|90|><|900|><|200|><p>The first paragraph.</p>"
        "<|Text|><|100|><|210|>"
    )

    doc = parse_surya_html(content, _SIZE, page_no=1)

    assert [item.text for item in doc.texts] == ["Results", "The first paragraph."]


def test_cut_table_keeps_its_cells():
    content = (
        "<|Table|><|100|><|100|><|900|><|400|><table><tr><th>Name<th>Value"
        "<tr><td>alpha<td>1<tr><td>beta<td"
    )

    doc = parse_surya_html(content, _SIZE, page_no=1)

    cells = {cell.text for cell in doc.tables[0].data.table_cells}
    assert {"Name", "Value", "alpha", "1", "beta"} <= cells


def test_picture_label_and_nested_box_become_a_located_picture():
    content = (
        "<|Picture|><|100|><|100|><|500|><|400|>"
        '<|box|><|120|><|110|><|480|><|390|><img alt="A bar chart"/>'
        "<|Caption|><|100|><|410|><|500|><|440|><p>Figure 1. Sales.</p>"
    )

    doc = parse_surya_html(content, _SIZE, page_no=1)

    assert len(doc.pictures) == 1
    assert doc.pictures[0].prov
    assert any(item.label == DocItemLabel.CAPTION for item in doc.texts)
    assert not any("<|" in item.text for item in doc.texts)


def test_div_blocks_of_older_checkpoints_map_like_chandra():
    content = (
        '<div data-bbox="100 50 900 80" data-label="Section-Header"><h2>Intro</h2>'
        '</div><div data-bbox="100 90 900 200" data-label="Text"><p>Body.</p></div>'
    )

    surya = parse_surya_html(content, _SIZE, page_no=1)
    chandra = parse_chandra_html(content, _SIZE, page_no=1)

    assert surya.export_to_dict() == chandra.export_to_dict()


def test_end_token_only_is_an_empty_page():
    doc = parse_surya_html("<|im_end|>", _SIZE, page_no=1)

    assert not (doc.texts or doc.tables or doc.pictures)
    assert 1 in doc.pages


def test_response_without_html_is_invalid():
    with pytest.raises(ValueError):
        parse_surya_html("I cannot read this page.<|im_end|>", _SIZE, page_no=1)
