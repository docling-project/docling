"""Tests for chandra-ocr-2 HTML-with-bbox parser."""

import pytest
from docling_core.types.doc import DocItemLabel, Size

from docling.utils.chandra_utils import parse_chandra_html


@pytest.fixture
def page_size() -> Size:
    """Standard page size for tests: 500x700 points."""
    return Size(width=500.0, height=700.0)


class TestParseSingleTextBlock:
    def test_text_and_bbox(self, page_size: Size):
        html = '<div data-bbox="100 200 400 300" data-label="Text">Hello world</div>'
        doc = parse_chandra_html(html, page_size, page_no=1)

        items = list(doc.iterate_items())
        assert len(items) == 1

        item, _ = items[0]
        assert item.label == DocItemLabel.TEXT
        assert "Hello world" in item.text

        # bbox scaled: 100*0.5=50, 200*0.7=140, 400*0.5=200, 300*0.7=210
        prov = item.prov[0]
        assert abs(prov.bbox.l - 50.0) < 0.01
        assert abs(prov.bbox.t - 140.0) < 0.01
        assert abs(prov.bbox.r - 200.0) < 0.01
        assert abs(prov.bbox.b - 210.0) < 0.01


class TestParseTableBlock:
    def test_table_with_correct_bbox(self, page_size: Size):
        html = (
            '<div data-bbox="50 50 950 500" data-label="Table">'
            "<table><tr><th>A</th><th>B</th></tr>"
            "<tr><td>1</td><td>2</td></tr></table>"
            "</div>"
        )
        doc = parse_chandra_html(html, page_size, page_no=1)

        items = list(doc.iterate_items())
        assert len(items) == 1

        item, _ = items[0]
        assert item.label == DocItemLabel.TABLE

        prov = item.prov[0]
        # 50*0.5=25, 50*0.7=35, 950*0.5=475, 500*0.7=350
        assert abs(prov.bbox.l - 25.0) < 0.01
        assert abs(prov.bbox.t - 35.0) < 0.01
        assert abs(prov.bbox.r - 475.0) < 0.01
        assert abs(prov.bbox.b - 350.0) < 0.01


class TestParseMultipleBlocks:
    def test_three_different_blocks(self, page_size: Size):
        html = (
            '<div data-bbox="100 10 900 50" data-label="Title">Main Title</div>\n'
            '<div data-bbox="100 100 900 300" data-label="Text">Body text here.</div>\n'
            '<div data-bbox="100 900 900 950" data-label="Page-Footer">Page 1</div>'
        )
        doc = parse_chandra_html(html, page_size, page_no=1)

        items = list(doc.iterate_items())
        assert len(items) == 3


class TestParseEmptyContent:
    def test_empty_string(self, page_size: Size):
        doc = parse_chandra_html("", page_size, page_no=1)
        items = list(doc.iterate_items())
        assert len(items) == 0


class TestParseSectionHeader:
    def test_section_header_text(self, page_size: Size):
        html = '<div data-label="Section-Header" data-bbox="50 50 500 100">Introduction</div>'
        doc = parse_chandra_html(html, page_size, page_no=1)

        items = list(doc.iterate_items())
        assert len(items) == 1

        item, _ = items[0]
        # Section-Header should be added as heading
        assert "Introduction" in item.text


class TestParseAttributeOrder:
    """data-label and data-bbox can appear in either order."""

    def test_label_before_bbox(self, page_size: Size):
        html = '<div data-label="Text" data-bbox="0 0 500 500">Order test</div>'
        doc = parse_chandra_html(html, page_size, page_no=1)
        items = list(doc.iterate_items())
        assert len(items) == 1

    def test_bbox_before_label(self, page_size: Size):
        html = '<div data-bbox="0 0 500 500" data-label="Text">Order test</div>'
        doc = parse_chandra_html(html, page_size, page_no=1)
        items = list(doc.iterate_items())
        assert len(items) == 1


class TestParseNestedContent:
    """Inner content may contain nested HTML tags."""

    def test_strips_inner_html(self, page_size: Size):
        html = (
            '<div data-bbox="0 0 500 500" data-label="Text">'
            "Some <b>bold</b> and <i>italic</i> text"
            "</div>"
        )
        doc = parse_chandra_html(html, page_size, page_no=1)
        items = list(doc.iterate_items())
        assert len(items) == 1
        item, _ = items[0]
        assert "bold" in item.text
        assert "italic" in item.text
        # Tags should be stripped
        assert "<b>" not in item.text
