"""
Unit tests for facade/parser_processor.py.

Covers static/pure helpers and __call__ routing logic.
All external services and file I/O are mocked; no real documents required.
"""
import asyncio
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from PIL import Image

from docling_core.types.doc import (
    BoundingBox,
    DescriptionAnnotation,
    DocItemLabel,
    PictureItem,
    ProvenanceItem,
)
from docling_core.types.doc.base import CoordOrigin
from langchain_core.documents import Document

from facade.parser_processor import (
    DocumentProcessor,
    GenosServiceException,
    GenericDocumentLoader,
    IntelligentDocumentProcessor,
    TabularLoader,
)
from docling.prompts.prompt_manager import LLMApiError


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def dp():
    """DocumentProcessor with __init__ bypassed and all sub-processors mocked."""
    proc = object.__new__(DocumentProcessor)
    proc._intel = MagicMock()
    proc._hwp = MagicMock()
    proc._docx = MagicMock()
    proc._generic = MagicMock()
    proc._whisper_url = ""
    proc._whisper_req_data = {}
    proc._whisper_chunk_sec = 29
    proc._log_level = 4  # __call__ 이 setup_logging 에서 참조 (정상 __init__ 우회 보강)
    # __init__ 이 self._intel._xlsx_cfg 를 alias 하는 속성(이슈 #288). 우회 시 tabular 기본값으로 보강.
    proc._xlsx_cfg = {"processing_mode": "tabular", "header_row": 0, "multi_table": False}
    return proc


@pytest.fixture
def intel():
    """IntelligentDocumentProcessor with __init__ bypassed."""
    return object.__new__(IntelligentDocumentProcessor)


# ─── Helpers for _docling_to_parse_format tests ───────────────────────────────

def _make_prov(page_no=1):
    prov = MagicMock()
    prov.page_no = page_no
    prov.bbox = BoundingBox(l=10.0, t=20.0, r=90.0, b=80.0, coord_origin=CoordOrigin.TOPLEFT)
    return prov


def _make_text_item(text="hello", label="paragraph", page_no=1, has_prov=True, level=1):
    item = MagicMock()
    item.prov = [_make_prov(page_no)] if has_prov else None
    item.label = MagicMock()
    item.label.value = label
    item.text = text
    item.level = level  # needed when label == "section_header"
    return item


def _make_picture_item(page_no=1, annotations=None):
    return PictureItem(
        self_ref="#/pictures/0",
        parent=None,
        children=[],
        label=DocItemLabel.PICTURE,
        prov=[
            ProvenanceItem(
                page_no=page_no,
                bbox=BoundingBox(l=10, t=20, r=90, b=80, coord_origin=CoordOrigin.TOPLEFT),
                charspan=(0, 0),
            )
        ],
        annotations=annotations or [],
    )


def _make_mock_doc(items, num_pages=1):
    doc = MagicMock()
    doc.num_pages.return_value = num_pages
    doc.iterate_items.return_value = [(item, None) for item in items]
    page_size = MagicMock()
    page_size.width = 595.0
    page_size.height = 842.0
    doc.pages = {1: MagicMock(size=page_size)}
    return doc


# ─── _get_normalized_coords ───────────────────────────────────────────────────

@pytest.mark.unit
class TestGetNormalizedCoords:
    def test_topleft_origin_produces_four_corners(self):
        bbox = BoundingBox(l=10, t=20, r=90, b=80, coord_origin=CoordOrigin.TOPLEFT)
        result = DocumentProcessor._get_normalized_coords(bbox, page_w=100.0, page_h=100.0)

        assert len(result) == 4
        assert result[0] == {"x": 0.1, "y": 0.2}   # top-left
        assert result[1] == {"x": 0.9, "y": 0.2}   # top-right
        assert result[2] == {"x": 0.9, "y": 0.8}   # bottom-right
        assert result[3] == {"x": 0.1, "y": 0.8}   # bottom-left

    def test_full_page_bbox_spans_0_to_1(self):
        bbox = BoundingBox(l=0, t=0, r=100, b=100, coord_origin=CoordOrigin.TOPLEFT)
        result = DocumentProcessor._get_normalized_coords(bbox, page_w=100.0, page_h=100.0)

        xs = {p["x"] for p in result}
        ys = {p["y"] for p in result}
        assert xs == {0.0, 1.0}
        assert ys == {0.0, 1.0}

    def test_each_corner_has_x_and_y(self):
        bbox = BoundingBox(l=5, t=5, r=50, b=50, coord_origin=CoordOrigin.TOPLEFT)
        result = DocumentProcessor._get_normalized_coords(bbox, page_w=100.0, page_h=100.0)
        assert all(isinstance(p, dict) and "x" in p and "y" in p for p in result)


# ─── _audio_to_parse_format ───────────────────────────────────────────────────

@pytest.mark.unit
class TestAudioToParseFormat:
    def test_returns_exactly_one_element(self):
        result = DocumentProcessor._audio_to_parse_format("hello world")
        assert len(result["elements"]) == 1

    def test_element_category_is_paragraph(self):
        result = DocumentProcessor._audio_to_parse_format("hello")
        assert result["elements"][0]["category"] == "paragraph"

    def test_content_matches_input_text(self):
        text = "transcribed audio text"
        result = DocumentProcessor._audio_to_parse_format(text)
        assert result["elements"][0]["content"] == text

    def test_page_is_one(self):
        result = DocumentProcessor._audio_to_parse_format("x")
        assert result["elements"][0]["page"] == 1

    def test_usage_pages_is_one(self):
        result = DocumentProcessor._audio_to_parse_format("x")
        assert result["usage"]["pages"] == 1

    def test_coordinates_are_empty(self):
        result = DocumentProcessor._audio_to_parse_format("x")
        assert result["elements"][0]["coordinates"] == []


# ─── _sheet_to_html ───────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSheetToHtml:
    def test_empty_data_rows_returns_empty_table_tag(self):
        assert DocumentProcessor._sheet_to_html({"data_rows": []}) == "<table></table>"

    def test_column_headers_rendered(self):
        sheet = {"data_rows": [{"name": "Alice", "age": 30}]}
        html = DocumentProcessor._sheet_to_html(sheet)
        assert "<th>name</th>" in html
        assert "<th>age</th>" in html

    def test_cell_value_rendered(self):
        sheet = {"data_rows": [{"col": "value"}]}
        assert "<td>value</td>" in DocumentProcessor._sheet_to_html(sheet)

    def test_all_data_rows_rendered(self):
        sheet = {"data_rows": [{"x": 1}, {"x": 2}, {"x": 3}]}
        html = DocumentProcessor._sheet_to_html(sheet)
        assert html.count("<tr>") == 4  # 1 header + 3 data rows


# ─── _tabular_to_parse_format ─────────────────────────────────────────────────

@pytest.mark.unit
class TestTabularToParseFormat:
    def _make_data_dict(self, n=1):
        return {
            "data": [
                {"sheet_name": f"Sheet{i+1}", "data_rows": [{"a": i}], "data_types": []}
                for i in range(n)
            ]
        }

    def test_empty_data_returns_empty_elements(self):
        result = DocumentProcessor._tabular_to_parse_format({"data": []})
        assert result["elements"] == []
        assert result["usage"]["pages"] == 0

    def test_one_sheet_produces_one_element(self):
        result = DocumentProcessor._tabular_to_parse_format(self._make_data_dict(1))
        assert len(result["elements"]) == 1
        assert result["elements"][0]["category"] == "table"
        assert result["elements"][0]["page"] == 1

    def test_two_sheets_use_sequential_pages(self):
        result = DocumentProcessor._tabular_to_parse_format(self._make_data_dict(2))
        assert result["elements"][0]["page"] == 1
        assert result["elements"][1]["page"] == 2

    def test_usage_pages_equals_sheet_count(self):
        result = DocumentProcessor._tabular_to_parse_format(self._make_data_dict(3))
        assert result["usage"]["pages"] == 3

    def test_element_has_required_keys(self):
        element = DocumentProcessor._tabular_to_parse_format(self._make_data_dict(1))["elements"][0]
        for key in ("category", "content", "coordinates", "id", "page"):
            assert key in element


# ─── _langchain_to_parse_format ───────────────────────────────────────────────

@pytest.mark.unit
class TestLangchainToParseFormat:
    def test_page_0_becomes_1(self):
        docs = [Document(page_content="text", metadata={"page": 0})]
        assert DocumentProcessor._langchain_to_parse_format(docs)["elements"][0]["page"] == 1

    def test_page_2_becomes_3(self):
        docs = [Document(page_content="text", metadata={"page": 2})]
        assert DocumentProcessor._langchain_to_parse_format(docs)["elements"][0]["page"] == 3

    def test_missing_page_uses_index_plus_one(self):
        docs = [Document(page_content="a", metadata={})]
        assert DocumentProcessor._langchain_to_parse_format(docs)["elements"][0]["page"] == 1

    def test_usage_pages_is_max_converted_page(self):
        docs = [
            Document(page_content="a", metadata={"page": 0}),
            Document(page_content="b", metadata={"page": 4}),
        ]
        assert DocumentProcessor._langchain_to_parse_format(docs)["usage"]["pages"] == 5

    def test_element_has_required_keys(self):
        docs = [Document(page_content="text", metadata={})]
        element = DocumentProcessor._langchain_to_parse_format(docs)["elements"][0]
        for key in ("category", "content", "coordinates", "id", "page"):
            assert key in element

    def test_category_is_paragraph(self):
        docs = [Document(page_content="text", metadata={})]
        assert DocumentProcessor._langchain_to_parse_format(docs)["elements"][0]["category"] == "paragraph"


# ─── _docling_to_parse_format ─────────────────────────────────────────────────

@pytest.mark.unit
class TestDoclingToParseFormat:
    def test_item_without_prov_has_empty_coordinates(self):
        items = [_make_text_item(text="keep", has_prov=True), _make_text_item(text="no-prov", has_prov=False)]
        doc = _make_mock_doc(items)
        result = DocumentProcessor._docling_to_parse_format(doc)
        assert len(result["elements"]) == 2
        no_prov = next(e for e in result["elements"] if e["content"] == "no-prov")
        assert no_prov["coordinates"] == []

    def test_element_ids_are_sequential(self):
        doc = _make_mock_doc([_make_text_item() for _ in range(3)])
        result = DocumentProcessor._docling_to_parse_format(doc)
        assert [e["id"] for e in result["elements"]] == [0, 1, 2]

    def test_required_keys_present(self):
        doc = _make_mock_doc([_make_text_item()])
        element = DocumentProcessor._docling_to_parse_format(doc)["elements"][0]
        for key in ("category", "content", "coordinates", "id", "page"):
            assert key in element

    def test_usage_pages_comes_from_doc_num_pages(self):
        doc = _make_mock_doc([], num_pages=5)
        assert DocumentProcessor._docling_to_parse_format(doc)["usage"]["pages"] == 5

    def test_coordinates_is_list_of_four_points(self):
        doc = _make_mock_doc([_make_text_item()])
        coords = DocumentProcessor._docling_to_parse_format(doc)["elements"][0]["coordinates"]
        assert isinstance(coords, list) and len(coords) == 4

    def test_category_uses_label_value_directly(self):
        doc = _make_mock_doc([_make_text_item(label="section_header")])
        result = DocumentProcessor._docling_to_parse_format(doc)
        assert result["elements"][0]["category"] == "section_header"

    def test_picture_annotation_is_mapped_to_content(self):
        picture = _make_picture_item(
            annotations=[
                DescriptionAnnotation(
                    text="문맥 기반 이미지 설명",
                    provenance="facade_image_description",
                )
            ]
        )
        doc = _make_mock_doc([picture])
        element = DocumentProcessor._docling_to_parse_format(doc)["elements"][0]
        assert element["category"] == "picture"
        assert element["content"] == "문맥 기반 이미지 설명"


# ─── DocumentProcessor.__call__ routing ──────────────────────────────────────

_MOCK_RESULT = {"elements": [], "usage": {"pages": 1}}


@pytest.mark.unit
@pytest.mark.parametrize("filename,expected_method", [
    ("a.wav",  "_parse_audio"),
    ("a.mp3",  "_parse_audio"),
    ("a.m4a",  "_parse_audio"),
    ("a.csv",  "_parse_tabular"),
    ("a.xlsx", "_parse_tabular"),
    ("a.hwp",  "_parse_hwp_hwpx"),
    ("a.hwpx", "_parse_hwp_hwpx"),
    ("a.docx", "_parse_docx"),
    ("a.pdf",  "_parse_docling"),
    ("a.txt",  "_parse_other"),
    ("a.pptx", "_parse_other"),
])
def test_call_routes_to_correct_method(dp, filename, expected_method):
    """__call__ dispatches each file extension to the right internal method."""
    parse_mocks = {
        "_parse_audio":    MagicMock(return_value="transcript"),
        "_parse_tabular":  MagicMock(return_value={"data": []}),
        "_parse_hwp_hwpx": MagicMock(return_value=MagicMock()),
        "_parse_docx":     MagicMock(return_value=MagicMock()),
        "_parse_docling":  MagicMock(return_value=MagicMock()),
        "_parse_other":    MagicMock(return_value=[]),
    }
    for name, mock in parse_mocks.items():
        setattr(dp, name, mock)

    with patch.object(DocumentProcessor, "_docling_to_parse_format",   return_value=_MOCK_RESULT), \
         patch.object(DocumentProcessor, "_audio_to_parse_format",     return_value=_MOCK_RESULT), \
         patch.object(DocumentProcessor, "_tabular_to_parse_format",   return_value=_MOCK_RESULT), \
         patch.object(DocumentProcessor, "_langchain_to_parse_format", return_value=_MOCK_RESULT):
        result = asyncio.run(dp(None, filename))

    parse_mocks[expected_method].assert_called_once()
    for name, mock in parse_mocks.items():
        if name != expected_method:
            mock.assert_not_called()
    assert "elements" in result
    assert "usage" in result


@pytest.mark.unit
def test_docx_coordinates_cleared_after_parsing(dp):
    """For .docx files, element coordinates are reset to [] after parsing."""
    elements_with_coords = [
        {"category": "paragraph", "content": "text",
         "coordinates": [{"x": 0.1, "y": 0.2}], "id": 0, "page": 1}
    ]
    dp._parse_docx = MagicMock(return_value=MagicMock())
    with patch.object(DocumentProcessor, "_docling_to_parse_format",
                      return_value={"elements": elements_with_coords, "usage": {"pages": 1}}):
        result = asyncio.run(dp(None, "doc.docx"))

    for element in result["elements"]:
        assert element["coordinates"] == []


# ─── IntelligentDocumentProcessor.check_glyph_text ───────────────────────────

@pytest.mark.unit
class TestCheckGlyphText:
    def test_single_glyph_detected_at_default_threshold(self, intel):
        assert intel.check_glyph_text("GLYPH123") is True

    def test_threshold_requires_minimum_match_count(self, intel):
        assert intel.check_glyph_text("GLYPH123", threshold=2) is False
        assert intel.check_glyph_text("GLYPH123 GLYPHABC", threshold=2) is True

    def test_plain_text_returns_false(self, intel):
        assert intel.check_glyph_text("일반 텍스트 내용") is False

    def test_empty_string_returns_false(self, intel):
        assert intel.check_glyph_text("") is False

    def test_none_returns_false(self, intel):
        assert intel.check_glyph_text(None) is False


@pytest.mark.unit
class TestEnrichImageDescriptions:
    def _make_intel(self, enabled=True):
        intel = object.__new__(IntelligentDocumentProcessor)
        intel.image_description_enabled = enabled
        intel.image_description_api_url = "http://example.com/v1/chat/completions"
        intel.image_description_api_key = "secret"
        intel.image_description_model = "mock-model"
        intel.image_description_timeout = 10.0
        intel.image_description_concurrency = 1
        intel.image_description_before_items = 2
        intel.image_description_after_items = 2
        intel.image_description_max_context_chars = 1500
        intel.image_description_include_caption = False
        intel.image_description_include_section_header = False
        intel.image_description_same_page_first = True
        intel.image_description_headers = {}
        intel.image_description_params = {}
        intel.image_description_provenance = "facade_image_description"
        intel.image_description_prompt_template = (
            "[앞 문맥]\\n{before_context}\\n[캡션]\\n{caption}\\n[뒤 문맥]\\n{after_context}"
        )
        return intel

    def test_disabled_option_skips_image_description_api_call(self):
        intel = self._make_intel(enabled=False)
        doc = MagicMock()
        doc.iterate_items.return_value = []

        with patch("facade.parser_processor.api_image_request") as mock_api:
            result = intel.enrich_image_descriptions(doc)

        assert result is doc
        mock_api.assert_not_called()

    def test_enrichment_adds_description_annotation_with_context(self):
        intel = self._make_intel(enabled=True)
        before_item = _make_text_item(text="앞 문단 텍스트", page_no=1)
        picture_item = _make_picture_item(page_no=1)
        after_item = _make_text_item(text="뒤 문단 텍스트", page_no=1)
        doc = _make_mock_doc([before_item, picture_item, after_item])

        with patch.object(
            PictureItem,
            "get_image",
            return_value=Image.new("RGB", (8, 8), color="white"),
        ), patch(
            "facade.parser_processor.api_image_request",
            return_value="문맥 기반 설명 결과",
        ) as mock_api:
            result = intel.enrich_image_descriptions(doc)

        assert result is doc
        descriptions = [
            ann
            for ann in picture_item.annotations
            if isinstance(ann, DescriptionAnnotation)
        ]
        assert descriptions
        assert descriptions[0].text == "문맥 기반 설명 결과"
        assert descriptions[0].provenance == "facade_image_description"

        prompt = mock_api.call_args.kwargs["prompt"]
        assert "앞 문단 텍스트" in prompt
        assert "뒤 문단 텍스트" in prompt

    def test_enrichment_vlm_unreachable_is_ignored(self):
        intel = self._make_intel(enabled=True)
        before_item = _make_text_item(text="앞 문단 텍스트", page_no=1)
        picture_item = _make_picture_item(page_no=1)
        after_item = _make_text_item(text="뒤 문단 텍스트", page_no=1)
        doc = _make_mock_doc([before_item, picture_item, after_item])

        with patch.object(
            PictureItem,
            "get_image",
            return_value=Image.new("RGB", (8, 8), color="white"),
        ), patch(
            "facade.parser_processor.api_image_request",
            side_effect=RuntimeError("VLM endpoint is unreachable"),
        ):
            result = intel.enrich_image_descriptions(doc)

        assert result is doc
        descriptions = [
            ann
            for ann in picture_item.annotations
            if isinstance(ann, DescriptionAnnotation)
        ]
        assert descriptions == []


@pytest.mark.unit
def test_enrichment_provider_error_is_rethrown_as_genos_exception(intel):
    raw_error = """{"object":"error","message":"This model's maximum context length is 16384 tokens.","type":"BadRequestError","param":null,"code":400}"""
    intel.enrichment_options = MagicMock()
    dummy_doc = MagicMock()

    with patch(
        "facade.parser_processor.enrich_document",
        side_effect=LLMApiError(raw_error, status_code=400),
    ):
        with pytest.raises(GenosServiceException) as exc_info:
            intel.enrichment(dummy_doc)

    assert exc_info.value.error_msg == raw_error


# ─── _normalize_output_format ────────────────────────────────────────────────

@pytest.mark.unit
class TestNormalizeOutputFormat:
    def test_json_returns_json(self):
        assert DocumentProcessor._normalize_output_format("json") == "json"

    def test_html_returns_html(self):
        assert DocumentProcessor._normalize_output_format("html") == "html"

    def test_markdown_returns_markdown(self):
        assert DocumentProcessor._normalize_output_format("markdown") == "markdown"

    def test_uppercase_is_normalized(self):
        assert DocumentProcessor._normalize_output_format("JSON") == "json"
        assert DocumentProcessor._normalize_output_format("HTML") == "html"
        assert DocumentProcessor._normalize_output_format("Markdown") == "markdown"

    def test_whitespace_is_stripped(self):
        assert DocumentProcessor._normalize_output_format("  json  ") == "json"

    def test_invalid_value_falls_back_to_json(self):
        assert DocumentProcessor._normalize_output_format("xml") == "json"

    def test_empty_string_falls_back_to_json(self):
        assert DocumentProcessor._normalize_output_format("") == "json"


# ─── _normalize_table_format ─────────────────────────────────────────────────

@pytest.mark.unit
class TestNormalizeTableFormat:
    def test_html_returns_html(self):
        assert DocumentProcessor._normalize_table_format("html") == "html"

    def test_markdown_returns_markdown(self):
        assert DocumentProcessor._normalize_table_format("markdown") == "markdown"

    def test_uppercase_is_normalized(self):
        assert DocumentProcessor._normalize_table_format("HTML") == "html"
        assert DocumentProcessor._normalize_table_format("MARKDOWN") == "markdown"

    def test_whitespace_is_stripped(self):
        assert DocumentProcessor._normalize_table_format("  html  ") == "html"

    def test_invalid_value_falls_back_to_html(self):
        assert DocumentProcessor._normalize_table_format("text") == "html"

    def test_empty_string_falls_back_to_html(self):
        assert DocumentProcessor._normalize_table_format("") == "html"


# ─── _export_table_content ───────────────────────────────────────────────────

def _make_table_item(export_html="<table></table>", export_markdown="| a |", text="fallback"):
    item = MagicMock()
    item.export_to_html.return_value = export_html
    item.export_to_markdown.return_value = export_markdown
    item.text = text
    item.data = MagicMock()
    item.data.table_cells = []
    return item


@pytest.mark.unit
class TestExportTableContent:
    def test_html_format_calls_export_to_html(self):
        item = _make_table_item()
        doc = MagicMock()
        result = DocumentProcessor._export_table_content(item, doc, table_format="html")
        item.export_to_html.assert_called_once_with(doc=doc)
        assert result == "<table></table>"

    def test_markdown_format_calls_export_to_markdown(self):
        item = _make_table_item()
        doc = MagicMock()
        result = DocumentProcessor._export_table_content(item, doc, table_format="markdown")
        item.export_to_markdown.assert_called_once_with(doc=doc)
        assert result == "| a |"

    def test_default_format_is_html(self):
        item = _make_table_item()
        doc = MagicMock()
        DocumentProcessor._export_table_content(item, doc)
        item.export_to_html.assert_called_once()

    def test_empty_export_falls_back_to_cell_text(self):
        item = _make_table_item(export_html="   ")
        cell = MagicMock()
        cell.text = "cell value"
        item.data.table_cells = [cell]
        doc = MagicMock()
        result = DocumentProcessor._export_table_content(item, doc, table_format="html")
        assert result == "cell value"

    def test_export_exception_falls_back_to_cell_text(self):
        item = MagicMock()
        item.export_to_html.side_effect = RuntimeError("export failed")
        cell = MagicMock()
        cell.text = "rescued"
        item.data.table_cells = [cell]
        item.text = ""
        doc = MagicMock()
        result = DocumentProcessor._export_table_content(item, doc, table_format="html")
        assert result == "rescued"

    def test_all_fallbacks_fail_returns_item_text(self):
        item = MagicMock()
        item.export_to_html.side_effect = RuntimeError
        item.data.table_cells = []
        item.text = "last resort"
        doc = MagicMock()
        result = DocumentProcessor._export_table_content(item, doc, table_format="html")
        assert result == "last resort"


# ─── _docling_to_content ─────────────────────────────────────────────────────

def _make_proc_with_format(output_format: str, table_format: str):
    proc = object.__new__(DocumentProcessor)
    proc._output_format = output_format
    proc._table_format = table_format
    return proc


@pytest.mark.unit
class TestDoclingToContent:
    def test_html_format_calls_export_to_html(self):
        proc = _make_proc_with_format("html", "html")
        doc = MagicMock()
        doc.export_to_html.return_value = "<html>content</html>"
        result = proc._docling_to_content(doc)
        doc.export_to_html.assert_called_once()
        assert result == "<html>content</html>"

    def test_markdown_format_with_markdown_table_calls_export_to_markdown(self):
        proc = _make_proc_with_format("markdown", "markdown")
        doc = MagicMock()
        doc.export_to_markdown.return_value = "# heading\n| a | b |"
        result = proc._docling_to_content(doc)
        doc.export_to_markdown.assert_called_once()
        assert result == "# heading\n| a | b |"

    def test_markdown_format_with_html_table_uses_replace(self):
        proc = _make_proc_with_format("markdown", "html")
        doc = MagicMock()
        doc.export_to_markdown.return_value = "# heading\n| a | b |"
        doc.iterate_items.return_value = []
        result = proc._docling_to_content(doc)
        doc.export_to_markdown.assert_called_once()
        assert isinstance(result, str)

    def test_json_format_returns_empty_string(self):
        proc = _make_proc_with_format("json", "html")
        doc = MagicMock()
        result = proc._docling_to_content(doc)
        assert result == ""
        doc.export_to_html.assert_not_called()
        doc.export_to_markdown.assert_not_called()


# ─── _build_docling_response ─────────────────────────────────────────────────

def _make_parse_format_result():
    return {
        "elements": [
            {"category": "paragraph", "content": "text",
             "coordinates": [{"x": 0.1, "y": 0.2}], "id": 0, "page": 1}
        ],
        "usage": {"pages": 1},
    }


@pytest.mark.unit
class TestBuildDoclingResponse:
    def test_json_format_returns_elements_structure(self):
        proc = _make_proc_with_format("json", "html")
        doc = MagicMock()
        with patch.object(DocumentProcessor, "_docling_to_parse_format",
                          side_effect=lambda *a, **kw: _make_parse_format_result()):
            result = proc._build_docling_response(doc)
        assert "elements" in result
        assert "usage" in result
        assert "content" not in result

    def test_html_format_returns_content_structure(self):
        proc = _make_proc_with_format("html", "html")
        doc = MagicMock()
        doc.num_pages.return_value = 2
        with patch.object(DocumentProcessor, "_docling_to_content", return_value="<html/>"):
            result = proc._build_docling_response(doc)
        assert result["content"] == "<html/>"
        assert result["elements"] == []
        assert result["usage"]["pages"] == 2

    def test_markdown_format_returns_content_structure(self):
        proc = _make_proc_with_format("markdown", "markdown")
        doc = MagicMock()
        doc.num_pages.return_value = 1
        with patch.object(DocumentProcessor, "_docling_to_content", return_value="# title"):
            result = proc._build_docling_response(doc)
        assert result["content"] == "# title"
        assert result["elements"] == []

    def test_json_format_clear_coordinates_empties_coords(self):
        proc = _make_proc_with_format("json", "html")
        doc = MagicMock()
        with patch.object(DocumentProcessor, "_docling_to_parse_format",
                          side_effect=lambda *a, **kw: _make_parse_format_result()):
            result = proc._build_docling_response(doc, clear_coordinates=True)
        for element in result["elements"]:
            assert element["coordinates"] == []

    def test_json_format_without_clear_coordinates_keeps_coords(self):
        proc = _make_proc_with_format("json", "html")
        doc = MagicMock()
        with patch.object(DocumentProcessor, "_docling_to_parse_format",
                          side_effect=lambda *a, **kw: _make_parse_format_result()):
            result = proc._build_docling_response(doc, clear_coordinates=False)
        assert result["elements"][0]["coordinates"] != []

    def test_glyph_with_suffix_variants_detected(self, intel):
        assert intel.check_glyph_text("GLYPHXYZ") is True


# ─── TabularLoader.check_sql_dtypes ──────────────────────────────────────────

@pytest.mark.unit
class TestCheckSqlDtypes:
    @pytest.fixture
    def loader(self):
        return object.__new__(TabularLoader)

    def test_int_column_maps_to_int_type(self, loader):
        df = pd.DataFrame({"n": [1, 2, 3]})
        _, dtypes = loader.check_sql_dtypes(df)
        col_type = next(d[1] for d in dtypes if d[0] == "n")
        assert "INT" in col_type

    def test_float_column_maps_to_float(self, loader):
        df = pd.DataFrame({"f": [1.1, 2.2, 3.3]})
        _, dtypes = loader.check_sql_dtypes(df)
        col_type = next(d[1] for d in dtypes if d[0] == "f")
        assert col_type == "FLOAT"

    def test_string_column_maps_to_varchar(self, loader):
        df = pd.DataFrame({"s": ["hello", "world"]})
        _, dtypes = loader.check_sql_dtypes(df)
        col_type = next(d[1] for d in dtypes if d[0] == "s")
        assert "VARCHAR" in col_type

    def test_returns_df_and_paired_list(self, loader):
        df = pd.DataFrame({"a": [1]})
        result_df, dtypes = loader.check_sql_dtypes(df)
        assert isinstance(dtypes, list)
        assert len(dtypes) == 1
        assert len(dtypes[0]) == 2  # [col_name, sql_type]


# ─── GenericDocumentLoader.get_real_file_type ─────────────────────────────────

@pytest.mark.unit
class TestGetRealFileType:
    @pytest.fixture
    def loader(self):
        return object.__new__(GenericDocumentLoader)

    def test_pdf_magic_bytes_return_pdf(self, loader, tmp_path):
        f = tmp_path / "fake.txt"
        f.write_bytes(b"%PDF-1.4 fake content")
        assert loader.get_real_file_type(str(f)) == "pdf"

    def test_png_magic_bytes_return_png(self, loader, tmp_path):
        f = tmp_path / "fake.txt"
        f.write_bytes(b"\x89PNG\r\n\x1a\n fake png")
        assert loader.get_real_file_type(str(f)) == "png"

    def test_jpg_magic_bytes_return_jpg(self, loader, tmp_path):
        f = tmp_path / "fake.txt"
        f.write_bytes(b"\xff\xd8\xff fake jpeg")
        assert loader.get_real_file_type(str(f)) == "jpg"

    def test_unknown_magic_returns_file_extension(self, loader, tmp_path):
        f = tmp_path / "test.docx"
        f.write_bytes(b"PK\x03\x04 zip content")
        assert loader.get_real_file_type(str(f)) == ".docx"


# ─── IntelligentDocumentProcessor._build_ocr_options (yaml → ocr_options) ────

@pytest.mark.unit
class TestBuildOcrOptions:
    """yaml 의 ocr.engine 키에 따라 PaddleOcrOptions / UpstageOcrOptions 가 선택되는지 검증."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("UPSTAGE_API_KEY", raising=False)

    def test_default_engine_is_paddle(self):
        from docling.datamodel.pipeline_options import PaddleOcrOptions
        opts = IntelligentDocumentProcessor._build_ocr_options(
            {}, paddle_endpoint="http://paddle.example/ocr"
        )
        assert isinstance(opts, PaddleOcrOptions)
        assert opts.ocr_endpoint == "http://paddle.example/ocr"

    def test_explicit_paddle_engine(self):
        from docling.datamodel.pipeline_options import PaddleOcrOptions
        opts = IntelligentDocumentProcessor._build_ocr_options(
            {"engine": "paddle"}, paddle_endpoint="http://paddle.example/ocr"
        )
        assert isinstance(opts, PaddleOcrOptions)

    def test_upstage_engine_uses_yaml_values(self):
        from docling.datamodel.pipeline_options import UpstageOcrOptions
        opts = IntelligentDocumentProcessor._build_ocr_options(
            {
                "engine": "upstage",
                "upstage": {
                    "api_endpoint": "https://custom.upstage.example/ocr",
                    "api_key": "yaml-key",
                    "model": "ocr",
                    "timeout": 30,
                    "text_score": 0.6,
                    "lang": ["ko"],
                },
            },
            paddle_endpoint="",
        )
        assert isinstance(opts, UpstageOcrOptions)
        assert opts.api_endpoint == "https://custom.upstage.example/ocr"
        assert opts.api_key == "yaml-key"
        assert opts.model == "ocr"
        assert opts.timeout == 30
        assert opts.text_score == 0.6
        assert opts.lang == ["ko"]

    def test_upstage_engine_api_key_falls_back_to_env(self, monkeypatch):
        from docling.datamodel.pipeline_options import UpstageOcrOptions
        monkeypatch.setenv("UPSTAGE_API_KEY", "env-secret")
        opts = IntelligentDocumentProcessor._build_ocr_options(
            {"engine": "upstage", "upstage": {"api_key": ""}},
            paddle_endpoint="",
        )
        assert isinstance(opts, UpstageOcrOptions)
        assert opts.api_key == "env-secret"

    def test_upstage_yaml_key_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("UPSTAGE_API_KEY", "env-secret")
        opts = IntelligentDocumentProcessor._build_ocr_options(
            {"engine": "upstage", "upstage": {"api_key": "yaml-secret"}},
            paddle_endpoint="",
        )
        assert opts.api_key == "yaml-secret"

    def test_unknown_engine_falls_back_to_paddle(self):
        from docling.datamodel.pipeline_options import PaddleOcrOptions
        opts = IntelligentDocumentProcessor._build_ocr_options(
            {"engine": "bogus"}, paddle_endpoint="http://x/ocr"
        )
        assert isinstance(opts, PaddleOcrOptions)

    def test_engine_case_insensitive(self):
        from docling.datamodel.pipeline_options import UpstageOcrOptions
        opts = IntelligentDocumentProcessor._build_ocr_options(
            {"engine": "UPSTAGE", "upstage": {"api_key": "k"}},
            paddle_endpoint="",
        )
        assert isinstance(opts, UpstageOcrOptions)

    # ── yaml 의 잘못된 값으로 startup 이 깨지지 않는지 (#178 CodeRabbit) ─────

    def test_upstage_invalid_timeout_falls_back_to_default(self):
        from docling.datamodel.pipeline_options import UpstageOcrOptions
        opts = IntelligentDocumentProcessor._build_ocr_options(
            {"engine": "upstage", "upstage": {"api_key": "k", "timeout": "not-a-number"}},
            paddle_endpoint="",
        )
        assert isinstance(opts, UpstageOcrOptions)
        assert opts.timeout == 60

    def test_upstage_empty_timeout_falls_back_to_default(self):
        from docling.datamodel.pipeline_options import UpstageOcrOptions
        opts = IntelligentDocumentProcessor._build_ocr_options(
            {"engine": "upstage", "upstage": {"api_key": "k", "timeout": ""}},
            paddle_endpoint="",
        )
        assert opts.timeout == 60

    def test_upstage_zero_timeout_falls_back_to_default(self):
        opts = IntelligentDocumentProcessor._build_ocr_options(
            {"engine": "upstage", "upstage": {"api_key": "k", "timeout": 0}},
            paddle_endpoint="",
        )
        # 0 / 음수 timeout 도 의미 없으므로 default 로 복구
        assert opts.timeout == 60

    def test_upstage_invalid_text_score_falls_back_to_default(self):
        from docling.datamodel.pipeline_options import UpstageOcrOptions
        opts = IntelligentDocumentProcessor._build_ocr_options(
            {"engine": "upstage", "upstage": {"api_key": "k", "text_score": "bad"}},
            paddle_endpoint="",
        )
        assert isinstance(opts, UpstageOcrOptions)
        assert opts.text_score == 0.5

    def test_upstage_numeric_string_timeout_accepted(self):
        # int("60") 처럼 numeric-string 은 정상 변환되어야 한다
        opts = IntelligentDocumentProcessor._build_ocr_options(
            {"engine": "upstage", "upstage": {"api_key": "k", "timeout": "120", "text_score": "0.7"}},
            paddle_endpoint="",
        )
        assert opts.timeout == 120
        assert opts.text_score == 0.7
