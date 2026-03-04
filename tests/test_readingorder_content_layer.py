"""Unit tests for content_layer resolution in ReadingOrderModel.

Verifies that page_header/page_footer elements are assigned
ContentLayer.FURNITURE regardless of whether they are standalone
TextElements or children of container groups (issue #3015).
"""

import pytest
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.document import ContentLayer

from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)


@pytest.fixture
def model() -> ReadingOrderModel:
    return ReadingOrderModel(options=ReadingOrderOptions())


class TestResolveContentLayer:
    """Tests for _resolve_content_layer static method."""

    def test_page_footer_returns_furniture(self, model):
        assert (
            model._resolve_content_layer(DocItemLabel.PAGE_FOOTER)
            == ContentLayer.FURNITURE
        )

    def test_page_header_returns_furniture(self, model):
        assert (
            model._resolve_content_layer(DocItemLabel.PAGE_HEADER)
            == ContentLayer.FURNITURE
        )

    def test_text_returns_body(self, model):
        assert (
            model._resolve_content_layer(DocItemLabel.TEXT) == ContentLayer.BODY
        )

    def test_list_item_returns_body(self, model):
        assert (
            model._resolve_content_layer(DocItemLabel.LIST_ITEM)
            == ContentLayer.BODY
        )

    def test_section_header_returns_body(self, model):
        assert (
            model._resolve_content_layer(DocItemLabel.SECTION_HEADER)
            == ContentLayer.BODY
        )

    def test_caption_returns_body(self, model):
        assert (
            model._resolve_content_layer(DocItemLabel.CAPTION)
            == ContentLayer.BODY
        )
