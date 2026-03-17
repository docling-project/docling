from types import SimpleNamespace
from unittest.mock import MagicMock

from docling_core.types.doc import BoundingBox, DocItemLabel
from pydantic import AnyUrl

from docling.datamodel.base_models import Cluster, TextElement
from docling.models.stages.reading_order.readingorder_model import ReadingOrderModel


def _make_text_element(
    text: str,
    hyperlink: str | None,
    cluster_id: int,
) -> TextElement:
    return TextElement(
        label=DocItemLabel.TEXT,
        id=cluster_id,
        page_no=0,
        text=text,
        hyperlink=AnyUrl(hyperlink) if hyperlink is not None else None,
        cluster=Cluster(
            id=cluster_id,
            label=DocItemLabel.TEXT,
            bbox=BoundingBox(l=0, t=0, r=10, b=10),
        ),
    )


class TestMergeElements:
    def test_merge_does_not_expand_partial_hyperlink_coverage(self):
        model = ReadingOrderModel.__new__(ReadingOrderModel)
        element = _make_text_element("prefix", None, 1)
        merged_elem = _make_text_element("linked", "https://example.com", 2)
        new_item = SimpleNamespace(
            label=DocItemLabel.TEXT,
            text="prefix",
            orig="prefix",
            prov=[],
            hyperlink=None,
        )

        model._merge_elements(element, merged_elem, new_item, page_height=100)

        assert new_item.text == "prefix linked"
        assert new_item.orig == "prefix linked"
        assert len(new_item.prov) == 1
        assert new_item.hyperlink is None

    def test_merge_keeps_same_hyperlink(self):
        model = ReadingOrderModel.__new__(ReadingOrderModel)
        element = _make_text_element("first", "https://example.com", 1)
        merged_elem = _make_text_element("second", "https://example.com", 2)
        new_item = SimpleNamespace(
            label=DocItemLabel.TEXT,
            text="first",
            orig="first",
            prov=[],
            hyperlink=AnyUrl("https://example.com"),
        )

        model._merge_elements(element, merged_elem, new_item, page_height=100)

        assert str(new_item.hyperlink) == "https://example.com/"


class TestHandleTextElement:
    def test_formula_hyperlink_is_forwarded(self):
        model = ReadingOrderModel.__new__(ReadingOrderModel)
        out_doc = MagicMock()
        out_doc.add_text.return_value = object()
        element = TextElement(
            label=DocItemLabel.FORMULA,
            id=1,
            page_no=0,
            text="E = mc^2",
            hyperlink=AnyUrl("https://example.com/formula"),
            cluster=Cluster(
                id=1,
                label=DocItemLabel.FORMULA,
                bbox=BoundingBox(l=0, t=0, r=10, b=10),
            ),
        )

        model._handle_text_element(
            element=element,
            out_doc=out_doc,
            current_list=None,
            page_height=100,
        )

        kwargs = out_doc.add_text.call_args.kwargs
        assert kwargs["label"] == DocItemLabel.FORMULA
        assert str(kwargs["hyperlink"]) == "https://example.com/formula"
