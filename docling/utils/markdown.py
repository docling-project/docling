"""Markdown serialization utilities for docling."""

from typing import Any

from docling_core.transforms.serializer.base import (
    BaseFallbackSerializer,
    BaseDocSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownFallbackSerializer,
)
from docling_core.types.doc.document import DoclingDocument, GroupItem, NodeItem
from docling_core.types.doc.labels import GroupLabel
from typing_extensions import override


class MsExcelMarkdownFallbackSerializer(MarkdownFallbackSerializer):
    """Fallback serializer that renders ``GroupLabel.SHEET`` groups as headings.

    When a ``GroupItem`` with ``label=GroupLabel.SHEET`` is encountered the
    group's ``name`` is emitted as a level-2 Markdown heading (``##``) before
    the group's children, matching the visual structure of the original
    workbook where each worksheet has a name.
    """

    @override
    def serialize(
        self,
        *,
        item: NodeItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        if isinstance(item, GroupItem) and item.label == GroupLabel.SHEET:
            parts = doc_serializer.get_parts(item=item, **kwargs)
            content = "\n\n".join(p.text for p in parts if p.text)
            heading = f"## {item.name}"
            text = f"{heading}\n\n{content}" if content else heading
            return create_ser_result(text=text, span_source=parts)
        return super().serialize(
            item=item, doc_serializer=doc_serializer, doc=doc, **kwargs
        )


class MsExcelMarkdownDocSerializer(MarkdownDocSerializer):
    """``MarkdownDocSerializer`` variant for Excel-sourced ``DoclingDocument``\\s.

    Swap in :class:`MsExcelMarkdownFallbackSerializer` so that worksheet
    groups (``GroupLabel.SHEET``) are rendered with their name as a Markdown
    heading without requiring heading nodes to be injected into the document
    model by the backend.

    Usage::

        from docling.utils.markdown import MsExcelMarkdownDocSerializer
        from docling_core.transforms.serializer.markdown import MarkdownParams

        serializer = MsExcelMarkdownDocSerializer(doc=result.document)
        md = serializer.serialize().text
    """

    fallback_serializer: BaseFallbackSerializer = MsExcelMarkdownFallbackSerializer()
