# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""LaTeX serialization for source comparison in the corpus test."""

from typing import Any

from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    BaseTextSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.doclang import DocLangDocSerializer
from docling_core.transforms.serializer.latex import (
    LaTeXDocSerializer,
    LaTeXInlineSerializer,
    LaTeXTextSerializer,
)
from docling_core.types.doc import (
    ContentLayer,
    DoclingDocument,
    FloatingItem,
    InlineGroup,
    NodeItem,
    SectionHeaderItem,
    TextItem,
)
from pydantic import PrivateAttr


def _nested_caption_refs(doc: DoclingDocument) -> set[str]:
    nested: set[str] = set()
    for host, _ in doc.iterate_items(
        with_groups=True,
        traverse_pictures=True,
        included_content_layers=set(ContentLayer),
    ):
        if not isinstance(host, FloatingItem):
            continue
        for ref in host.captions:
            caption = ref.resolve(doc)
            if isinstance(caption, TextItem):
                nested.update(
                    item.self_ref
                    for item, _ in doc.iterate_items(
                        root=caption,
                        with_groups=True,
                        traverse_pictures=True,
                        included_content_layers=set(ContentLayer),
                    )
                    if item.self_ref != caption.self_ref
                )
    return nested


class SourceLaTeXInlineSerializer(LaTeXInlineSerializer):
    """Keep the spacing stored in LaTeX backend inline text runs."""

    def serialize(
        self,
        *,
        item: InlineGroup,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        list_level: int = 0,
        **kwargs: Any,
    ) -> SerializationResult:
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level,
            is_inline_scope=True,
            **kwargs,
        )
        return create_ser_result(
            text="".join(part.text for part in parts if part.text),
            span_source=parts,
        )


class SourceLaTeXTextSerializer(LaTeXTextSerializer):
    """Keep source commands as LaTeX rather than escaped text."""

    def serialize(
        self,
        *,
        item: TextItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        if item.content_layer == ContentLayer.FURNITURE:
            return create_ser_result(text=item.text, span_source=item)
        if isinstance(item, SectionHeaderItem) and item.level in (4, 5):
            if item.text == "" and len(item.children) == 1:
                child = item.children[0].resolve(doc)
                if isinstance(child, InlineGroup):
                    text = doc_serializer.serialize(
                        item=child, is_inline_scope=True, **kwargs
                    ).text
                else:
                    text = doc_serializer.post_process(text=item.text)
            else:
                text = doc_serializer.post_process(
                    text=item.text,
                    formatting=item.formatting,
                    hyperlink=item.hyperlink,
                )
            command = "paragraph" if item.level == 4 else "subparagraph"
            return create_ser_result(text=f"\\{command}{{{text}}}", span_source=item)
        return super().serialize(
            item=item, doc_serializer=doc_serializer, doc=doc, **kwargs
        )


class SourceLaTeXDocSerializer(LaTeXDocSerializer):
    """Serialize rich captions from the LaTeX backend for source comparison."""

    text_serializer: BaseTextSerializer = SourceLaTeXTextSerializer()
    _nested_caption_refs_cache: set[str] | None = PrivateAttr(default=None)

    def serialize_doc(
        self, *, parts: list[SerializationResult], **kwargs: Any
    ) -> SerializationResult:
        preamble_parts: list[SerializationResult] = []
        body_parts: list[SerializationResult] = []
        for part in parts:
            items = part.get_unique_doc_items()
            if items and all(
                item.content_layer == ContentLayer.FURNITURE for item in items
            ):
                preamble_parts.append(part)
            else:
                body_parts.append(part)

        result = super().serialize_doc(parts=body_parts, **kwargs)
        if not preamble_parts:
            return result

        header, separator, body = result.text.partition(r"\begin{document}")
        source_preamble = "\n".join(part.text for part in preamble_parts)
        return create_ser_result(
            text=f"{header}{source_preamble}\n\n{separator}{body}",
            span_source=parts,
        )

    def _skip_in_walk(self, *, node: NodeItem, root: NodeItem | None) -> bool:
        if self._nested_caption_refs_cache is None:
            self._nested_caption_refs_cache = _nested_caption_refs(self.doc)
        nested = self._nested_caption_refs_cache
        return node.self_ref in nested and (root is None or root.self_ref not in nested)

    def serialize_captions(
        self, item: FloatingItem, **kwargs: Any
    ) -> SerializationResult:
        parts: list[SerializationResult] = []
        for ref in item.captions:
            caption = ref.resolve(self.doc)
            if isinstance(caption, TextItem) and len(caption.children) == 1:
                inline = caption.children[0].resolve(self.doc)
                if isinstance(inline, InlineGroup):
                    parts.append(self.serialize(item=inline, **kwargs))
                    continue
            if isinstance(caption, TextItem):
                parts.append(
                    create_ser_result(
                        text=self.post_process(text=caption.text), span_source=caption
                    )
                )
        return create_ser_result(
            text=self.params.caption_delim.join(part.text for part in parts),
            span_source=parts,
        )


class SourceDocLangSerializer(DocLangDocSerializer):
    """Write rich floating captions until docling-core handles their children."""

    _caption_refs_cache: tuple[set[str], set[str]] | None = PrivateAttr(default=None)

    def _satellite_refs(self) -> tuple[set[str], set[str]]:
        if self._caption_refs_cache is None:
            satellites, nested = super()._satellite_refs()
            satellites = set(satellites)
            nested = set(nested)
            for host, _ in self.doc.iterate_items(
                with_groups=True,
                traverse_pictures=True,
                included_content_layers=set(ContentLayer),
            ):
                if not isinstance(host, FloatingItem):
                    continue
                for ref in host.captions[:1]:
                    caption = ref.resolve(self.doc)
                    if not isinstance(caption, TextItem):
                        continue
                    satellites.add(caption.self_ref)
            nested.update(_nested_caption_refs(self.doc))
            self._caption_refs_cache = satellites, nested
        return self._caption_refs_cache

    def serialize_captions(
        self, item: FloatingItem, **kwargs: Any
    ) -> SerializationResult:
        if len(item.captions) == 1:
            caption = item.captions[0].resolve(self.doc)
            if isinstance(caption, TextItem) and len(caption.children) == 1:
                inline = caption.children[0].resolve(self.doc)
                if isinstance(inline, InlineGroup):
                    part = self.serialize(item=inline, **kwargs)
                    return create_ser_result(
                        text=f"<caption>{part.text}</caption>", span_source=[part]
                    )
        return super().serialize_captions(item=item, **kwargs)
