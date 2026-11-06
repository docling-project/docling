# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Keep Markdown pictures and formulas through the internal HTML round trip."""

import html
from copy import copy
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.html import (
    HTMLDocSerializer,
    HTMLInlineSerializer,
    HTMLParams,
    HTMLPictureSerializer,
    HTMLTextSerializer,
)
from docling_core.types.doc import (
    DoclingDocument,
    FormulaItem,
    ImageRef,
    InlineGroup,
    ListItem,
    PictureItem,
    RefItem,
    SectionHeaderItem,
    TextItem,
    TitleItem,
)
from docling_core.types.doc.common.constants import DOCUMENT_TOKENS_EXPORT_LABELS
from docling_core.types.doc.common.content_layer import DEFAULT_CONTENT_LAYERS
from pydantic import Field
from typing_extensions import override

from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.backend_options import HTMLBackendOptions
from docling.datamodel.document import InputDocument

if TYPE_CHECKING:
    from bs4 import Tag


@dataclass(frozen=True)
class _FormulaReference:
    formula: FormulaItem
    inline: bool


@dataclass(frozen=True)
class MarkdownHTMLPayload:
    text: str
    picture_refs: dict[str, ImageRef | None]
    formula_refs: dict[str, _FormulaReference]


class _MarkdownTextSerializer(HTMLTextSerializer):
    reference_prefix: str
    formula_refs: dict[str, _FormulaReference] = Field(default_factory=dict)

    @override
    def serialize(
        self,
        *,
        item: TextItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        is_inline_scope: bool = False,
        **kwargs: Any,
    ) -> SerializationResult:
        if not isinstance(item, FormulaItem):
            return super().serialize(
                item=item,
                doc_serializer=doc_serializer,
                doc=doc,
                is_inline_scope=is_inline_scope,
                **kwargs,
            )
        reference = self.reference_prefix + item.self_ref + ":"
        self.formula_refs[reference] = _FormulaReference(item, is_inline_scope)
        escaped = html.escape(reference, quote=True)
        # Distinct links keep adjacent formulas/code from merging in the HTML
        # text buffer. These document-local URNs are removed after parsing.
        text = f'<code><a href="{escaped}">{escaped}</a></code>'
        if not is_inline_scope:
            text = f"<pre>{text}</pre>"
        return create_ser_result(text=text, span_source=item)


class _MarkdownPictureSerializer(HTMLPictureSerializer):
    def __init__(self, reference_prefix: str) -> None:
        self._reference_prefix = reference_prefix

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        # The HTML backend must see an img even when fetching was disabled or
        # failed. Its source is a document-local key, not a new fetch request.
        reference = html.escape(self._reference_prefix + item.self_ref, quote=True)
        parts = [
            doc_serializer.serialize_captions(item=item, tag="figcaption", **kwargs),
            create_ser_result(text=f'<img src="{reference}">', span_source=item),
        ]
        text = "".join(part.text for part in parts)
        return create_ser_result(text=f"<figure>{text}</figure>", span_source=parts)


class _MarkdownInlineSerializer(HTMLInlineSerializer):
    @override
    def serialize(
        self,
        *,
        item: InlineGroup,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        result = super().serialize(
            item=item, doc_serializer=doc_serializer, doc=doc, **kwargs
        )
        if (
            item.parent is not None
            and not isinstance(item.parent.resolve(doc), TextItem | InlineGroup)
            and any(isinstance(ref.resolve(doc), FormulaItem) for ref in item.children)
        ):
            return create_ser_result(text=f"<p>{result.text}</p>", span_source=[result])
        return result


class MarkdownHTMLDocumentBackend(HTMLDocumentBackend):
    def __init__(
        self,
        in_doc: InputDocument,
        *,
        path_or_stream: BytesIO | Path,
        options: HTMLBackendOptions | None,
        picture_refs: dict[str, ImageRef | None] | None = None,
        formula_refs: dict[str, _FormulaReference] | None = None,
    ) -> None:
        # InputDocument constructs a backend to inspect it before conversion;
        # that instance does not yet need the already parsed Markdown pictures.
        super().__init__(in_doc=in_doc, path_or_stream=path_or_stream, options=options)
        self._picture_refs = picture_refs if picture_refs is not None else {}
        self._formula_refs = formula_refs if formula_refs is not None else {}

    @override
    def _handle_heading(self, tag: "Tag", doc: DoclingDocument) -> list[RefItem]:
        if not any(key in tag.get_text() for key in self._formula_refs):
            return super()._handle_heading(tag, doc)
        # The general HTML heading path flattens inline content. Let it create
        # the heading/section, then walk the original rich content under it.
        plain_tag = copy(tag)
        plain_tag.clear()
        plain_tag.append(tag.get_text())
        refs = super()._handle_heading(plain_tag, doc)
        heading = refs[0].resolve(doc)
        if isinstance(heading, TitleItem | SectionHeaderItem):
            heading.text = heading.orig = ""
            heading.formatting = None
            heading.hyperlink = None
            self._walk(tag, doc)
            # Core's rich heading owns one InlineGroup. Subsequent blocks stay
            # in the containing section rather than becoming heading text.
            self.parents[self.level] = (
                heading.parent.resolve(doc) if heading.parent is not None else None
            )
        return refs

    @override
    def convert(self) -> DoclingDocument:
        doc = super().convert()
        for index, item in enumerate(list(doc.texts)):
            reference = self._formula_refs.get(item.text)
            if reference is None:
                continue
            if isinstance(item, TitleItem | SectionHeaderItem | ListItem):
                # A formula-only heading/list item may have been folded into
                # its owner's text by the HTML backend. Keep that owner.
                item.text = item.orig = ""
                item.hyperlink = None
                item.formatting = None
                group = doc.add_inline_group(parent=item)
                item = doc.add_formula(text="", parent=group)
                index = len(doc.texts) - 1
            elif reference.inline and item.parent is not None:
                parent = item.parent.resolve(doc)
                if not isinstance(parent, InlineGroup):
                    # HTML can collapse a one-run paragraph. Restore its inline
                    # scope at the same position in the parsed reading order.
                    position = parent.children.index(item.get_ref())
                    group = doc.add_inline_group(parent=parent)
                    parent.children.remove(group.get_ref())
                    parent.children[position] = group.get_ref()
                    group.children.append(item.get_ref())
                    item.parent = group.get_ref()
            doc.texts[index] = reference.formula.model_copy(
                deep=True,
                update={
                    "self_ref": item.self_ref,
                    "parent": item.parent,
                    "children": item.children,
                },
            )
        for table in doc.tables:
            for cell in table.data.table_cells:
                for key, reference in self._formula_refs.items():
                    cell.text = cell.text.replace(key, reference.formula.orig)
        return doc

    @override
    def _create_image_ref(self, src_url: str) -> ImageRef | None:
        if src_url in self._picture_refs:
            return self._picture_refs[src_url]
        # Images in the original HTML retain the normal loader and its guards.
        return super()._create_image_ref(src_url)


def serialize_markdown_for_html(
    doc: DoclingDocument, *, source_hash: str
) -> MarkdownHTMLPayload:
    # Include the source hash so original HTML cannot collide with a reference
    # merely by using the same picture index. The namespace is deterministic.
    prefix = f"data:application/x-docling-md-picture,{source_hash}/"
    picture_refs = {
        prefix + picture.self_ref: picture.image for picture in doc.pictures
    }
    text_serializer = _MarkdownTextSerializer(
        reference_prefix=f"urn:docling:md-formula:{source_hash}:"
    )
    serializer = HTMLDocSerializer(
        doc=doc,
        params=HTMLParams(
            labels=DOCUMENT_TOKENS_EXPORT_LABELS,
            layers=DEFAULT_CONTENT_LAYERS,
        ),
        picture_serializer=_MarkdownPictureSerializer(prefix),
        text_serializer=text_serializer,
        inline_serializer=_MarkdownInlineSerializer(),
    )
    return MarkdownHTMLPayload(
        serializer.serialize().text, picture_refs, text_serializer.formula_refs
    )
