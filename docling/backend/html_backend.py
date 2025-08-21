import logging
import re
from contextlib import contextmanager
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Final, Optional, Union, cast
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString, PageElement, Tag
from bs4.element import PreformattedString
from docling_core.types.doc import (
    DocItem,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupItem,
    GroupLabel,
    TableCell,
    TableData,
    TextItem,
)
from docling_core.types.doc.document import ContentLayer, Formatting, Script
from pydantic import AnyUrl, BaseModel, ValidationError
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)

DEFAULT_IMAGE_WIDTH = 128
DEFAULT_IMAGE_HEIGHT = 128

# Tags that initiate distinct Docling items
_BLOCK_TAGS: Final = {
    "address",
    "details",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "pre",
    "code",
    "ul",
    "ol",
    "summary",
    "table",
}

_FORMAT_TAG_MAP: Final = {
    "b": {"bold": True},
    "strong": {"bold": True},
    "i": {"italic": True},
    "em": {"italic": True},
    # "mark",
    # "small",
    "s": {"strikethrough": True},
    "del": {"strikethrough": True},
    "u": {"underline": True},
    "ins": {"underline": True},
    "sub": {"script": Script.SUB},
    "sup": {"script": Script.SUPER},
}


class _Context(BaseModel):
    list_ordered_flag_by_ref: dict[str, bool] = {}
    list_start_by_ref: dict[str, int] = {}


class AnnotatedText(BaseModel):
    text: str
    hyperlink: Union[AnyUrl, Path, None] = None
    formatting: Union[Formatting, None] = None


class AnnotatedTextList(list):
    def to_single_text_element(self) -> AnnotatedText:
        current_h = None
        current_text = ""
        current_f = None
        for at in self:
            t = at.text
            h = at.hyperlink
            f = at.formatting
            current_text += t.strip() + " "
            if f is not None and current_f is None:
                current_f = f
            elif f is not None and current_f is not None and f != current_f:
                _log.warning(
                    f"Clashing formatting: '{f}' and '{current_f}'! Chose '{current_f}'"
                )
            if h is not None and current_h is None:
                current_h = h
            elif h is not None and current_h is not None and h != current_h:
                _log.warning(
                    f"Clashing hyperlinks: '{h}' and '{current_h}'! Chose '{current_h}'"
                )
        return AnnotatedText(
            text=current_text.strip(), hyperlink=current_h, formatting=current_f
        )

    def simplify_text_elements(self) -> "AnnotatedTextList":
        simplified = AnnotatedTextList()
        if not self:
            return self
        text = self[0].text
        hyperlink = self[0].hyperlink
        formatting = self[0].formatting
        last_elm = text
        for i in range(1, len(self)):
            if hyperlink == self[i].hyperlink and formatting == self[i].formatting:
                sep = " "
                if not self[i].text.strip() or not last_elm.strip():
                    sep = ""
                text += sep + self[i].text
                last_elm = self[i].text
            else:
                simplified.append(
                    AnnotatedText(text=text, hyperlink=hyperlink, formatting=formatting)
                )
                text = self[i].text
                last_elm = text
                hyperlink = self[i].hyperlink
                formatting = self[i].formatting
        if text:
            simplified.append(
                AnnotatedText(text=text, hyperlink=hyperlink, formatting=formatting)
            )
        return simplified

    def split_by_newline(self):
        super_list = []
        active_annotated_text_list = AnnotatedTextList()
        for el in self:
            sub_texts = el.text.split("\n")
            if len(sub_texts) == 1:
                active_annotated_text_list.append(el)
            else:
                for text in sub_texts:
                    sub_el = deepcopy(el)
                    sub_el.text = text
                    active_annotated_text_list.append(sub_el)
                    super_list.append(active_annotated_text_list)
                    active_annotated_text_list = AnnotatedTextList()
        if active_annotated_text_list:
            super_list.append(active_annotated_text_list)
        return super_list


class HTMLDocumentBackend(DeclarativeDocumentBackend):
    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
        original_url: Optional[AnyUrl] = None,
    ):
        super().__init__(in_doc, path_or_stream)
        self.soup: Optional[Tag] = None
        self.path_or_stream = path_or_stream

        # Initialize the parents for the hierarchy
        self.max_levels = 10
        self.level = 0
        self.parents: dict[int, Optional[Union[DocItem, GroupItem]]] = {}
        self.ctx = _Context()
        for i in range(self.max_levels):
            self.parents[i] = None
        self.hyperlink = None
        self.original_url = original_url
        self.format_tags: list[str] = []

        try:
            raw = (
                path_or_stream.getvalue()
                if isinstance(path_or_stream, BytesIO)
                else Path(path_or_stream).read_bytes()
            )
            self.soup = BeautifulSoup(raw, "html.parser")
        except Exception as e:
            raise RuntimeError(
                "Could not initialize HTML backend for file with "
                f"hash {self.document_hash}."
            ) from e

    @override
    def is_valid(self) -> bool:
        return self.soup is not None

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return False

    @override
    def unload(self):
        if isinstance(self.path_or_stream, BytesIO):
            self.path_or_stream.close()
        self.path_or_stream = None

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.HTML}

    @override
    def convert(self) -> DoclingDocument:
        _log.debug("Starting HTML conversion...")
        if not self.is_valid():
            raise RuntimeError("Invalid HTML document.")

        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="text/html",
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)

        assert self.soup is not None
        # set the title as furniture, since it is part of the document metadata
        title = self.soup.title
        if title:
            title_text = title.get_text(separator=" ", strip=True)
            title_clean = HTMLDocumentBackend._clean_unicode(title_text)
            doc.add_title(
                text=title_clean,
                orig=title_text,
                content_layer=ContentLayer.FURNITURE,
            )
        # remove scripts/styles
        for tag in self.soup(["script", "style"]):
            tag.decompose()
        content = self.soup.body or self.soup
        # normalize <br> tags
        for br in content("br"):
            br.replace_with(NavigableString("\n"))
        # set default content layer
        headers = content.find(["h1", "h2", "h3", "h4", "h5", "h6"])
        self.content_layer = (
            ContentLayer.BODY if headers is None else ContentLayer.FURNITURE
        )
        # reset context
        self.ctx = _Context()
        self._walk(content, doc)

        return doc

    def _walk(self, element: Tag, doc: DoclingDocument) -> None:
        """Parse an XML tag by recursively walking its content.

        While walking, the method buffers inline text across tags like <b> or <span>,
        emitting text nodes only at block boundaries.

        Args:
            element: The XML tag to parse.
            doc: The Docling document to be updated with the parsed content.
        """
        buffer: AnnotatedTextList = AnnotatedTextList()

        def flush_buffer():
            if not buffer:
                return
            annotated_text_list = buffer.simplify_text_elements()
            parts = annotated_text_list.split_by_newline()
            buffer.clear()

            if not "".join([el.text for el in annotated_text_list]):
                return

            for annotated_text_list in parts:
                with self.use_inline_group(annotated_text_list, doc):
                    for annotated_text in annotated_text_list:
                        if annotated_text.text.strip():
                            seg_clean = HTMLDocumentBackend._clean_unicode(
                                annotated_text.text.strip()
                            )
                            doc.add_text(
                                parent=self.parents[self.level],
                                label=DocItemLabel.TEXT,
                                text=seg_clean,
                                content_layer=self.content_layer,
                                formatting=annotated_text.formatting,
                                hyperlink=annotated_text.hyperlink,
                            )

        for node in element.contents:
            if isinstance(node, Tag):
                name = node.name.lower()
                if name == "img":
                    flush_buffer()
                    self._emit_image(node, doc)
                elif name in _FORMAT_TAG_MAP:
                    with self.use_format([name]):
                        self._walk(node, doc)
                elif name == "a":
                    with self.use_hyperlink(node):
                        self._walk(node, doc)
                elif name in _BLOCK_TAGS:
                    flush_buffer()
                    self._handle_block(node, doc)
                elif node.find(_BLOCK_TAGS):
                    flush_buffer()
                    self._walk(node, doc)
                else:
                    buffer.extend(
                        self._extract_text_and_hyperlink_recursively(
                            node, find_parent_annotation=True, keep_newlines=True
                        )
                    )
            elif isinstance(node, NavigableString) and not isinstance(
                node, PreformattedString
            ):
                if str(node).strip("\n\r") == "":
                    flush_buffer()
                else:
                    buffer.extend(
                        self._extract_text_and_hyperlink_recursively(
                            node, find_parent_annotation=True, keep_newlines=True
                        )
                    )

        flush_buffer()

    @staticmethod
    def _collect_parent_format_tags(item: PageElement) -> list[str]:
        tags = []
        for format_tag in _FORMAT_TAG_MAP:
            this_parent = item.parent
            while this_parent is not None:
                if this_parent.name == format_tag:
                    tags.append(format_tag)
                    break
                this_parent = this_parent.parent
        return tags

    @property
    def _formatting(self):
        kwargs = {}
        for t in self.format_tags:
            kwargs.update(_FORMAT_TAG_MAP[t])
        if not kwargs:
            return None
        return Formatting(**kwargs)

    def _extract_text_and_hyperlink_recursively(
        self,
        item: PageElement,
        ignore_list=False,
        find_parent_annotation=False,
        keep_newlines=False,
    ) -> AnnotatedTextList:
        result: AnnotatedTextList = AnnotatedTextList()

        # If find_parent_annotation, make sure that we keep track of
        # any a- or formatting-tag that has been present in the
        # DOM-parents already.
        if find_parent_annotation:
            format_tags = self._collect_parent_format_tags(item)
            this_parent = item.parent
            while this_parent is not None:
                if this_parent.name == "a" and this_parent.get("href"):
                    with self.use_format(format_tags):
                        with self.use_hyperlink(this_parent):
                            return self._extract_text_and_hyperlink_recursively(
                                item, ignore_list
                            )
                this_parent = this_parent.parent

        if isinstance(item, PreformattedString):
            return AnnotatedTextList()

        if isinstance(item, NavigableString):
            text = item.strip()
            if text:
                return AnnotatedTextList(
                    [
                        AnnotatedText(
                            text=text,
                            hyperlink=self.hyperlink,
                            formatting=self._formatting,
                        )
                    ]
                )
            if keep_newlines and item.strip("\n\r") == "":
                return AnnotatedTextList(
                    [
                        AnnotatedText(
                            text="\n",
                            hyperlink=self.hyperlink,
                            formatting=self._formatting,
                        )
                    ]
                )
            return AnnotatedTextList()

        tag = cast(Tag, item)
        if not ignore_list or (tag.name not in ["ul", "ol"]):
            for child in tag:
                if isinstance(child, Tag) and child.name in _FORMAT_TAG_MAP:
                    with self.use_format([child.name]):
                        result.extend(
                            self._extract_text_and_hyperlink_recursively(
                                child, ignore_list, keep_newlines=keep_newlines
                            )
                        )
                elif isinstance(child, Tag) and child.name == "a":
                    with self.use_hyperlink(child):
                        result.extend(
                            self._extract_text_and_hyperlink_recursively(
                                child, ignore_list, keep_newlines=keep_newlines
                            )
                        )
                else:
                    # Recursively get the child's text content
                    result.extend(
                        self._extract_text_and_hyperlink_recursively(
                            child, ignore_list, keep_newlines=keep_newlines
                        )
                    )
        return result

    @contextmanager
    def use_hyperlink(self, tag):
        this_href = tag.get("href")
        if this_href is None:
            yield None
        else:
            if this_href:
                old_hyperlink = self.hyperlink
                if self.original_url is not None:
                    this_href = urljoin(self.original_url, this_href)
                # ugly fix for relative links since pydantic does not support them.
                try:
                    AnyUrl(this_href)
                except ValidationError:
                    this_href = Path(this_href)
                self.hyperlink = this_href
            try:
                yield None
            finally:
                if this_href:
                    self.hyperlink = old_hyperlink

    @contextmanager
    def use_format(self, tags: list[str]):
        if not tags:
            yield None
        else:
            self.format_tags.extend(tags)
            try:
                yield None
            finally:
                self.format_tags = self.format_tags[: -len(tags)]

    @contextmanager
    def use_inline_group(
        self, annotated_text_list: AnnotatedTextList, doc: DoclingDocument
    ):
        """Create an inline group for annotated texts.

        Checks if annotated_text_list has more than one item and if so creates an inline
        group in which the text elements can then be generated. While the context manager
        is active the inline group is set as the current parent.

        Args:
            annotated_text_list (AnnotatedTextList): Annotated text
            doc (DoclingDocument): Currently used document

        Yields:
            None: _description_
        """
        if len(annotated_text_list) > 1:
            inline_fmt = doc.add_group(
                label=GroupLabel.INLINE,
                parent=self.parents[self.level],
                content_layer=self.content_layer,
            )
            self.parents[self.level + 1] = inline_fmt
            self.level += 1
            try:
                yield None
            finally:
                self.parents[self.level] = None
                self.level -= 1
        else:
            yield None

    def _handle_heading(self, tag: Tag, doc: DoclingDocument) -> None:
        tag_name = tag.name.lower()
        # set default content layer to BODY as soon as we encounter a heading
        self.content_layer = ContentLayer.BODY
        level = int(tag_name[1])
        annotated_text_list = self._extract_text_and_hyperlink_recursively(
            tag, find_parent_annotation=True
        )
        annotated_text = annotated_text_list.to_single_text_element()
        text_clean = HTMLDocumentBackend._clean_unicode(annotated_text.text)
        # the first level is for the title item
        if level == 1:
            for key in self.parents.keys():
                self.parents[key] = None
            self.level = 0
            self.parents[self.level + 1] = doc.add_title(
                text_clean,
                content_layer=self.content_layer,
                formatting=annotated_text.formatting,
                hyperlink=annotated_text.hyperlink,
            )
        # the other levels need to be lowered by 1 if a title was set
        else:
            level -= 1
            if level > self.level:
                # add invisible group
                for i in range(self.level, level):
                    _log.debug(f"Adding invisible group to level {i}")
                    self.parents[i + 1] = doc.add_group(
                        name=f"header-{i + 1}",
                        label=GroupLabel.SECTION,
                        parent=self.parents[i],
                        content_layer=self.content_layer,
                    )
                self.level = level
            elif level < self.level:
                # remove the tail
                for key in self.parents.keys():
                    if key > level + 1:
                        _log.debug(f"Remove the tail of level {key}")
                        self.parents[key] = None
                self.level = level
            self.parents[self.level + 1] = doc.add_heading(
                parent=self.parents[self.level],
                text=text_clean,
                orig=annotated_text.text,
                level=self.level,
                content_layer=self.content_layer,
                formatting=annotated_text.formatting,
                hyperlink=annotated_text.hyperlink,
            )
        self.level += 1
        for img_tag in tag("img"):
            if isinstance(img_tag, Tag):
                self._emit_image(img_tag, doc)

    def _handle_list(self, tag: Tag, doc: DoclingDocument) -> None:
        tag_name = tag.name.lower()
        start: Optional[int] = None
        name: str = ""
        is_ordered = tag_name == "ol"
        if is_ordered:
            start_attr = tag.get("start")
            if isinstance(start_attr, str) and start_attr.isnumeric():
                start = int(start_attr)
            name = "ordered list" + (f" start {start}" if start is not None else "")
        else:
            name = "list"
        # Create the list container
        list_group = doc.add_list_group(
            name=name,
            parent=self.parents[self.level],
            content_layer=self.content_layer,
        )
        self.parents[self.level + 1] = list_group
        self.ctx.list_ordered_flag_by_ref[list_group.self_ref] = is_ordered
        if is_ordered and start is not None:
            self.ctx.list_start_by_ref[list_group.self_ref] = start
        self.level += 1

        # For each top-level <li> in this list
        for li in tag.find_all({"li", "ul", "ol"}, recursive=False):
            if not isinstance(li, Tag):
                continue

            # sub-list items should be indented under main list items, but temporarily
            # addressing invalid HTML (docling-core/issues/357)
            if li.name in {"ul", "ol"}:
                self._handle_block(li, doc)

            else:
                # 1) determine the marker
                if is_ordered and start is not None:
                    marker = f"{start + len(list_group.children)}."
                else:
                    marker = ""

                # 2) extract only the "direct" text from this <li>
                parts = self._extract_text_and_hyperlink_recursively(
                    li, ignore_list=True, find_parent_annotation=True
                )
                min_parts = parts.simplify_text_elements()
                li_text = re.sub(
                    r"\s+|\n+", " ", "".join([el.text for el in min_parts])
                ).strip()

                # 3) add the list item
                if li_text:
                    if len(min_parts) > 1:
                        # create an empty list element in order to hook the inline group onto that one
                        self.parents[self.level + 1] = doc.add_list_item(
                            text="",
                            enumerated=is_ordered,
                            marker=marker,
                            parent=list_group,
                            content_layer=self.content_layer,
                        )
                        self.level += 1
                        with self.use_inline_group(min_parts, doc):
                            for annotated_text in min_parts:
                                li_text = re.sub(
                                    r"\s+|\n+", " ", annotated_text.text
                                ).strip()
                                li_clean = HTMLDocumentBackend._clean_unicode(li_text)
                                doc.add_text(
                                    parent=self.parents[self.level],
                                    label=DocItemLabel.TEXT,
                                    text=li_clean,
                                    content_layer=self.content_layer,
                                    formatting=annotated_text.formatting,
                                    hyperlink=annotated_text.hyperlink,
                                )

                        # 4) recurse into any nested lists, attaching them to this <li> item
                        for sublist in li({"ul", "ol"}, recursive=False):
                            if isinstance(sublist, Tag):
                                self._handle_block(sublist, doc)

                        # now the list element with inline group is not a parent anymore
                        self.parents[self.level] = None
                        self.level -= 1
                    else:
                        annotated_text = min_parts[0]
                        li_text = re.sub(r"\s+|\n+", " ", annotated_text.text).strip()
                        li_clean = HTMLDocumentBackend._clean_unicode(li_text)
                        self.parents[self.level + 1] = doc.add_list_item(
                            text=li_clean,
                            enumerated=is_ordered,
                            marker=marker,
                            orig=li_text,
                            parent=list_group,
                            content_layer=self.content_layer,
                            formatting=annotated_text.formatting,
                            hyperlink=annotated_text.hyperlink,
                        )

                        # 4) recurse into any nested lists, attaching them to this <li> item
                        for sublist in li({"ul", "ol"}, recursive=False):
                            if isinstance(sublist, Tag):
                                self.level += 1
                                self._handle_block(sublist, doc)
                                self.parents[self.level + 1] = None
                                self.level -= 1
                else:
                    for sublist in li({"ul", "ol"}, recursive=False):
                        if isinstance(sublist, Tag):
                            self._handle_block(sublist, doc)

                # 5) extract any images under this <li>
                for img_tag in li("img"):
                    if isinstance(img_tag, Tag):
                        self._emit_image(img_tag, doc)

        self.parents[self.level + 1] = None
        self.level -= 1

    def _handle_block(self, tag: Tag, doc: DoclingDocument) -> None:
        tag_name = tag.name.lower()

        if tag_name == "figure":
            img_tag = tag.find("img")
            if isinstance(img_tag, Tag):
                self._emit_image(img_tag, doc)

        elif tag_name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._handle_heading(tag, doc)

        elif tag_name in {"ul", "ol"}:
            self._handle_list(tag, doc)

        elif tag_name in {"p", "address", "summary"}:
            text_list = self._extract_text_and_hyperlink_recursively(
                tag, find_parent_annotation=True
            )
            annotated_texts = text_list.simplify_text_elements()
            for part in annotated_texts.split_by_newline():
                with self.use_inline_group(part, doc):
                    for annotated_text in part:
                        if seg := annotated_text.text.strip():
                            seg_clean = HTMLDocumentBackend._clean_unicode(seg)
                            doc.add_text(
                                parent=self.parents[self.level],
                                label=DocItemLabel.TEXT,
                                text=seg_clean,
                                content_layer=self.content_layer,
                                formatting=annotated_text.formatting,
                                hyperlink=annotated_text.hyperlink,
                            )

            for img_tag in tag("img"):
                if isinstance(img_tag, Tag):
                    self._emit_image(img_tag, doc)

        elif tag_name == "table":
            data = HTMLDocumentBackend.parse_table_data(tag)
            for img_tag in tag("img"):
                if isinstance(img_tag, Tag):
                    self._emit_image(tag, doc)
            if data is not None:
                doc.add_table(
                    data=data,
                    parent=self.parents[self.level],
                    content_layer=self.content_layer,
                )

        elif tag_name in {"pre", "code"}:
            # handle monospace code snippets (pre).
            text_list = self._extract_text_and_hyperlink_recursively(
                tag, find_parent_annotation=True
            )
            annotated_texts = text_list.simplify_text_elements()
            with self.use_inline_group(annotated_texts, doc):
                for annotated_text in annotated_texts:
                    text_clean = HTMLDocumentBackend._clean_unicode(
                        annotated_text.text.strip()
                    )
                    doc.add_code(
                        parent=self.parents[self.level],
                        text=text_clean,
                        content_layer=self.content_layer,
                        formatting=annotated_text.formatting,
                        hyperlink=annotated_text.hyperlink,
                    )

        elif tag_name in {"details", "footer"}:
            if tag_name == "footer":
                current_layer = self.content_layer
                self.content_layer = ContentLayer.FURNITURE
            self.parents[self.level + 1] = doc.add_group(
                name=tag_name,
                label=GroupLabel.SECTION,
                parent=self.parents[self.level],
                content_layer=self.content_layer,
            )
            self.level += 1
            self._walk(tag, doc)
            self.parents[self.level + 1] = None
            self.level -= 1
            if tag_name == "footer":
                self.content_layer = current_layer

    def _emit_image(self, img_tag: Tag, doc: DoclingDocument) -> None:
        figure = img_tag.find_parent("figure")
        caption: AnnotatedTextList = AnnotatedTextList()

        # check if the figure has a link - this is HACK:
        def get_img_hyperlink(img_tag):
            this_parent = img_tag.parent
            while this_parent is not None:
                if this_parent.name == "a" and this_parent.get("href"):
                    return this_parent.get("href")
                this_parent = this_parent.parent
            return None

        if img_hyperlink := get_img_hyperlink(img_tag):
            caption.append(
                AnnotatedText(text="Image Hyperlink.", hyperlink=img_hyperlink)
            )

        if isinstance(figure, Tag):
            caption_tag = figure.find("figcaption", recursive=False)
            if isinstance(caption_tag, Tag):
                caption = self._extract_text_and_hyperlink_recursively(
                    caption_tag, find_parent_annotation=True
                )
        if not caption and img_tag.get("alt"):
            caption = AnnotatedTextList([AnnotatedText(text=img_tag.get("alt"))])

        caption_anno_text = caption.to_single_text_element()

        caption_item: Optional[TextItem] = None
        if caption_anno_text.text:
            text_clean = HTMLDocumentBackend._clean_unicode(
                caption_anno_text.text.strip()
            )
            caption_item = doc.add_text(
                label=DocItemLabel.CAPTION,
                text=text_clean,
                orig=caption_anno_text.text,
                content_layer=self.content_layer,
                formatting=caption_anno_text.formatting,
                hyperlink=caption_anno_text.hyperlink,
            )

        doc.add_picture(
            caption=caption_item,
            parent=self.parents[self.level],
            content_layer=self.content_layer,
        )

    @staticmethod
    def get_text(item: PageElement) -> str:
        """Concatenate all child strings of a PageElement.

        This method is equivalent to `PageElement.get_text()` but also considers
        certain tags. When called on a <p> or <li> tags, it returns the text with a
        trailing space, otherwise the text is concatenated without separators.
        """

        def _extract_text_recursively(item: PageElement) -> list[str]:
            """Recursively extract text from all child nodes."""
            result: list[str] = []

            if isinstance(item, NavigableString):
                result = [item]
            elif isinstance(item, Tag):
                tag = cast(Tag, item)
                parts: list[str] = []
                for child in tag:
                    parts.extend(_extract_text_recursively(child))
                result.append(
                    "".join(parts) + " " if tag.name in {"p", "li"} else "".join(parts)
                )

            return result

        parts: list[str] = _extract_text_recursively(item)

        return "".join(parts)

    @staticmethod
    def _clean_unicode(text: str) -> str:
        """Replace typical Unicode characters in HTML for text processing.

        Several Unicode characters (e.g., non-printable or formatting) are typically
        found in HTML but are worth replacing to sanitize text and ensure consistency
        in text processing tasks.

        Args:
            text: The original text.

        Returns:
            The sanitized text without typical Unicode characters.
        """
        replacements = {
            "\u00a0": " ",  # non-breaking space
            "\u200b": "",  # zero-width space
            "\u200c": "",  # zero-width non-joiner
            "\u200d": "",  # zero-width joiner
            "\u2010": "-",  # hyphen
            "\u2011": "-",  # non-breaking hyphen
            "\u2012": "-",  # dash
            "\u2013": "-",  # dash
            "\u2014": "-",  # dash
            "\u2015": "-",  # horizontal bar
            "\u2018": "'",  # left single quotation mark
            "\u2019": "'",  # right single quotation mark
            "\u201c": '"',  # left double quotation mark
            "\u201d": '"',  # right double quotation mark
            "\u2026": "...",  # ellipsis
            "\u00ad": "",  # soft hyphen
            "\ufeff": "",  # zero width non-break space
            "\u202f": " ",  # narrow non-break space
            "\u2060": "",  # word joiner
        }
        for raw, clean in replacements.items():
            text = text.replace(raw, clean)

        return text

    @staticmethod
    def _get_cell_spans(cell: Tag) -> tuple[int, int]:
        """Extract colspan and rowspan values from a table cell tag.

        This function retrieves the 'colspan' and 'rowspan' attributes from a given
        table cell tag.
        If the attribute does not exist or it is not numeric, it defaults to 1.
        """
        raw_spans: tuple[str, str] = (
            str(cell.get("colspan", "1")),
            str(cell.get("rowspan", "1")),
        )

        def _extract_num(s: str) -> int:
            if s and s[0].isnumeric():
                match = re.search(r"\d+", s)
                if match:
                    return int(match.group())
            return 1

        int_spans: tuple[int, int] = (
            _extract_num(raw_spans[0]),
            _extract_num(raw_spans[1]),
        )

        return int_spans

    @staticmethod
    def parse_table_data(element: Tag) -> Optional[TableData]:  # noqa: C901
        nested_tables = element.find("table")
        if nested_tables is not None:
            _log.debug("Skipping nested table.")
            return None

        # Find the number of rows and columns (taking into account spans)
        num_rows = 0
        num_cols = 0
        for row in element("tr"):
            col_count = 0
            is_row_header = True
            if not isinstance(row, Tag):
                continue
            for cell in row(["td", "th"]):
                if not isinstance(row, Tag):
                    continue
                cell_tag = cast(Tag, cell)
                col_span, row_span = HTMLDocumentBackend._get_cell_spans(cell_tag)
                col_count += col_span
                if cell_tag.name == "td" or row_span == 1:
                    is_row_header = False
            num_cols = max(num_cols, col_count)
            if not is_row_header:
                num_rows += 1

        _log.debug(f"The table has {num_rows} rows and {num_cols} cols.")

        grid: list = [[None for _ in range(num_cols)] for _ in range(num_rows)]

        data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=[])

        # Iterate over the rows in the table
        start_row_span = 0
        row_idx = -1
        for row in element("tr"):
            if not isinstance(row, Tag):
                continue

            # For each row, find all the column cells (both <td> and <th>)
            cells = row(["td", "th"])

            # Check if cell is in a column header or row header
            col_header = True
            row_header = True
            for html_cell in cells:
                if isinstance(html_cell, Tag):
                    _, row_span = HTMLDocumentBackend._get_cell_spans(html_cell)
                    if html_cell.name == "td":
                        col_header = False
                        row_header = False
                    elif row_span == 1:
                        row_header = False
            if not row_header:
                row_idx += 1
                start_row_span = 0
            else:
                start_row_span += 1

            # Extract the text content of each cell
            col_idx = 0
            for html_cell in cells:
                if not isinstance(html_cell, Tag):
                    continue

                # extract inline formulas
                for formula in html_cell("inline-formula"):
                    math_parts = formula.text.split("$$")
                    if len(math_parts) == 3:
                        math_formula = f"$${math_parts[1]}$$"
                        formula.replace_with(NavigableString(math_formula))

                # TODO: extract content correctly from table-cells with lists
                text = HTMLDocumentBackend.get_text(html_cell).strip()
                col_span, row_span = HTMLDocumentBackend._get_cell_spans(html_cell)
                if row_header:
                    row_span -= 1
                while (
                    col_idx < num_cols
                    and grid[row_idx + start_row_span][col_idx] is not None
                ):
                    col_idx += 1
                for r in range(start_row_span, start_row_span + row_span):
                    for c in range(col_span):
                        if row_idx + r < num_rows and col_idx + c < num_cols:
                            grid[row_idx + r][col_idx + c] = text

                table_cell = TableCell(
                    text=text,
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=start_row_span + row_idx,
                    end_row_offset_idx=start_row_span + row_idx + row_span,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + col_span,
                    column_header=col_header,
                    row_header=((not col_header) and html_cell.name == "th"),
                )
                data.table_cells.append(table_cell)

        return data
