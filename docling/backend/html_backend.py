import base64
import logging
import os
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Final, Optional, Union, cast
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, PageElement, Tag
from bs4.element import PreformattedString
from docling_core.types.doc import (
    DocItem,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupItem,
    GroupLabel,
    PictureItem,
    RefItem,
    RichTableCell,
    TableCell,
    TableData,
    TableItem,
    TextItem,
)
from docling_core.types.doc.document import ContentLayer, Formatting, ImageRef, Script
from PIL import Image, UnidentifiedImageError
from pydantic import AnyUrl, BaseModel, ValidationError
from typing_extensions import override

from docling.backend.abstract_backend import (
    DeclarativeDocumentBackend,
)
from docling.datamodel.backend_options import HTMLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import OperationNotAllowed

_log = logging.getLogger(__name__)

DEFAULT_IMAGE_WIDTH = 128
DEFAULT_IMAGE_HEIGHT = 128

# Tags that initiate distinct Docling items
_BLOCK_TAGS: Final = {
    "address",
    "details",
    "figure",
    "footer",
    "img",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "ol",
    "p",
    "pre",
    "summary",
    "table",
    "ul",
}

# Block-level elements that should not appear inside <p>
_PARA_BREAKERS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "div",
    "dl",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "main",
    "nav",
    "ol",
    "ul",
    "li",
    "p",  # <p> inside <p> also forces closing
    "pre",
    "section",
    "table",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "td",
}

_CODE_TAG_SET: Final = {"code", "kbd", "samp"}

_FORMAT_TAG_MAP: Final = {
    "b": {"bold": True},
    "strong": {"bold": True},
    "i": {"italic": True},
    "em": {"italic": True},
    "var": {"italic": True},
    # "mark",
    # "small",
    "s": {"strikethrough": True},
    "del": {"strikethrough": True},
    "u": {"underline": True},
    "ins": {"underline": True},
    "sub": {"script": Script.SUB},
    "sup": {"script": Script.SUPER},
    **{k: {} for k in _CODE_TAG_SET},
}


class _Context(BaseModel):
    list_ordered_flag_by_ref: dict[str, bool] = {}
    list_start_by_ref: dict[str, int] = {}


class AnnotatedText(BaseModel):
    text: str
    hyperlink: Union[AnyUrl, Path, None] = None
    formatting: Union[Formatting, None] = None
    code: bool = False


class AnnotatedTextList(list):
    def to_single_text_element(self) -> AnnotatedText:
        current_h = None
        current_text = ""
        current_f = None
        current_code = False
        for at in self:
            t = at.text
            h = at.hyperlink
            f = at.formatting
            c = at.code
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
            current_code = c if c else current_code

        return AnnotatedText(
            text=current_text.strip(),
            hyperlink=current_h,
            formatting=current_f,
            code=current_code,
        )

    def simplify_text_elements(self) -> "AnnotatedTextList":
        simplified = AnnotatedTextList()
        if not self:
            return self
        text = self[0].text
        hyperlink = self[0].hyperlink
        formatting = self[0].formatting
        code = self[0].code
        last_elm = text
        for i in range(1, len(self)):
            if (
                hyperlink == self[i].hyperlink
                and formatting == self[i].formatting
                and code == self[i].code
            ):
                sep = " "
                if not self[i].text.strip() or not last_elm.strip():
                    sep = ""
                text += sep + self[i].text
                last_elm = self[i].text
            else:
                simplified.append(
                    AnnotatedText(
                        text=text, hyperlink=hyperlink, formatting=formatting, code=code
                    )
                )
                text = self[i].text
                last_elm = text
                hyperlink = self[i].hyperlink
                formatting = self[i].formatting
                code = self[i].code
        if text:
            simplified.append(
                AnnotatedText(
                    text=text, hyperlink=hyperlink, formatting=formatting, code=code
                )
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
        options: HTMLBackendOptions = HTMLBackendOptions(),
    ):
        super().__init__(in_doc, path_or_stream, options)
        self.options: HTMLBackendOptions
        self.soup: Optional[BeautifulSoup] = None
        self.path_or_stream: Union[BytesIO, Path] = path_or_stream
        self.base_path: Optional[str] = str(options.source_uri)

        # Initialize the parents for the hierarchy
        self.max_levels = 10
        self.level = 0
        self.parents: dict[int, Optional[Union[DocItem, GroupItem]]] = {}
        self.ctx = _Context()
        for i in range(self.max_levels):
            self.parents[i] = None
        self.hyperlink: Union[AnyUrl, Path, None] = None
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
        if title and self.options.add_title:
            title_text = title.get_text(separator=" ", strip=True)
            title_clean = HTMLDocumentBackend._clean_unicode(title_text)
            doc.add_title(
                text=title_clean,
                orig=title_text,
                content_layer=ContentLayer.FURNITURE,
            )
        # remove script and style tags
        for tag in self.soup(["script", "noscript", "style"]):
            tag.decompose()
        # remove any hidden tag
        for tag in self.soup(hidden=True):
            tag.decompose()
        # fix flow content that is not permitted inside <p>
        HTMLDocumentBackend._fix_invalid_paragraph_structure(self.soup)

        content = self.soup.body or self.soup
        # normalize <br> tags
        for br in content("br"):
            br.replace_with(NavigableString("\n"))
        # set default content layer

        # Furniture before the first heading rule, except for headers in tables
        header = None
        # Find all headers first
        all_headers = content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        # Keep only those that do NOT have a <table> in a parent chain
        clean_headers = [h for h in all_headers if not h.find_parent("table")]
        # Pick the first header from the remaining
        if len(clean_headers):
            header = clean_headers[0]
        # Set starting content layer
        self.content_layer = (
            ContentLayer.BODY
            if (not self.options.infer_furniture) or (header is None)
            else ContentLayer.FURNITURE
        )
        # reset context
        self.ctx = _Context()
        self._walk(content, doc)
        return doc

    @staticmethod
    def _fix_invalid_paragraph_structure(soup: BeautifulSoup) -> None:
        """Rewrite <p> elements that contain block-level breakers.

        This function emulates browser logic when other block-level elements
        are found inside a <p> element.
        When a <p> is open and a block-level breaker (e.g., h1-h6, div, table)
        appears, automatically close the <p>, emit it, then emit the breaker,
        and if needed open a new <p> for trailing text.

        Args:
            soup: The HTML document. The DOM may be rewritten.
        """

        def _start_para():
            nonlocal current_p
            if current_p is None:
                current_p = soup.new_tag("p")
                new_nodes.append(current_p)

        def _flush_para_if_empty():
            nonlocal current_p
            if current_p is not None and not current_p.get_text(strip=True):
                # remove empty paragraph placeholder
                if current_p in new_nodes:
                    new_nodes.remove(current_p)
            current_p = None

        paragraphs = soup.select(f"p:has({','.join(tag for tag in _PARA_BREAKERS)})")

        for p in paragraphs:
            parent = p.parent
            if parent is None:
                continue

            new_nodes = []
            current_p = None

            for node in list(p.contents):
                if isinstance(node, NavigableString):
                    text = str(node)
                    node.extract()
                    if text.strip():
                        _start_para()
                        if current_p is not None:
                            current_p.append(NavigableString(text))
                    # skip whitespace-only text
                    continue

                if isinstance(node, Tag):
                    node.extract()

                    if node.name in _PARA_BREAKERS:
                        _flush_para_if_empty()
                        new_nodes.append(node)
                        continue
                    else:
                        _start_para()
                        if current_p is not None:
                            current_p.append(node)
                        continue

            _flush_para_if_empty()

            siblings = list(parent.children)
            try:
                idx = siblings.index(p)
            except ValueError:
                # p might have been removed
                continue

            p.extract()
            for n in reversed(new_nodes):
                parent.insert(idx, n)

    @staticmethod
    def _is_remote_url(value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https", "ftp", "s3", "gs"}

    def _resolve_relative_path(self, loc: str) -> str:
        abs_loc = loc

        if self.base_path:
            if loc.startswith("//"):
                # Protocol-relative URL - default to https
                abs_loc = "https:" + loc
            elif not loc.startswith(("http://", "https://", "data:", "file://")):
                if HTMLDocumentBackend._is_remote_url(self.base_path):  # remote fetch
                    abs_loc = urljoin(self.base_path, loc)
                elif self.base_path:  # local fetch
                    # For local files, resolve relative to the HTML file location
                    abs_loc = str(Path(self.base_path).parent / loc)

        _log.debug(f"Resolved location {loc} to {abs_loc}")
        return abs_loc

    @staticmethod
    def group_cell_elements(
        group_name: str,
        doc: DoclingDocument,
        provs_in_cell: list[RefItem],
        docling_table: TableItem,
    ) -> RefItem:
        group_element = doc.add_group(
            label=GroupLabel.UNSPECIFIED,
            name=group_name,
            parent=docling_table,
        )
        for prov in provs_in_cell:
            group_element.children.append(prov)
            pr_item = prov.resolve(doc)
            item_parent = pr_item.parent.resolve(doc)
            if pr_item.get_ref() in item_parent.children:
                item_parent.children.remove(pr_item.get_ref())
            pr_item.parent = group_element.get_ref()
        ref_for_rich_cell = group_element.get_ref()
        return ref_for_rich_cell

    @staticmethod
    def process_rich_table_cells(
        provs_in_cell: list[RefItem],
        group_name: str,
        doc: DoclingDocument,
        docling_table: TableItem,
    ) -> tuple[bool, Union[RefItem, None]]:
        rich_table_cell = False
        ref_for_rich_cell = None
        if len(provs_in_cell) >= 1:
            # Cell rich cell has multiple elements, we need to group them
            rich_table_cell = True
            ref_for_rich_cell = HTMLDocumentBackend.group_cell_elements(
                group_name, doc, provs_in_cell, docling_table
            )

        return rich_table_cell, ref_for_rich_cell

    def _is_rich_table_cell(self, table_cell: Tag) -> bool:
        """Determine whether an table cell should be parsed as a Docling RichTableCell.

        A table cell can hold rich content and be parsed with a Docling RichTableCell.
        However, this requires walking through the content elements and creating
        Docling node items. If the cell holds only plain text, the parsing is simpler
        and using a TableCell is prefered.

        Args:
            table_cell: The HTML tag representing a table cell.

        Returns:
            Whether the cell should be parsed as RichTableCell.
        """
        is_rich: bool = True

        children = table_cell.find_all(recursive=True)  # all descendants of type Tag
        if not children:
            content = [
                item
                for item in table_cell.contents
                if isinstance(item, NavigableString)
            ]
            is_rich = len(content) > 1
        else:
            annotations = self._extract_text_and_hyperlink_recursively(
                table_cell, find_parent_annotation=True
            )
            if not annotations:
                is_rich = bool(item for item in children if item.name == "img")
            elif len(annotations) == 1:
                anno: AnnotatedText = annotations[0]
                is_rich = bool(anno.formatting) or bool(anno.hyperlink) or anno.code

        return is_rich

    def parse_table_data(
        self,
        element: Tag,
        doc: DoclingDocument,
        docling_table: TableItem,
        num_rows: int,
        num_cols: int,
    ) -> Optional[TableData]:
        for t in cast(list[Tag], element.find_all(["thead", "tbody"], recursive=False)):
            t.unwrap()

        _log.debug(f"The table has {num_rows} rows and {num_cols} cols.")
        grid: list = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=[])

        # Iterate over the rows in the table
        start_row_span = 0
        row_idx = -1

        # We don't want this recursive to support nested tables
        for row in element("tr", recursive=False):
            if not isinstance(row, Tag):
                continue
            # For each row, find all the column cells (both <td> and <th>)
            # We don't want this recursive to support nested tables
            cells = row(["td", "th"], recursive=False)
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

                provs_in_cell: list[RefItem] = []
                rich_table_cell = self._is_rich_table_cell(html_cell)
                if rich_table_cell:
                    # Parse table cell sub-tree for Rich Cells content:
                    with self._use_table_cell_context():
                        provs_in_cell = self._walk(html_cell, doc)

                    group_name = f"rich_cell_group_{len(doc.tables)}_{col_idx}_{start_row_span + row_idx}"
                    rich_table_cell, ref_for_rich_cell = (
                        HTMLDocumentBackend.process_rich_table_cells(
                            provs_in_cell, group_name, doc, docling_table
                        )
                    )

                # Extracting text
                text = HTMLDocumentBackend._clean_unicode(
                    self.get_text(html_cell).strip()
                )
                col_span, row_span = self._get_cell_spans(html_cell)
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

                if rich_table_cell:
                    rich_cell = RichTableCell(
                        text=text,
                        row_span=row_span,
                        col_span=col_span,
                        start_row_offset_idx=start_row_span + row_idx,
                        end_row_offset_idx=start_row_span + row_idx + row_span,
                        start_col_offset_idx=col_idx,
                        end_col_offset_idx=col_idx + col_span,
                        column_header=col_header,
                        row_header=((not col_header) and html_cell.name == "th"),
                        ref=ref_for_rich_cell,  # points to an artificial group around children
                    )
                    doc.add_table_cell(table_item=docling_table, cell=rich_cell)
                else:
                    simple_cell = TableCell(
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
                    doc.add_table_cell(table_item=docling_table, cell=simple_cell)
        return data

    def _walk(self, element: Tag, doc: DoclingDocument) -> list[RefItem]:
        """Parse an XML tag by recursively walking its content.

        While walking, the method buffers inline text across tags like <b> or <span>,
        emitting text nodes only at block boundaries.

        Args:
            element: The XML tag to parse.
            doc: The Docling document to be updated with the parsed content.
        """
        added_refs: list[RefItem] = []
        buffer: AnnotatedTextList = AnnotatedTextList()

        def _flush_buffer() -> None:
            if not buffer:
                return
            annotated_text_list: AnnotatedTextList = buffer.simplify_text_elements()
            parts = annotated_text_list.split_by_newline()
            buffer.clear()

            if not "".join([el.text for el in annotated_text_list]):
                return

            for annotated_text_list in parts:
                with self._use_inline_group(annotated_text_list, doc):
                    for annotated_text in annotated_text_list:
                        if annotated_text.text.strip():
                            seg_clean = HTMLDocumentBackend._clean_unicode(
                                annotated_text.text.strip()
                            )
                            if annotated_text.code:
                                docling_code2 = doc.add_code(
                                    parent=self.parents[self.level],
                                    text=seg_clean,
                                    content_layer=self.content_layer,
                                    formatting=annotated_text.formatting,
                                    hyperlink=annotated_text.hyperlink,
                                )
                                added_refs.append(docling_code2.get_ref())
                            else:
                                docling_text2 = doc.add_text(
                                    parent=self.parents[self.level],
                                    label=DocItemLabel.TEXT,
                                    text=seg_clean,
                                    content_layer=self.content_layer,
                                    formatting=annotated_text.formatting,
                                    hyperlink=annotated_text.hyperlink,
                                )
                                added_refs.append(docling_text2.get_ref())

        for node in element.contents:
            if isinstance(node, Tag):
                name = node.name.lower()
                if name == "img":
                    _flush_buffer()
                    im_ref3 = self._emit_image(node, doc)
                    if im_ref3:
                        added_refs.append(im_ref3)
                elif name in _FORMAT_TAG_MAP:
                    _flush_buffer()
                    with self._use_format([name]):
                        wk = self._walk(node, doc)
                        added_refs.extend(wk)
                elif name == "a":
                    with self._use_hyperlink(node):
                        wk2 = self._walk(node, doc)
                        added_refs.extend(wk2)
                elif name in _BLOCK_TAGS:
                    _flush_buffer()
                    blk = self._handle_block(node, doc)
                    added_refs.extend(blk)
                elif node.find(_BLOCK_TAGS):
                    _flush_buffer()
                    wk3 = self._walk(node, doc)
                    added_refs.extend(wk3)
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
                    _flush_buffer()
                else:
                    buffer.extend(
                        self._extract_text_and_hyperlink_recursively(
                            node, find_parent_annotation=True, keep_newlines=True
                        )
                    )

        _flush_buffer()
        return added_refs

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
                    with self._use_format(format_tags):
                        with self._use_hyperlink(this_parent):
                            return self._extract_text_and_hyperlink_recursively(
                                item, ignore_list
                            )
                this_parent = this_parent.parent

        if isinstance(item, PreformattedString):
            return AnnotatedTextList()

        if isinstance(item, NavigableString):
            text = item.strip()
            code = any(code_tag in self.format_tags for code_tag in _CODE_TAG_SET)
            if text:
                return AnnotatedTextList(
                    [
                        AnnotatedText(
                            text=text,
                            hyperlink=self.hyperlink,
                            formatting=self._formatting,
                            code=code,
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
                            code=code,
                        )
                    ]
                )
            return AnnotatedTextList()

        tag = cast(Tag, item)
        if not ignore_list or (tag.name not in ["ul", "ol"]):
            for child in tag:
                if isinstance(child, Tag) and child.name in _FORMAT_TAG_MAP:
                    with self._use_format([child.name]):
                        result.extend(
                            self._extract_text_and_hyperlink_recursively(
                                child, ignore_list, keep_newlines=keep_newlines
                            )
                        )
                elif isinstance(child, Tag) and child.name == "a":
                    with self._use_hyperlink(child):
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
    def _use_hyperlink(self, tag: Tag):
        old_hyperlink: Union[AnyUrl, Path, None] = None
        new_hyperlink: Union[AnyUrl, Path, None] = None
        this_href = tag.get("href")
        if this_href is None:
            yield None
        else:
            if isinstance(this_href, str) and this_href:
                old_hyperlink = self.hyperlink
                this_href = self._resolve_relative_path(this_href)
                # ugly fix for relative links since pydantic does not support them.
                try:
                    new_hyperlink = AnyUrl(this_href)
                except ValidationError:
                    new_hyperlink = Path(this_href)
                self.hyperlink = new_hyperlink
            try:
                yield None
            finally:
                if new_hyperlink:
                    self.hyperlink = old_hyperlink

    @contextmanager
    def _use_format(self, tags: list[str]):
        if not tags:
            yield None
        else:
            self.format_tags.extend(tags)
            try:
                yield None
            finally:
                self.format_tags = self.format_tags[: -len(tags)]

    @contextmanager
    def _use_inline_group(
        self, annotated_text_list: AnnotatedTextList, doc: DoclingDocument
    ):
        """Create an inline group for annotated texts.

        Checks if annotated_text_list has more than one item and if so creates an inline
        group in which the text elements can then be generated. While the context manager
        is active the inline group is set as the current parent.

        Args:
            annotated_text_list (AnnotatedTextList): Annotated text
            doc (DoclingDocument): Currently used document
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

    @contextmanager
    def _use_details(self, tag: Tag, doc: DoclingDocument):
        """Create a group with the content of a details tag.

        While the context manager is active, the hierarchy level is set one
        level higher as the cuurent parent.

        Args:
            tag: The details tag.
            doc: Currently used document.
        """
        self.parents[self.level + 1] = doc.add_group(
            name=tag.name,
            label=GroupLabel.SECTION,
            parent=self.parents[self.level],
            content_layer=self.content_layer,
        )
        self.level += 1
        try:
            yield None
        finally:
            self.parents[self.level + 1] = None
            self.level -= 1

    @contextmanager
    def _use_footer(self, tag: Tag, doc: DoclingDocument):
        """Create a group with a footer.

        Create a group with the content of a footer tag. While the context manager
        is active, the hierarchy level is set one level higher as the cuurent parent.

        Args:
            tag: The footer tag.
            doc: Currently used document.
        """
        current_layer = self.content_layer
        self.content_layer = ContentLayer.FURNITURE
        self.parents[self.level + 1] = doc.add_group(
            name=tag.name,
            label=GroupLabel.SECTION,
            parent=self.parents[self.level],
            content_layer=self.content_layer,
        )
        self.level += 1
        try:
            yield None
        finally:
            self.parents[self.level + 1] = None
            self.level -= 1
            self.content_layer = current_layer

    @contextmanager
    def _use_table_cell_context(self):
        """Preserve the hierarchy level and parents during table cell processing.

        While the context manager is active, the hierarchy level and parents can be modified.
        When exiting, the original level and parents are restored.
        """
        original_level = self.level
        original_parents = self.parents.copy()
        try:
            yield
        finally:
            self.level = original_level
            self.parents = original_parents

    def _handle_heading(self, tag: Tag, doc: DoclingDocument) -> list[RefItem]:
        added_ref = []
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
            p1 = self.parents[self.level + 1]
            if p1 is not None:
                added_ref = [p1.get_ref()]
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
            p2 = self.parents[self.level + 1]
            if p2 is not None:
                added_ref = [p2.get_ref()]
        self.level += 1
        for img_tag in tag("img"):
            if isinstance(img_tag, Tag):
                im_ref = self._emit_image(img_tag, doc)
                if im_ref:
                    added_ref.append(im_ref)
        return added_ref

    def _handle_list(self, tag: Tag, doc: DoclingDocument) -> RefItem:
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
                        with self._use_inline_group(min_parts, doc):
                            for annotated_text in min_parts:
                                li_text = re.sub(
                                    r"\s+|\n+", " ", annotated_text.text
                                ).strip()
                                li_clean = HTMLDocumentBackend._clean_unicode(li_text)
                                if annotated_text.code:
                                    doc.add_code(
                                        parent=self.parents[self.level],
                                        text=li_clean,
                                        content_layer=self.content_layer,
                                        formatting=annotated_text.formatting,
                                        hyperlink=annotated_text.hyperlink,
                                    )
                                else:
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
        return list_group.get_ref()

    @staticmethod
    def get_html_table_row_col(tag: Tag) -> tuple[int, int]:
        for t in cast(list[Tag], tag.find_all(["thead", "tbody"], recursive=False)):
            t.unwrap()
        # Find the number of rows and columns (taking into account spans)
        num_rows: int = 0
        num_cols: int = 0
        for row in tag("tr", recursive=False):
            col_count = 0
            is_row_header = True
            if not isinstance(row, Tag):
                continue
            for cell in row(["td", "th"], recursive=False):
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
        return num_rows, num_cols

    def _handle_block(self, tag: Tag, doc: DoclingDocument) -> list[RefItem]:
        added_refs = []
        tag_name = tag.name.lower()

        if tag_name == "figure":
            img_tag = tag.find("img")
            if isinstance(img_tag, Tag):
                im_ref = self._emit_image(img_tag, doc)
                if im_ref is not None:
                    added_refs.append(im_ref)

        elif tag_name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            heading_refs = self._handle_heading(tag, doc)
            added_refs.extend(heading_refs)

        elif tag_name in {"ul", "ol"}:
            list_ref = self._handle_list(tag, doc)
            added_refs.append(list_ref)

        elif tag_name in {"p", "address", "summary"}:
            text_list = self._extract_text_and_hyperlink_recursively(
                tag, find_parent_annotation=True
            )
            annotated_texts: AnnotatedTextList = text_list.simplify_text_elements()
            for part in annotated_texts.split_by_newline():
                with self._use_inline_group(part, doc):
                    for annotated_text in part:
                        if seg := annotated_text.text.strip():
                            seg_clean = HTMLDocumentBackend._clean_unicode(seg)
                            if annotated_text.code:
                                docling_code = doc.add_code(
                                    parent=self.parents[self.level],
                                    text=seg_clean,
                                    content_layer=self.content_layer,
                                    formatting=annotated_text.formatting,
                                    hyperlink=annotated_text.hyperlink,
                                )
                                added_refs.append(docling_code.get_ref())
                            else:
                                docling_text = doc.add_text(
                                    parent=self.parents[self.level],
                                    label=DocItemLabel.TEXT,
                                    text=seg_clean,
                                    content_layer=self.content_layer,
                                    formatting=annotated_text.formatting,
                                    hyperlink=annotated_text.hyperlink,
                                )
                                added_refs.append(docling_text.get_ref())

            for img_tag in tag("img"):
                if isinstance(img_tag, Tag):
                    self._emit_image(img_tag, doc)

        elif tag_name == "table":
            num_rows, num_cols = self.get_html_table_row_col(tag)
            data_e = TableData(num_rows=num_rows, num_cols=num_cols)
            docling_table = doc.add_table(
                data=data_e,
                parent=self.parents[self.level],
                content_layer=self.content_layer,
            )
            added_refs.append(docling_table.get_ref())
            self.parse_table_data(tag, doc, docling_table, num_rows, num_cols)

            for img_tag in tag("img"):
                if isinstance(img_tag, Tag):
                    im_ref2 = self._emit_image(tag, doc)
                    if im_ref2 is not None:
                        added_refs.append(im_ref2)

        elif tag_name in {"pre"}:
            # handle monospace code snippets (pre).
            text_list = self._extract_text_and_hyperlink_recursively(
                tag, find_parent_annotation=True, keep_newlines=True
            )
            annotated_texts = text_list.simplify_text_elements()
            with self._use_inline_group(annotated_texts, doc):
                for annotated_text in annotated_texts:
                    text_clean = HTMLDocumentBackend._clean_unicode(
                        annotated_text.text.strip()
                    )
                    docling_code2 = doc.add_code(
                        parent=self.parents[self.level],
                        text=text_clean,
                        content_layer=self.content_layer,
                        formatting=annotated_text.formatting,
                        hyperlink=annotated_text.hyperlink,
                    )
                    added_refs.append(docling_code2.get_ref())

        elif tag_name == "footer":
            with self._use_footer(tag, doc):
                self._walk(tag, doc)

        elif tag_name == "details":
            with self._use_details(tag, doc):
                self._walk(tag, doc)
        return added_refs

    def _emit_image(self, img_tag: Tag, doc: DoclingDocument) -> Optional[RefItem]:
        figure = img_tag.find_parent("figure")
        caption: AnnotatedTextList = AnnotatedTextList()

        parent = self.parents[self.level]

        # check if the figure has a link - this is HACK:
        def get_img_hyperlink(img_tag):
            this_parent = img_tag.parent
            while this_parent is not None:
                if this_parent.name == "a" and this_parent.get("href"):
                    return this_parent.get("href")
                this_parent = this_parent.parent
            return None

        if img_hyperlink := get_img_hyperlink(img_tag):
            img_text = img_tag.get("alt") or ""
            caption.append(AnnotatedText(text=img_text, hyperlink=img_hyperlink))

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

        src_loc: str = self._get_attr_as_string(img_tag, "src")
        if not cast(HTMLBackendOptions, self.options).fetch_images or not src_loc:
            # Do not fetch the image, just add a placeholder
            placeholder: PictureItem = doc.add_picture(
                caption=caption_item,
                parent=parent,
                content_layer=self.content_layer,
            )
            return placeholder.get_ref()

        src_loc = self._resolve_relative_path(src_loc)
        img_ref = self._create_image_ref(src_loc)

        docling_pic = doc.add_picture(
            image=img_ref,
            caption=caption_item,
            parent=parent,
            content_layer=self.content_layer,
        )
        return docling_pic.get_ref()

    def _create_image_ref(self, src_url: str) -> Optional[ImageRef]:
        try:
            img_data = self._load_image_data(src_url)
            if img_data:
                img = Image.open(BytesIO(img_data))
                return ImageRef.from_pil(img, dpi=int(img.info.get("dpi", (72,))[0]))
        except (
            requests.HTTPError,
            ValidationError,
            UnidentifiedImageError,
            OperationNotAllowed,
            TypeError,
            ValueError,
        ) as e:
            warnings.warn(f"Could not process an image from {src_url}: {e}")

        return None

    def _load_image_data(self, src_loc: str) -> Optional[bytes]:
        if src_loc.lower().endswith(".svg"):
            _log.debug(f"Skipping SVG file: {src_loc}")
            return None

        if HTMLDocumentBackend._is_remote_url(src_loc):
            if not self.options.enable_remote_fetch:
                raise OperationNotAllowed(
                    "Fetching remote resources is only allowed when set explicitly. "
                    "Set options.enable_remote_fetch=True."
                )
            response = requests.get(src_loc, stream=True)
            response.raise_for_status()
            return response.content
        elif src_loc.startswith("data:"):
            data = re.sub(r"^data:image/.+;base64,", "", src_loc)
            return base64.b64decode(data)

        if src_loc.startswith("file://"):
            src_loc = src_loc[7:]

        if not self.options.enable_local_fetch:
            raise OperationNotAllowed(
                "Fetching local resources is only allowed when set explicitly. "
                "Set options.enable_local_fetch=True."
            )
        # add check that file exists and can read
        if os.path.isfile(src_loc) and os.access(src_loc, os.R_OK):
            with open(src_loc, "rb") as f:
                return f.read()
        else:
            raise ValueError("File does not exist or it is not readable.")

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
    def _get_attr_as_string(tag: Tag, attr: str, default: str = "") -> str:
        """Get attribute value as string, handling list values."""
        value = tag.get(attr)
        if not value:
            return default

        return value[0] if isinstance(value, list) else value
