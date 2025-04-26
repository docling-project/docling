import base64
import logging
import re
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    Size,
    TableCell,
    TableData,
)
from docling_core.types.doc.document import ContentLayer, ImageRef
from PIL import Image, UnidentifiedImageError
from pydantic import AnyUrl, HttpUrl, ValidationError
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)

# Tags that initiate distinct Docling items
_BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table"}


class ImageOptions(str, Enum):
    """Image options for HTML backend."""

    NONE = "none"
    INLINE = "inline"
    REFERENCED = "referenced"


class BaseHTMLDocumentBackend(DeclarativeDocumentBackend):
    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
        image_options: Optional[ImageOptions] = ImageOptions.NONE,
    ):
        super().__init__(in_doc, path_or_stream)
        self.image_options = image_options
        self.soup: Optional[Tag] = None
        try:
            raw = (
                path_or_stream.getvalue()
                if isinstance(path_or_stream, BytesIO)
                else Path(path_or_stream).read_bytes()
            )
            self.soup = BeautifulSoup(raw, "html.parser")
        except Exception as e:
            raise RuntimeError(f"Could not initialize HTML backend: {e}")

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
        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="text/html",
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)
        _log.debug("Starting HTML conversion...")
        if not self.is_valid():
            raise RuntimeError("Invalid HTML document.")
        assert self.soup is not None

        # Remove all script/style content
        for tag in self.soup.find_all(["script", "style"]):
            tag.decompose()

        body = self.soup.body or self.soup
        # Normalize <br> tags to newline strings
        for br in body.find_all("br"):
            br.replace_with(NavigableString("\n"))

        # Decide content layer by presence of headers
        headers = body.find(list(_BLOCK_TAGS))
        self.content_layer = (
            ContentLayer.BODY if headers is None else ContentLayer.FURNITURE
        )

        # Walk the body to build the DoclingDocument
        self._walk(body, doc, parent=doc.body)
        return doc

    def _walk(self, element: Tag, doc: DoclingDocument, parent) -> None:
        """
        Recursively walk element.contents, buffering inline text across tags like <b> or <span>,
        emitting text nodes only at block boundaries, and extracting images immediately.
        """
        buffer: list[str] = []

        def flush_buffer():
            if not buffer:
                return
            text = "".join(buffer).strip()
            buffer.clear()
            if not text:
                return
            # Split on newlines for <br>
            for part in text.split("\n"):
                seg = part.strip()
                if seg:
                    doc.add_text(DocItemLabel.TEXT, seg, parent=parent)

        for node in element.contents:
            # Skip scripts/styles
            if isinstance(node, Tag) and node.name.lower() in ("script", "style"):
                continue
            # Immediate image extraction
            if isinstance(node, Tag) and node.name.lower() == "img":
                flush_buffer()
                self._emit_image(node, doc, parent)
                continue
            # Block-level element triggers flush + handle
            if isinstance(node, Tag) and node.name.lower() in _BLOCK_TAGS:
                flush_buffer()
                self._handle_block(node, doc, parent)
            # Inline tag with nested blocks: recurse
            elif isinstance(node, Tag) and node.find(list(_BLOCK_TAGS)):
                flush_buffer()
                self._walk(node, doc, parent)
            # Inline text
            elif isinstance(node, Tag):
                buffer.append(node.get_text())
            elif isinstance(node, NavigableString):
                buffer.append(str(node))

        # Flush any remaining text
        flush_buffer()

    def _handle_block(self, tag: Tag, doc: DoclingDocument, parent) -> None:
        tag_name = tag.name.lower()
        if tag_name == "h1":
            text = tag.get_text(strip=True)
            if text:
                doc.add_title(text, parent=parent)
            for img_tag in tag.find_all("img", recursive=True):
                self._emit_image(img_tag, doc, parent)
        elif tag_name in {"h2", "h3", "h4", "h5", "h6"}:
            level = int(tag_name[1])
            text = tag.get_text(strip=True)
            if text:
                doc.add_heading(text, level=level, parent=parent)
            for img_tag in tag.find_all("img", recursive=True):
                self._emit_image(img_tag, doc, parent)
        elif tag_name == "p":
            for part in tag.get_text().split("\n"):
                seg = part.strip()
                if seg:
                    doc.add_text(DocItemLabel.TEXT, seg, parent=parent)
                for img_tag in tag.find_all("img", recursive=True):
                    self._emit_image(img_tag, doc, parent)
        elif tag_name in {"ul", "ol"}:
            is_ordered = tag_name == "ol"
            group = (
                doc.add_ordered_list(parent=parent)
                if is_ordered
                else doc.add_unordered_list(parent=parent)
            )
            for li in tag.find_all("li", recursive=False):
                li_text = li.get_text(separator=" ", strip=True)
                li_item = doc.add_list_item(
                    text=li_text, enumerated=is_ordered, parent=group
                )
                # Nested lists inside <li>
                for sub in li.find_all(["ul", "ol"], recursive=False):
                    self._handle_block(sub, doc, parent=group)
                for img_tag in li.find_all("img", recursive=True):
                    self._emit_image(img_tag, doc, li_item)
        elif tag_name == "table":
            # Add table item and extract nested images
            data = self._parse_table(tag, doc, parent)
            doc.add_table(data=data, parent=parent)

    def _emit_image(self, img_tag: Tag, doc: DoclingDocument, parent) -> None:
        """
        Helper to create a PictureItem (with optional CAPTION) for an <img> tag.
        """

        if ImageOptions.NONE == self.image_options:
            return

        alt = (img_tag.get("alt") or "").strip()
        caption_item = None
        if alt:
            caption_item = doc.add_text(DocItemLabel.CAPTION, alt, parent=parent)

        src_url = img_tag.get("src")
        width = img_tag.get("width", "128")
        height = img_tag.get("height", "128")
        img_ref = None
        if ImageOptions.INLINE == self.image_options:
            try:
                if src_url.startswith("http"):
                    img = Image.open(requests.get(src_url, stream=True).raw)
                elif src_url.startswith("file:"):
                    img = Image.open(src_url)
                elif src_url.startswith("data:"):
                    image_data = re.sub("^data:image/.+;base64,", "", src_url)
                    img = Image.open(BytesIO(base64.b64decode(image_data)))
                else:
                    return
                img_ref = ImageRef.from_pil(img, dpi=int(img.info.get("dpi")[0]))
            except (FileNotFoundError, UnidentifiedImageError) as ve:
                _log.warning(f"Could not load image (src={src_url}): {ve}")
                return
        elif ImageOptions.REFERENCED == self.image_options:
            try:
                img_url = AnyUrl(src_url)
                img_ref = ImageRef(
                    uri=img_url,
                    dpi=72,
                    mimetype="image/png",
                    size=Size(width=float(width), height=float(height)),
                )
            except ValidationError as ve:
                _log.warning(f"Could not load image (src={src_url}): {ve}")
                return

        doc.add_picture(image=img_ref, caption=caption_item, parent=parent)

    def _parse_table(self, table_tag: Tag, doc: DoclingDocument, parent) -> TableData:
        """
        Convert an HTML table into TableData, capturing cell spans and text,
        and emitting any nested images as PictureItems.
        """
        # Build TableData
        rows = []
        for sec in ("thead", "tbody", "tfoot"):
            section = table_tag.find(sec)
            if section:
                rows.extend(section.find_all("tr", recursive=False))
        if not rows:
            rows = table_tag.find_all("tr", recursive=False)
        occupied: dict[tuple[int, int], bool] = {}
        cells: list[TableCell] = []
        max_cols = 0
        for r, tr in enumerate(rows):
            c = 0
            for cell_tag in tr.find_all(("td", "th"), recursive=False):
                while occupied.get((r, c)):
                    c += 1
                rs = int(cell_tag.get("rowspan", 1) or 1)
                cs = int(cell_tag.get("colspan", 1) or 1)
                txt = cell_tag.get_text(strip=True)
                cell = TableCell(
                    bbox=None,
                    row_span=rs,
                    col_span=cs,
                    start_row_offset_idx=r,
                    end_row_offset_idx=r + rs,
                    start_col_offset_idx=c,
                    end_col_offset_idx=c + cs,
                    text=txt,
                    column_header=(cell_tag.name == "th"),
                    row_header=False,
                    row_section=False,
                )
                cells.append(cell)
                for dr in range(rs):
                    for dc in range(cs):
                        occupied[(r + dr, c + dc)] = True
                c += cs
            max_cols = max(max_cols, c)
        # Emit images inside this table
        for img_tag in table_tag.find_all("img", recursive=True):
            self._emit_image(img_tag, doc, parent)
        return TableData(table_cells=cells, num_rows=len(rows), num_cols=max_cols)


class HTMLDocumentBackend(BaseHTMLDocumentBackend):
    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
    ):
        super().__init__(in_doc, path_or_stream, image_options=ImageOptions.NONE)


class HTMLDocumentBackendImagesInline(BaseHTMLDocumentBackend):
    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
    ):
        super().__init__(in_doc, path_or_stream, image_options=ImageOptions.INLINE)


class HTMLDocumentBackendImagesReferenced(BaseHTMLDocumentBackend):
    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
    ):
        super().__init__(in_doc, path_or_stream, image_options=ImageOptions.REFERENCED)
