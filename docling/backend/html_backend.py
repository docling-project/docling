import base64
import logging
import re
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import requests
from PIL import Image, UnidentifiedImageError
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
from pydantic import AnyUrl, ValidationError
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)

DEFAULT_IMAGE_WIDTH = 128
DEFAULT_IMAGE_HEIGHT = 128

# Tags that initiate distinct Docling items
_BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table"}


class ImageOptions(str, Enum):
    """Image options for HTML backend."""
    NONE = "none"
    EMBEDDED = "embedded"
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
        title = self.soup.find("title")
        if title:
            doc.add_title(title.get_text())
        # remove scripts/styles
        for tag in self.soup.find_all(["script", "style"]):
            tag.decompose()

        body = self.soup.body or self.soup
        # normalize <br>
        for br in body.find_all("br"):
            br.replace_with(NavigableString("\n"))

        headers = body.find(list(_BLOCK_TAGS))
        self.content_layer = (
            ContentLayer.BODY if headers is None else ContentLayer.FURNITURE
        )

        self._walk(body, doc, parent=doc.body)
        return doc

    def _walk(self, element: Tag, doc: DoclingDocument, parent) -> None:
        buffer: list[str] = []

        def flush_buffer():
            if not buffer:
                return
            text = "".join(buffer).strip()
            buffer.clear()
            if not text:
                return
            for part in text.split("\n"):
                seg = part.strip()
                if seg:
                    doc.add_text(DocItemLabel.TEXT, seg, parent=parent)

        for node in element.contents:
            if isinstance(node, Tag) and node.name.lower() in ("script", "style"):
                continue
            if isinstance(node, Tag) and node.name.lower() == "img":
                flush_buffer()
                self._emit_image(node, doc, parent)
                continue
            if isinstance(node, Tag) and node.name.lower() in _BLOCK_TAGS:
                flush_buffer()
                self._handle_block(node, doc, parent)
            elif isinstance(node, Tag) and node.find(list(_BLOCK_TAGS)):
                flush_buffer()
                self._walk(node, doc, parent)
            elif isinstance(node, Tag):
                buffer.append(node.get_text())
            elif isinstance(node, NavigableString):
                buffer.append(str(node))

        flush_buffer()

    def _handle_block(self, tag: Tag, doc: DoclingDocument, parent) -> None:
        tag_name = tag.name.lower()

        if tag_name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag_name[1])
            text = tag.get_text(strip=False)
            if text:
                doc.add_heading(text.strip(), level=level, parent=parent)
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
            is_ordered = (tag_name == "ol")
            # Create the list container
            list_group = (
                doc.add_ordered_list(parent=parent)
                if is_ordered
                else doc.add_unordered_list(parent=parent)
            )

            # For each top-level <li> in this list
            for li in tag.find_all("li", recursive=False):
                # 1) extract only the "direct" text from this <li>
                parts: list[str] = []
                for child in li.contents:
                    if isinstance(child, NavigableString):
                        text_part = child.strip()
                        if text_part:
                            parts.append(text_part)
                    elif isinstance(child, Tag) and child.name not in ("ul", "ol"):
                        text_part = child.get_text(separator=" ", strip=True)
                        if text_part:
                            parts.append(text_part)
                li_text = " ".join(parts)

                # 2) add the list item
                li_item = doc.add_list_item(
                    text=li_text, enumerated=is_ordered, parent=list_group
                )

                # 3) recurse into any nested lists, attaching them to this <li> item
                for sublist in li.find_all(["ul", "ol"], recursive=False):
                    self._handle_block(sublist, doc, parent=li_item)

                # 4) extract any images under this <li>
                for img_tag in li.find_all("img", recursive=True):
                    self._emit_image(img_tag, doc, li_item)

        elif tag_name == "table":
            data = self._parse_table(tag, doc, parent)
            doc.add_table(data=data, parent=parent)

    def _emit_image(self, img_tag: Tag, doc: DoclingDocument, parent) -> None:
        if self.image_options == ImageOptions.NONE:
            return

        alt = (img_tag.get("alt") or "").strip()
        caption_item = None
        if alt:
            caption_item = doc.add_text(DocItemLabel.CAPTION, alt, parent=parent)

        src_url = img_tag.get("src", "")
        width = img_tag.get("width", str(DEFAULT_IMAGE_WIDTH))
        height = img_tag.get("height", str(DEFAULT_IMAGE_HEIGHT))
        img_ref: Optional[ImageRef] = None

        if self.image_options == ImageOptions.EMBEDDED:
            try:
                if src_url.startswith("http"):
                    img = Image.open(requests.get(src_url, stream=True).raw)
                elif src_url.startswith("data:"):
                    data = re.sub(r"^data:image/.+;base64,", "", src_url)
                    img = Image.open(BytesIO(base64.b64decode(data)))
                else:
                    return
                img_ref = ImageRef.from_pil(img, dpi=int(img.info.get("dpi", (72,))[0]))
            except (FileNotFoundError, UnidentifiedImageError) as e:
                _log.warning(f"Could not load image (src={src_url}): {e}")
                return

        elif self.image_options == ImageOptions.REFERENCED:
            try:
                img_ref = ImageRef(
                    uri=AnyUrl(src_url),
                    dpi=72,
                    mimetype="image/png",
                    size=Size(width=float(width), height=float(height)),
                )
            except ValidationError as e:
                _log.warning(f"Could not load image (src={src_url}): {e}")
                return

        doc.add_picture(image=img_ref, caption=caption_item, parent=parent)

    def _parse_table(self, table_tag: Tag, doc: DoclingDocument, parent) -> TableData:
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

        # emit any images in the table
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


class HTMLDocumentBackendImagesEmbedded(BaseHTMLDocumentBackend):
    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
    ):
        super().__init__(in_doc, path_or_stream, image_options=ImageOptions.EMBEDDED)


class HTMLDocumentBackendImagesReferenced(BaseHTMLDocumentBackend):
    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
    ):
        super().__init__(in_doc, path_or_stream, image_options=ImageOptions.REFERENCED)
