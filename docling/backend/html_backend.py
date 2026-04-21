import base64
import logging
import math
import os
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field as dataclass_field
from io import BytesIO
from pathlib import Path
from typing import Any, Final, Iterator, Literal, Optional, Union, cast
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, PageElement, Tag
from bs4.element import PreformattedString
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GraphCell,
    GraphCellLabel,
    GraphData,
    GraphLink,
    GraphLinkLabel,
    GroupItem,
    GroupLabel,
    PictureClassificationLabel,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureItem,
    PictureMeta,
    ProvenanceItem,
    RefItem,
    RichTableCell,
    Size,
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
    "signature",
    "stamp",
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

_DATA_DOCLING_ID_ATTR: Final = "data-docling-id"
_FORM_CONTAINER_CLASS: Final = "form_region"
_FORM_KEY_ID_RE: Final = re.compile(r"^key(?P<key_id>[A-Za-z0-9]+)$")
_FORM_MARKER_ID_RE: Final = re.compile(r"^key(?P<key_id>[A-Za-z0-9]+)_marker$")
_FORM_VALUE_ID_RE: Final = re.compile(
    r"^key(?P<key_id>[A-Za-z0-9]+)_value(?P<value_id>[A-Za-z0-9]+)$"
)
_CUSTOM_CHECKBOX_CLASSES: Final = {"checkbox", "checkbox-box", "checkbox-input"}
_CHECKBOX_MARK_TEXTS: Final = {"x", "✓", "✔", "☑"}
_CHECKBOX_CONTAINER_CLASSES: Final = {
    "checkbox-container",
    "checkbox-item",
    "checkbox-option",
    "option",
}
_INLINE_HTML_TAGS: Final = {
    "a",
    "abbr",
    "b",
    "bdi",
    "bdo",
    "cite",
    "code",
    "data",
    "dfn",
    "em",
    "i",
    "kbd",
    "label",
    "mark",
    "q",
    "s",
    "samp",
    "small",
    "span",
    "strong",
    "sub",
    "sup",
    "u",
    "var",
}


@dataclass(frozen=True)
class _RenderedBBox:
    page_no: int
    bbox: BoundingBox


@dataclass
class _ExtractedFormValue:
    tag: Tag
    order: int
    orig: str
    text: str
    prov: Optional[ProvenanceItem]
    kind: Literal["read_only", "fillable"] = "read_only"
    checkbox_label: Optional[DocItemLabel] = None
    consumed_label_tag_obj_ids: set[int] = dataclass_field(default_factory=set)
    checkbox_label_tags: list[Tag] = dataclass_field(default_factory=list)


@dataclass
class _ExtractedFormMarker:
    tag: Tag
    order: int
    orig: str
    text: str
    prov: Optional[ProvenanceItem]


@dataclass
class _ExtractedFormText:
    tag: Tag
    order: int
    orig: str
    text: str
    prov: Optional[ProvenanceItem]
    label: DocItemLabel = DocItemLabel.TEXT


@dataclass
class _ExtractedFormField:
    key_tag: Optional[Tag]
    key_order: int
    key_orig: str
    key_text: str
    key_prov: Optional[ProvenanceItem]
    marker: Optional[_ExtractedFormMarker]
    values: list[_ExtractedFormValue]
    extra_texts: list[_ExtractedFormText] = dataclass_field(default_factory=list)


@dataclass
class _ExtractedFormRegion:
    fields: list[_ExtractedFormField]
    consumed_tag_ids: set[str]


class _Context(BaseModel):
    list_ordered_flag_by_ref: dict[str, bool] = {}
    list_start_by_ref: dict[str, int] = {}


class AnnotatedText(BaseModel):
    text: str
    hyperlink: Union[AnyUrl, Path, None] = None
    formatting: Union[Formatting, None] = None
    code: bool = False
    source_tag_id: Optional[str] = None


class AnnotatedTextList(list):
    def to_single_text_element(self) -> AnnotatedText:
        current_h = None
        current_text = ""
        current_f = None
        current_code = False
        current_source_tag_id = None
        for at in self:
            t = at.text
            h = at.hyperlink
            f = at.formatting
            c = at.code
            s = at.source_tag_id
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
            if s is not None and current_source_tag_id is None:
                current_source_tag_id = s
            elif (
                s is not None
                and current_source_tag_id is not None
                and s != current_source_tag_id
            ):
                _log.warning(
                    "Clashing provenance tags: "
                    f"'{s}' and '{current_source_tag_id}'! "
                    f"Chose '{current_source_tag_id}'"
                )
            current_code = c if c else current_code

        return AnnotatedText(
            text=current_text.strip(),
            hyperlink=current_h,
            formatting=current_f,
            code=current_code,
            source_tag_id=current_source_tag_id,
        )

    def simplify_text_elements(self) -> "AnnotatedTextList":
        simplified = AnnotatedTextList()
        if not self:
            return self
        text = self[0].text
        hyperlink = self[0].hyperlink
        formatting = self[0].formatting
        code = self[0].code
        source_tag_id = self[0].source_tag_id
        last_elm = text
        for i in range(1, len(self)):
            if (
                hyperlink == self[i].hyperlink
                and formatting == self[i].formatting
                and code == self[i].code
                and source_tag_id == self[i].source_tag_id
            ):
                sep = " "
                if not self[i].text.strip() or not last_elm.strip():
                    sep = ""
                text += sep + self[i].text
                last_elm = self[i].text
            else:
                simplified.append(
                    AnnotatedText(
                        text=text,
                        hyperlink=hyperlink,
                        formatting=formatting,
                        code=code,
                        source_tag_id=source_tag_id,
                    )
                )
                text = self[i].text
                last_elm = text
                hyperlink = self[i].hyperlink
                formatting = self[i].formatting
                code = self[i].code
                source_tag_id = self[i].source_tag_id
        if text:
            simplified.append(
                AnnotatedText(
                    text=text,
                    hyperlink=hyperlink,
                    formatting=formatting,
                    code=code,
                    source_tag_id=source_tag_id,
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
        self.base_path: Optional[str] = (
            str(options.source_uri) if options.source_uri is not None else None
        )

        # Initialize the parents for the hierarchy
        self.max_levels = 10
        self.level = 0
        self.parents: dict[int, Optional[Union[DocItem, GroupItem]]] = {}
        self.ctx = _Context()
        self._disable_inline_group_depth: int = 0
        for i in range(self.max_levels):
            self.parents[i] = None
        self.hyperlink: Union[AnyUrl, Path, None] = None
        self.format_tags: list[str] = []
        self._raw_html_bytes: Optional[bytes] = None
        self._rendered_html: Optional[str] = None
        self._rendered_bbox_by_id: dict[str, _RenderedBBox] = {}
        self._rendered_text_bbox_by_id: dict[str, _RenderedBBox] = {}
        self._rendered_page_images: list[Image.Image] = []
        self._rendered_page_size: Optional[Size] = None
        self._suppressed_tag_ids_stack: list[set[str]] = []
        self._suppressed_tag_obj_ids_stack: list[set[int]] = []
        self._form_fields_by_key_id_stack: list[dict[str, _ExtractedFormField]] = []
        self._tag_name_by_docling_id_cache: dict[str, str] = {}
        self._generated_html_id_counter: int = 0
        self._render_visibility_cache: dict[int, bool] = {}

        try:
            raw = (
                path_or_stream.getvalue()
                if isinstance(path_or_stream, BytesIO)
                else Path(path_or_stream).read_bytes()
            )
            self._raw_html_bytes = raw
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

        if cast(HTMLBackendOptions, self.options).render_page:
            self._render_with_browser()
            if self._rendered_html:
                self.soup = BeautifulSoup(self._rendered_html, "html.parser")

        if self._rendered_page_images and self._rendered_page_size:
            render_dpi = cast(HTMLBackendOptions, self.options).render_dpi
            for page_no, page_image in enumerate(self._rendered_page_images, start=1):
                doc.add_page(
                    page_no=page_no,
                    size=self._rendered_page_size,
                    image=ImageRef.from_pil(image=page_image, dpi=render_dpi),
                )

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
        self._render_visibility_cache.clear()
        self._walk(content, doc)
        return doc

    def _get_render_page_size(self) -> tuple[int, int]:
        options = cast(HTMLBackendOptions, self.options)
        width = options.render_page_width
        height = options.render_page_height
        if options.render_page_orientation == "landscape":
            width, height = height, width
        return width, height

    def _coerce_base_url(self, value: str) -> str:
        if HTMLDocumentBackend._is_remote_url(value) or value.startswith("file://"):
            return value
        return Path(value).resolve().as_uri()

    @staticmethod
    def _host_matches_allowed_entry(hostname: str, allowed_entry: str) -> bool:
        allowed = allowed_entry.strip().lower()
        if not allowed:
            return False
        if allowed.startswith("*."):
            suffix = allowed[2:]
            return bool(suffix) and (
                hostname == suffix or hostname.endswith(f".{suffix}")
            )
        return hostname == allowed or hostname.endswith(f".{allowed}")

    def _is_browser_request_allowed(self, request_url: str) -> bool:
        options = cast(HTMLBackendOptions, self.options)
        policy = options.render_network_policy
        if policy == "open":
            return True

        parsed = urlparse(request_url)
        scheme = (parsed.scheme or "").lower()
        allowed_schemes = {
            scheme_name.strip().lower()
            for scheme_name in options.render_allowed_url_schemes
            if scheme_name and scheme_name.strip()
        }
        if scheme in allowed_schemes:
            return True

        if policy == "strict":
            return False

        if scheme not in {"http", "https"}:
            return False

        hostname = (parsed.hostname or "").strip().lower()
        if not hostname:
            return False
        for allowed_host in options.render_allowed_hosts:
            if self._host_matches_allowed_entry(hostname, allowed_host):
                return True
        return False

    def _get_render_html_text(self) -> str:
        if self._raw_html_bytes is None:
            return ""
        return self._raw_html_bytes.decode("utf-8", errors="replace")

    def _inject_base_tag(self, html_text: str, base_url: Optional[str]) -> str:
        if not base_url:
            return html_text
        soup = BeautifulSoup(html_text, "html.parser")
        if soup.head is None:
            return html_text
        if soup.head.find("base") is not None:
            return html_text
        base_tag = soup.new_tag("base", href=base_url)
        soup.head.insert(0, base_tag)
        return str(soup)

    def _pad_image(self, image: Image.Image, width: int, height: int) -> Image.Image:
        if image.width == width and image.height == height:
            return image
        canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
        canvas.paste(image, (0, 0))
        return canvas

    def _render_with_browser(self) -> None:
        options = cast(HTMLBackendOptions, self.options)
        if not options.render_page:
            return

        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is required for HTML rendering. "
                "Install it with 'pip install \"docling[htmlrender]\"' and run "
                "'playwright install'."
            ) from exc

        width, height = self._get_render_page_size()
        self._rendered_page_size = Size(width=width, height=height)

        render_url: Optional[str] = None
        render_html = self._get_render_html_text()

        if isinstance(self.path_or_stream, Path):
            render_url = self.path_or_stream.resolve().as_uri()
        elif self.base_path:
            render_html = self._inject_base_tag(
                render_html, self._coerce_base_url(self.base_path)
            )

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            offline_mode = options.render_network_policy == "strict"
            context = browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=options.render_device_scale,
                # Disable page JavaScript execution for deterministic static rendering.
                java_script_enabled=False,
                offline=offline_mode,
                service_workers="block"
                if options.render_block_service_workers
                else "allow",
            )

            if options.render_network_policy != "open":

                def _route_request(route, request) -> None:
                    if self._is_browser_request_allowed(request.url):
                        route.continue_()
                    else:
                        route.abort("blockedbyclient")

                context.route("**/*", _route_request)

            page = context.new_page()
            if options.render_print_media:
                page.emulate_media(media="print")

            if render_url:
                page.goto(render_url, wait_until=options.render_wait_until)
            else:
                page.set_content(render_html, wait_until=options.render_wait_until)

            if options.page_padding > 0:
                page.evaluate(
                    """
                    (padding) => {
                      if (!document || !document.body) {
                        return;
                      }
                      document.body.style.padding = `${padding}px`;
                      document.body.style.boxSizing = "border-box";
                    }
                    """,
                    options.page_padding,
                )

            if options.render_wait_ms:
                page.wait_for_timeout(options.render_wait_ms)

            # Some pages settle to their final layout only after first full-page capture.
            # Warm up with a throwaway screenshot so bbox extraction and saved image align.
            page.screenshot(full_page=True)

            render_data = page.evaluate(
                """
                () => {
                  const nodes = Array.from(document.querySelectorAll('*'));
                  const boxes = {};
                  const textBoxes = {};
                  let idx = 0;
                  for (const node of nodes) {
                    idx += 1;
                    const id = String(idx);
                    node.setAttribute('data-docling-id', id);
                    const rect = node.getBoundingClientRect();
                    if (!rect) {
                      continue;
                    }
                    const width = rect.width || 0;
                    const height = rect.height || 0;
                    if (width <= 0 && height <= 0) {
                      continue;
                    }
                    const x = rect.left + window.scrollX;
                    const y = rect.top + window.scrollY;
                    boxes[id] = { x, y, width, height };

                    let textLeft = null;
                    let textTop = null;
                    let textRight = null;
                    let textBottom = null;
                    const textNodes = Array.from(node.childNodes).filter(
                      (child) =>
                        child &&
                        child.nodeType === Node.TEXT_NODE &&
                        child.textContent &&
                        child.textContent.trim()
                    );
                    for (const textNode of textNodes) {
                      const range = document.createRange();
                      range.selectNodeContents(textNode);
                      const rects = Array.from(range.getClientRects());
                      for (const tRect of rects) {
                        const tWidth = tRect.width || 0;
                        const tHeight = tRect.height || 0;
                        if (tWidth <= 0 && tHeight <= 0) {
                          continue;
                        }
                        const tX = tRect.left + window.scrollX;
                        const tY = tRect.top + window.scrollY;
                        const tR = tX + tWidth;
                        const tB = tY + tHeight;
                        textLeft = textLeft === null ? tX : Math.min(textLeft, tX);
                        textTop = textTop === null ? tY : Math.min(textTop, tY);
                        textRight = textRight === null ? tR : Math.max(textRight, tR);
                        textBottom = textBottom === null ? tB : Math.max(textBottom, tB);
                      }
                      range.detach();
                    }
                    if (
                      textLeft !== null &&
                      textTop !== null &&
                      textRight !== null &&
                      textBottom !== null
                    ) {
                      textBoxes[id] = {
                        x: textLeft,
                        y: textTop,
                        width: textRight - textLeft,
                        height: textBottom - textTop
                      };
                    }
                  }
                  const doc = document.documentElement;
                  const body = document.body;
                  const scrollWidth = Math.max(
                    doc ? doc.scrollWidth : 0,
                    body ? body.scrollWidth : 0
                  );
                  const scrollHeight = Math.max(
                    doc ? doc.scrollHeight : 0,
                    body ? body.scrollHeight : 0
                  );
                  return { boxes, textBoxes, scrollWidth, scrollHeight };
                }
                """
            )

            self._rendered_html = page.content()
            scroll_width = int(render_data.get("scrollWidth", width))
            scroll_height = int(render_data.get("scrollHeight", height))
            self._rendered_page_images = self._capture_page_images(
                page=page,
                render_data=render_data,
                page_width=width,
                page_height=height,
                full_page=options.render_full_page,
            )
            if self._rendered_page_images and self._rendered_page_size:
                self._rendered_page_size = Size(
                    width=scroll_width,
                    height=scroll_height if options.render_full_page else height,
                )

            self._rendered_bbox_by_id = self._build_bbox_mapping(
                render_data=render_data,
                page_height=int(self._rendered_page_size.height)
                if self._rendered_page_size
                else height,
                full_page=options.render_full_page,
            )
            self._rendered_text_bbox_by_id = self._build_bbox_mapping(
                render_data={
                    "boxes": render_data.get("textBoxes", {}),
                    "scrollHeight": render_data.get("scrollHeight"),
                },
                page_height=int(self._rendered_page_size.height)
                if self._rendered_page_size
                else height,
                full_page=options.render_full_page,
            )

            context.close()
            browser.close()

    def _capture_page_images(
        self,
        page,
        render_data: dict,
        page_width: int,
        page_height: int,
        full_page: bool,
    ) -> list[Image.Image]:
        scroll_height = int(render_data.get("scrollHeight", page_height))
        if scroll_height <= 0:
            return []

        screenshot_bytes = page.screenshot(full_page=True)
        full_image = Image.open(BytesIO(screenshot_bytes)).convert("RGB")

        if full_page:
            return [full_image]

        page_images: list[Image.Image] = []
        page_count = max(1, math.ceil(scroll_height / page_height))
        scale_y = full_image.height / float(scroll_height)
        target_height = round(page_height * scale_y)

        for page_idx in range(page_count):
            top_css = page_idx * page_height
            bottom_css = min(top_css + page_height, scroll_height)
            top_px = round(top_css * scale_y)
            bottom_px = round(bottom_css * scale_y)
            if bottom_px <= top_px:
                continue
            cropped = full_image.crop((0, top_px, full_image.width, bottom_px))
            cropped = self._pad_image(
                image=cropped, width=full_image.width, height=target_height
            )
            page_images.append(cropped)

        return page_images

    def _build_bbox_mapping(
        self, render_data: dict, page_height: int, full_page: bool
    ) -> dict[str, _RenderedBBox]:
        boxes = render_data.get("boxes", {}) or {}
        scroll_height = float(render_data.get("scrollHeight", page_height))

        if full_page:
            page_count = 1
        else:
            page_count = max(1, math.ceil(scroll_height / page_height))

        mapping: dict[str, _RenderedBBox] = {}
        for tag_id, rect in boxes.items():
            left = float(rect.get("x", 0.0))
            top = float(rect.get("y", 0.0))
            width = float(rect.get("width", 0.0))
            height = float(rect.get("height", 0.0))
            if width <= 0 and height <= 0:
                continue
            right = left + width
            bottom = top + height
            if full_page:
                page_no = 1
                offset = 0.0
            else:
                page_no = int(top // page_height) + 1
                page_no = min(max(page_no, 1), page_count)
                offset = (page_no - 1) * page_height
            bbox = BoundingBox(
                l=left,
                t=top - offset,
                r=right,
                b=bottom - offset,
                coord_origin=CoordOrigin.TOPLEFT,
            )
            mapping[str(tag_id)] = _RenderedBBox(page_no=page_no, bbox=bbox)

        return mapping

    def _get_tag_id(self, tag: Optional[Tag]) -> Optional[str]:
        if tag is None:
            return None
        tag_id = tag.get(_DATA_DOCLING_ID_ATTR)
        if not tag_id:
            return None
        return str(tag_id)

    @staticmethod
    def _get_html_id(tag: Optional[Tag]) -> Optional[str]:
        if tag is None:
            return None
        tag_id = tag.get("id")
        if not isinstance(tag_id, str) or not tag_id:
            return None
        return tag_id

    def _get_rendered_bbox_for_tag(self, tag: Optional[Tag]) -> Optional[_RenderedBBox]:
        tag_id = self._get_tag_id(tag)
        if tag_id is None:
            return None
        return self._rendered_bbox_by_id.get(tag_id)

    def _get_rendered_text_bbox_for_tag(
        self, tag: Optional[Tag]
    ) -> Optional[_RenderedBBox]:
        tag_id = self._get_tag_id(tag)
        if tag_id is None:
            return None
        return self._rendered_text_bbox_by_id.get(tag_id)

    @staticmethod
    def _has_negative_bbox_coordinates(rendered_bbox: Optional[_RenderedBBox]) -> bool:
        if rendered_bbox is None:
            return False
        bbox = rendered_bbox.bbox
        return bbox.l < 0 or bbox.t < 0 or bbox.r < 0 or bbox.b < 0

    def _is_tag_outside_capture_area(self, tag: Tag) -> bool:
        rendered = self._get_rendered_text_bbox_for_tag(
            tag
        ) or self._get_rendered_bbox_for_tag(tag)
        return self._has_negative_bbox_coordinates(rendered)

    @staticmethod
    def _has_inline_hidden_style(tag: Tag) -> bool:
        style = tag.get("style")
        if not isinstance(style, str) or not style.strip():
            return False
        normalized = re.sub(r"\s+", "", style.lower())
        if "display:none" in normalized:
            return True
        if "visibility:hidden" in normalized or "visibility:collapse" in normalized:
            return True
        if re.search(r"opacity:0(?:[;]|$)", normalized):
            return True
        return False

    def _has_rendered_presence(self, tag: Tag) -> bool:
        if not self._rendered_bbox_by_id and not self._rendered_text_bbox_by_id:
            return True

        cache_key = id(tag)
        if cache_key in self._render_visibility_cache:
            return self._render_visibility_cache[cache_key]

        if (
            self._get_rendered_text_bbox_for_tag(tag) is not None
            or self._get_rendered_bbox_for_tag(tag) is not None
        ):
            self._render_visibility_cache[cache_key] = True
            return True

        has_visible_descendant = False
        for descendant in tag.find_all(True):
            if not isinstance(descendant, Tag):
                continue
            if (
                self._get_rendered_text_bbox_for_tag(descendant) is not None
                or self._get_rendered_bbox_for_tag(descendant) is not None
            ):
                has_visible_descendant = True
                break

        self._render_visibility_cache[cache_key] = has_visible_descendant
        return has_visible_descendant

    def _is_invisible_tag(self, tag: Tag) -> bool:
        if tag.has_attr("hidden"):
            return True
        aria_hidden = tag.get("aria-hidden")
        if isinstance(aria_hidden, str) and aria_hidden.strip().lower() in {
            "true",
            "1",
            "yes",
        }:
            return True
        if self._has_inline_hidden_style(tag):
            return True
        if not self._has_rendered_presence(tag):
            return True
        return False

    def _make_prov(
        self,
        text: str,
        tag: Optional[Tag] = None,
        source_tag_id: Optional[str] = None,
    ) -> Optional[ProvenanceItem]:
        if not self._rendered_bbox_by_id:
            return None

        render_box: Optional[_RenderedBBox] = None
        if source_tag_id:
            render_box = self._rendered_bbox_by_id.get(source_tag_id)
        if render_box is None:
            render_box = self._get_rendered_bbox_for_tag(tag)
        if render_box is None:
            return None

        return ProvenanceItem(
            page_no=render_box.page_no,
            bbox=render_box.bbox,
            charspan=(0, len(text)),
        )

    def _make_text_prov(
        self,
        text: str,
        tag: Optional[Tag] = None,
        source_tag_id: Optional[str] = None,
    ) -> Optional[ProvenanceItem]:
        if not self._rendered_text_bbox_by_id:
            return self._make_prov(text=text, tag=tag, source_tag_id=source_tag_id)

        render_box: Optional[_RenderedBBox] = None
        if source_tag_id:
            render_box = self._rendered_text_bbox_by_id.get(source_tag_id)
        if render_box is None:
            render_box = self._get_rendered_text_bbox_for_tag(tag)
        if render_box is None and isinstance(tag, Tag):
            descendant_boxes: list[_RenderedBBox] = []
            for descendant in [tag, *tag.find_all(True)]:
                if not isinstance(descendant, Tag):
                    continue
                descendant_box = self._get_rendered_text_bbox_for_tag(descendant)
                if descendant_box is not None:
                    descendant_boxes.append(descendant_box)
            if descendant_boxes:
                page_no = descendant_boxes[0].page_no
                same_page_boxes = [
                    rendered.bbox
                    for rendered in descendant_boxes
                    if rendered.page_no == page_no
                ]
                if same_page_boxes:
                    render_box = _RenderedBBox(
                        page_no=page_no,
                        bbox=(
                            same_page_boxes[0]
                            if len(same_page_boxes) == 1
                            else BoundingBox.enclosing_bbox(same_page_boxes)
                        ),
                    )
        if render_box is None:
            return self._make_prov(text=text, tag=tag, source_tag_id=source_tag_id)

        return ProvenanceItem(
            page_no=render_box.page_no,
            bbox=render_box.bbox,
            charspan=(0, len(text)),
        )

    def _make_text_prov_for_source_tag_ids(
        self, text: str, tag: Optional[Tag], source_tag_ids: list[str]
    ) -> Optional[ProvenanceItem]:
        unique_ids = list(dict.fromkeys(source_tag_ids))
        if not unique_ids:
            return self._make_text_prov(text=text, tag=tag)

        boxes: list[BoundingBox] = []
        page_no: Optional[int] = None
        for source_id in unique_ids:
            rendered = None
            if self._rendered_text_bbox_by_id:
                rendered = self._rendered_text_bbox_by_id.get(source_id)
            if rendered is None and self._rendered_bbox_by_id:
                rendered = self._rendered_bbox_by_id.get(source_id)
            if rendered is None:
                continue
            if page_no is None:
                page_no = rendered.page_no
            if rendered.page_no != page_no:
                continue
            boxes.append(rendered.bbox)

        if not boxes:
            return self._make_text_prov(text=text, tag=tag, source_tag_id=unique_ids[0])

        bbox = boxes[0] if len(boxes) == 1 else BoundingBox.enclosing_bbox(boxes)
        return ProvenanceItem(
            page_no=page_no if page_no is not None else 1,
            bbox=bbox,
            charspan=(0, len(text)),
        )

    def _get_rendered_bbox_for_source_tag_id(
        self, source_tag_id: str
    ) -> Optional[_RenderedBBox]:
        rendered = None
        if self._rendered_text_bbox_by_id:
            rendered = self._rendered_text_bbox_by_id.get(source_tag_id)
        if rendered is None and self._rendered_bbox_by_id:
            rendered = self._rendered_bbox_by_id.get(source_tag_id)
        return rendered

    def _are_source_tag_ids_inline_neighbors(
        self, left_source_tag_id: str, right_source_tag_id: str
    ) -> bool:
        left = self._get_rendered_bbox_for_source_tag_id(left_source_tag_id)
        right = self._get_rendered_bbox_for_source_tag_id(right_source_tag_id)
        if left is None or right is None:
            return False
        if left.page_no != right.page_no:
            return False

        left_box = left.bbox
        right_box = right.bbox
        min_height = max(1.0, min(left_box.height, right_box.height))
        max_height = max(1.0, max(left_box.height, right_box.height))
        vertical_overlap = min(left_box.b, right_box.b) - max(left_box.t, right_box.t)
        if vertical_overlap <= 0 or (vertical_overlap / min_height) < 0.6:
            return False

        horizontal_gap = right_box.l - left_box.r
        max_gap = max(8.0, 1.5 * max_height)
        min_gap = -0.5 * min(left_box.width, right_box.width)
        return min_gap <= horizontal_gap <= max_gap

    def _compact_adjacent_single_char_parts(
        self, parts: AnnotatedTextList
    ) -> list[tuple[AnnotatedText, list[str]]]:
        compacted: list[tuple[AnnotatedText, list[str]]] = []
        idx = 0
        while idx < len(parts):
            current = parts[idx]
            current_text = HTMLDocumentBackend._clean_unicode(current.text.strip())

            if len(current_text) == 1 and current.source_tag_id is not None:
                run_chars: list[str] = []
                run_source_ids: list[str] = []
                prev_source_id: Optional[str] = None
                run_end = idx
                while run_end < len(parts):
                    candidate = parts[run_end]
                    candidate_text = HTMLDocumentBackend._clean_unicode(
                        candidate.text.strip()
                    )
                    if (
                        len(candidate_text) != 1
                        or candidate.hyperlink != current.hyperlink
                        or candidate.formatting != current.formatting
                        or candidate.code != current.code
                    ):
                        break
                    candidate_source_id = candidate.source_tag_id
                    if candidate_source_id is None:
                        break
                    if (
                        prev_source_id is not None
                        and not self._are_source_tag_ids_inline_neighbors(
                            prev_source_id, candidate_source_id
                        )
                    ):
                        break
                    run_chars.append(candidate_text)
                    run_source_ids.append(candidate_source_id)
                    prev_source_id = candidate_source_id
                    run_end += 1

                if len(run_chars) > 1:
                    compacted.append(
                        (
                            AnnotatedText(
                                text="".join(run_chars),
                                hyperlink=current.hyperlink,
                                formatting=current.formatting,
                                code=current.code,
                                source_tag_id=(
                                    run_source_ids[0] if run_source_ids else None
                                ),
                            ),
                            run_source_ids,
                        )
                    )
                    idx = run_end
                    continue

            source_tag_ids = [current.source_tag_id] if current.source_tag_id else []
            compacted.append((current, source_tag_ids))
            idx += 1
        return compacted

    def _make_checkbox_with_label_prov(
        self, text: str, checkbox_tag: Tag, label_tags: list[Tag]
    ) -> Optional[ProvenanceItem]:
        checkbox_rendered = self._get_rendered_bbox_for_tag(checkbox_tag)
        if checkbox_rendered is None:
            return self._make_prov(text=text, tag=checkbox_tag)

        boxes: list[BoundingBox] = [checkbox_rendered.bbox]
        for label_tag in label_tags:
            rendered = self._get_rendered_text_bbox_for_tag(
                label_tag
            ) or self._get_rendered_bbox_for_tag(label_tag)
            if rendered is None:
                continue
            if rendered.page_no != checkbox_rendered.page_no:
                continue
            boxes.append(rendered.bbox)

        bbox = (
            checkbox_rendered.bbox
            if len(boxes) == 1
            else BoundingBox.enclosing_bbox(boxes)
        )
        return ProvenanceItem(
            page_no=checkbox_rendered.page_no,
            bbox=bbox,
            charspan=(0, len(text)),
        )

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
                if p.get(_DATA_DOCLING_ID_ATTR):
                    current_p[_DATA_DOCLING_ID_ATTR] = p.get(_DATA_DOCLING_ID_ATTR)
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
        has_input = any(child.name == "input" for child in children)
        has_custom_checkbox = any(
            self._is_custom_checkbox_tag(child) for child in children
        )
        has_line_break = any(child.name == "br" for child in children)
        direct_block_text_children = [
            child
            for child in table_cell.find_all(recursive=False)
            if isinstance(child, Tag) and child.name in {"p", "div", "li"}
        ]
        has_nested_form_semantic_id = any(
            self._is_form_semantic_tag(child)
            for child in children
            if isinstance(child, Tag)
        )
        if has_nested_form_semantic_id:
            return True
        if has_line_break or len(direct_block_text_children) > 1:
            return True
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
                is_rich = bool(
                    item for item in children if item.name in {"img", "input"}
                )
            elif len(annotations) == 1:
                anno: AnnotatedText = annotations[0]
                is_rich = (
                    bool(anno.formatting)
                    or bool(anno.hyperlink)
                    or anno.code
                    or has_input
                    or has_custom_checkbox
                )

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
                cell_bbox = None
                rendered_cell = self._get_rendered_bbox_for_tag(html_cell)
                if rendered_cell is not None:
                    cell_bbox = rendered_cell.bbox
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
                        bbox=cell_bbox,
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
                        bbox=cell_bbox,
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

    def _walk(self, element: Tag, doc: DoclingDocument) -> list[RefItem]:  # noqa: C901
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
                compacted_parts = self._compact_adjacent_single_char_parts(
                    annotated_text_list
                )
                force_inline_group = (
                    len(annotated_text_list) == 1
                    and bool(annotated_text_list[0].code)
                    and element.name not in {"p", "pre"}
                )
                with self._use_inline_group(
                    annotated_text_list, doc, force=force_inline_group
                ) as inline_ref:
                    for annotated_text, source_tag_ids in compacted_parts:
                        if annotated_text.text.strip():
                            seg_clean = HTMLDocumentBackend._clean_unicode(
                                annotated_text.text.strip()
                            )
                            if annotated_text.code:
                                prov = self._make_text_prov_for_source_tag_ids(
                                    text=seg_clean,
                                    tag=element,
                                    source_tag_ids=source_tag_ids,
                                )
                                docling_code2 = doc.add_code(
                                    parent=self.parents[self.level],
                                    text=seg_clean,
                                    content_layer=self.content_layer,
                                    formatting=annotated_text.formatting,
                                    hyperlink=annotated_text.hyperlink,
                                    prov=prov,
                                )
                                if inline_ref is None:
                                    added_refs.append(docling_code2.get_ref())
                            else:
                                prov = self._make_text_prov_for_source_tag_ids(
                                    text=seg_clean,
                                    tag=element,
                                    source_tag_ids=source_tag_ids,
                                )
                                docling_text2 = doc.add_text(
                                    parent=self.parents[self.level],
                                    label=DocItemLabel.TEXT,
                                    text=seg_clean,
                                    content_layer=self.content_layer,
                                    formatting=annotated_text.formatting,
                                    hyperlink=annotated_text.hyperlink,
                                    prov=prov,
                                )
                                if inline_ref is None:
                                    added_refs.append(docling_text2.get_ref())
                    if inline_ref is not None:
                        added_refs.append(inline_ref)

        for node in element.contents:
            if isinstance(node, Tag):
                name = node.name.lower()
                if form_field := self._consume_form_field_for_tag(node):
                    _flush_buffer()
                    added_refs.extend(
                        self._add_field_item_from_extracted(
                            field=form_field,
                            doc=doc,
                            parent=self.parents[self.level],
                        )
                    )
                    continue
                if self._is_suppressed_tag(node):
                    if name == "br":
                        # Keep explicit line breaks as text boundaries even when
                        # the <br> tag itself has no rendered bbox.
                        _flush_buffer()
                    continue
                has_block_descendants = bool(
                    node.find(_BLOCK_TAGS)
                    or node.find("input")
                    or node.find(
                        lambda item: isinstance(item, Tag)
                        and self._is_custom_checkbox_tag(item)
                    )
                )
                has_pending_form_fields = self._has_pending_form_field_in_subtree(node)
                if self._is_form_container(node):
                    _flush_buffer()
                    form_refs = self._handle_form_container(node, doc)
                    added_refs.extend(form_refs)
                    continue
                if self._should_flatten_info_text(node):
                    _flush_buffer()
                    flattened_ref = self._emit_flattened_text_tag(node, doc)
                    if flattened_ref is not None:
                        added_refs.append(flattened_ref)
                    continue
                if self._is_custom_checkbox_tag(node):
                    _flush_buffer()
                    checkbox_ref = self._emit_custom_checkbox(node, doc)
                    if checkbox_ref is not None:
                        added_refs.append(checkbox_ref)
                    continue
                if name == "img":
                    _flush_buffer()
                    im_ref3 = self._emit_image(node, doc)
                    if im_ref3:
                        added_refs.append(im_ref3)
                elif name == "input":
                    _flush_buffer()
                    input_ref = self._emit_input(node, doc)
                    if input_ref:
                        added_refs.append(input_ref)
                elif name in _FORMAT_TAG_MAP:
                    if has_block_descendants or has_pending_form_fields:
                        _flush_buffer()
                        with self._use_format([name]):
                            wk = self._walk(node, doc)
                            added_refs.extend(wk)
                    else:
                        with self._use_format([name]):
                            buffer.extend(
                                self._extract_text_and_hyperlink_recursively(
                                    node,
                                    find_parent_annotation=True,
                                    keep_newlines=False,
                                )
                            )
                elif name == "a":
                    if has_block_descendants or has_pending_form_fields:
                        _flush_buffer()
                        with self._use_hyperlink(node):
                            wk2 = self._walk(node, doc)
                            added_refs.extend(wk2)
                    else:
                        with self._use_hyperlink(node):
                            buffer.extend(
                                self._extract_text_and_hyperlink_recursively(
                                    node,
                                    find_parent_annotation=True,
                                    keep_newlines=False,
                                )
                            )
                elif name in _BLOCK_TAGS:
                    if name != "table":
                        for field in self._consume_form_fields_in_subtree(node):
                            _flush_buffer()
                            added_refs.extend(
                                self._add_field_item_from_extracted(
                                    field=field,
                                    doc=doc,
                                    parent=self.parents[self.level],
                                )
                            )
                    _flush_buffer()
                    blk = self._handle_block(node, doc)
                    added_refs.extend(blk)
                elif has_block_descendants:
                    _flush_buffer()
                    wk3 = self._walk(node, doc)
                    added_refs.extend(wk3)
                elif has_pending_form_fields:
                    # Preserve DOM reading order: recurse into inline containers
                    # with pending form fields instead of bulk-emitting them.
                    _flush_buffer()
                    wk4 = self._walk(node, doc)
                    added_refs.extend(wk4)
                elif self._should_buffer_tag_text_inline(node):
                    buffer.extend(
                        self._extract_text_and_hyperlink_recursively(
                            node, find_parent_annotation=True, keep_newlines=False
                        )
                    )
                else:
                    _flush_buffer()
                    wk5 = self._walk(node, doc)
                    added_refs.extend(wk5)
            elif isinstance(node, NavigableString) and not isinstance(
                node, PreformattedString
            ):
                node_text = str(node)
                if node_text.strip("\n\r") == "":
                    parent_tag = node.parent if isinstance(node.parent, Tag) else None
                    if (
                        parent_tag is not None
                        and parent_tag.name in {"td", "th"}
                        and "\n" in node_text
                    ):
                        _flush_buffer()
                    continue
                else:
                    buffer.extend(
                        self._extract_text_and_hyperlink_recursively(
                            node, find_parent_annotation=True, keep_newlines=False
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
            if isinstance(item.parent, Tag):
                if self._is_suppressed_tag(item.parent):
                    return AnnotatedTextList()
                if self._is_checkbox_label_container(item.parent):
                    return AnnotatedTextList()
            text = item.strip()
            code = any(code_tag in self.format_tags for code_tag in _CODE_TAG_SET)
            source_tag_id = (
                self._get_tag_id(item.parent) if isinstance(item.parent, Tag) else None
            )
            if text:
                return AnnotatedTextList(
                    [
                        AnnotatedText(
                            text=text,
                            hyperlink=self.hyperlink,
                            formatting=self._formatting,
                            code=code,
                            source_tag_id=source_tag_id,
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
                            source_tag_id=source_tag_id,
                        )
                    ]
                )
            return AnnotatedTextList()

        tag = cast(Tag, item)
        if self._is_suppressed_tag(tag):
            return AnnotatedTextList()
        if self._is_checkbox_like_tag(tag):
            return AnnotatedTextList()
        if self._is_checkbox_label_tag(tag):
            return AnnotatedTextList()
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

    def _get_tag_name_for_docling_id(self, source_tag_id: str) -> Optional[str]:
        if source_tag_id in self._tag_name_by_docling_id_cache:
            tag_name = self._tag_name_by_docling_id_cache[source_tag_id]
            return tag_name or None
        if self.soup is None:
            return None
        tag = self.soup.find(attrs={_DATA_DOCLING_ID_ATTR: source_tag_id})
        tag_name = tag.name if isinstance(tag, Tag) else ""
        self._tag_name_by_docling_id_cache[source_tag_id] = tag_name
        return tag_name or None

    def _should_create_inline_group(
        self, annotated_text_list: AnnotatedTextList
    ) -> bool:
        if len(annotated_text_list) <= 1:
            return False
        # In non-render mode there are no source tag ids. Still keep mixed
        # inline formatting (e.g. <p>...<strong>...</strong>) as one flow.
        if all(
            annotated_text.source_tag_id is None
            for annotated_text in annotated_text_list
        ):
            return True
        # Allow paragraph-like block containers to contribute inline segments
        # when mixed with formatting tags (e.g., <p>text <strong>bold</strong>).
        inline_group_container_tags = {"p", "address", "summary", "td", "th"}
        for annotated_text in annotated_text_list:
            source_tag_id = annotated_text.source_tag_id
            if source_tag_id is None:
                return False
            tag_name = self._get_tag_name_for_docling_id(source_tag_id)
            if tag_name is None:
                return False
            if (
                tag_name not in _INLINE_HTML_TAGS
                and tag_name not in inline_group_container_tags
            ):
                return False
        return True

    @contextmanager
    def _use_inline_group(
        self,
        annotated_text_list: AnnotatedTextList,
        doc: DoclingDocument,
        force: bool = False,
    ) -> Iterator[RefItem | None]:
        """Create an inline group for annotated texts.

        Checks if annotated_text_list has more than one item and if so creates an inline
        group in which the text elements can then be generated. While the context manager
        is active the inline group is set as the current parent.

        Args:
            annotated_text_list (AnnotatedTextList): Annotated text
            doc (DoclingDocument): Currently used document

        Yields:
            The RefItem of the created InlineGroup, or None when the list has only one
                element and no group is created.
        """
        if self._disable_inline_group_depth > 0:
            yield None
            return
        if not force and not self._should_create_inline_group(annotated_text_list):
            yield None
            return
        inline_fmt = doc.add_group(
            label=GroupLabel.INLINE,
            parent=self.parents[self.level],
            content_layer=self.content_layer,
        )
        self.parents[self.level + 1] = inline_fmt
        self.level += 1
        try:
            yield inline_fmt.get_ref()
        finally:
            self.parents[self.level] = None
            self.level -= 1

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
    def _use_form_container(self, form_item: DocItem):
        """Create a form container group and set it as the current parent."""
        self.parents[self.level + 1] = form_item
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
        prov = self._make_text_prov(
            text=text_clean,
            tag=tag,
            source_tag_id=annotated_text.source_tag_id,
        )
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
                prov=prov,
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
                prov=prov,
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

    def _handle_list(self, tag: Tag, doc: DoclingDocument) -> RefItem:  # noqa: C901
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
                inputs_in_li = [
                    input_tag
                    for input_tag in li.find_all("input")
                    if input_tag.find_parent("li") is li
                ]
                custom_checkboxes_in_li = [
                    checkbox_tag
                    for checkbox_tag in li.find_all(
                        lambda item: isinstance(item, Tag)
                        and self._is_custom_checkbox_tag(item)
                    )
                    if checkbox_tag.find_parent("li") is li
                ]

                # 3) add the list item
                if li_text or inputs_in_li or custom_checkboxes_in_li:
                    if len(min_parts) > 1:
                        li_prov = self._make_text_prov(text=li_text, tag=li)
                        # create an empty list element in order to hook the inline group onto that one
                        self.parents[self.level + 1] = doc.add_list_item(
                            text="",
                            enumerated=is_ordered,
                            marker=marker,
                            parent=list_group,
                            content_layer=self.content_layer,
                            prov=li_prov,
                        )
                        self.level += 1
                        with self._use_inline_group(min_parts, doc):
                            compacted_parts = self._compact_adjacent_single_char_parts(
                                min_parts
                            )
                            for annotated_text, source_tag_ids in compacted_parts:
                                li_text = re.sub(
                                    r"\s+|\n+", " ", annotated_text.text
                                ).strip()
                                li_clean = HTMLDocumentBackend._clean_unicode(li_text)
                                if annotated_text.code:
                                    prov = self._make_text_prov_for_source_tag_ids(
                                        text=li_clean,
                                        tag=li,
                                        source_tag_ids=source_tag_ids,
                                    )
                                    doc.add_code(
                                        parent=self.parents[self.level],
                                        text=li_clean,
                                        content_layer=self.content_layer,
                                        formatting=annotated_text.formatting,
                                        hyperlink=annotated_text.hyperlink,
                                        prov=prov,
                                    )
                                else:
                                    prov = self._make_text_prov_for_source_tag_ids(
                                        text=li_clean,
                                        tag=li,
                                        source_tag_ids=source_tag_ids,
                                    )
                                    doc.add_text(
                                        parent=self.parents[self.level],
                                        label=DocItemLabel.TEXT,
                                        text=li_clean,
                                        content_layer=self.content_layer,
                                        formatting=annotated_text.formatting,
                                        hyperlink=annotated_text.hyperlink,
                                        prov=prov,
                                    )

                        for input_tag in inputs_in_li:
                            if isinstance(input_tag, Tag):
                                self._emit_input(input_tag, doc)
                        for checkbox_tag in custom_checkboxes_in_li:
                            if isinstance(checkbox_tag, Tag):
                                self._emit_custom_checkbox(checkbox_tag, doc)

                        # 4) recurse into any nested lists, attaching them to this <li> item
                        for sublist in li({"ul", "ol"}, recursive=False):
                            if isinstance(sublist, Tag):
                                self._handle_block(sublist, doc)

                        # now the list element with inline group is not a parent anymore
                        self.parents[self.level] = None
                        self.level -= 1
                    elif li_text:
                        annotated_text = min_parts[0]
                        li_text = re.sub(r"\s+|\n+", " ", annotated_text.text).strip()
                        li_clean = HTMLDocumentBackend._clean_unicode(li_text)
                        prov = self._make_text_prov(
                            text=li_clean,
                            tag=li,
                            source_tag_id=annotated_text.source_tag_id,
                        )
                        self.parents[self.level + 1] = doc.add_list_item(
                            text=li_clean,
                            enumerated=is_ordered,
                            marker=marker,
                            orig=li_text,
                            parent=list_group,
                            content_layer=self.content_layer,
                            formatting=annotated_text.formatting,
                            hyperlink=annotated_text.hyperlink,
                            prov=prov,
                        )

                        if inputs_in_li or custom_checkboxes_in_li:
                            self.level += 1
                            for input_tag in inputs_in_li:
                                if isinstance(input_tag, Tag):
                                    self._emit_input(input_tag, doc)
                            for checkbox_tag in custom_checkboxes_in_li:
                                if isinstance(checkbox_tag, Tag):
                                    self._emit_custom_checkbox(checkbox_tag, doc)
                            self.level -= 1

                        # 4) recurse into any nested lists, attaching them to this <li> item
                        for sublist in li({"ul", "ol"}, recursive=False):
                            if isinstance(sublist, Tag):
                                self.level += 1
                                self._handle_block(sublist, doc)
                                self.parents[self.level + 1] = None
                                self.level -= 1
                    else:
                        li_prov = self._make_text_prov(text="", tag=li)
                        self.parents[self.level + 1] = doc.add_list_item(
                            text="",
                            enumerated=is_ordered,
                            marker=marker,
                            parent=list_group,
                            content_layer=self.content_layer,
                            prov=li_prov,
                        )
                        self.level += 1
                        for input_tag in inputs_in_li:
                            if isinstance(input_tag, Tag):
                                self._emit_input(input_tag, doc)
                        for checkbox_tag in custom_checkboxes_in_li:
                            if isinstance(checkbox_tag, Tag):
                                self._emit_custom_checkbox(checkbox_tag, doc)
                        for sublist in li({"ul", "ol"}, recursive=False):
                            if isinstance(sublist, Tag):
                                self._handle_block(sublist, doc)
                        self.parents[self.level] = None
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

    def _handle_block(self, tag: Tag, doc: DoclingDocument) -> list[RefItem]:  # noqa: C901
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
                compacted_part = self._compact_adjacent_single_char_parts(part)
                with self._use_inline_group(part, doc) as inline_ref:
                    for annotated_text, source_tag_ids in compacted_part:
                        if seg := annotated_text.text.strip():
                            seg_clean = HTMLDocumentBackend._clean_unicode(seg)
                            if annotated_text.code:
                                prov = self._make_text_prov_for_source_tag_ids(
                                    text=seg_clean,
                                    tag=tag,
                                    source_tag_ids=source_tag_ids,
                                )
                                docling_code = doc.add_code(
                                    parent=self.parents[self.level],
                                    text=seg_clean,
                                    content_layer=self.content_layer,
                                    formatting=annotated_text.formatting,
                                    hyperlink=annotated_text.hyperlink,
                                    prov=prov,
                                )
                                if inline_ref is None:
                                    added_refs.append(docling_code.get_ref())
                            else:
                                prov = self._make_text_prov_for_source_tag_ids(
                                    text=seg_clean,
                                    tag=tag,
                                    source_tag_ids=source_tag_ids,
                                )
                                docling_text = doc.add_text(
                                    parent=self.parents[self.level],
                                    label=DocItemLabel.TEXT,
                                    text=seg_clean,
                                    content_layer=self.content_layer,
                                    formatting=annotated_text.formatting,
                                    hyperlink=annotated_text.hyperlink,
                                    prov=prov,
                                )
                                if inline_ref is None:
                                    added_refs.append(docling_text.get_ref())
                    if inline_ref is not None:
                        added_refs.append(inline_ref)

            for img_tag in tag("img"):
                if isinstance(img_tag, Tag):
                    self._emit_image(img_tag, doc)
            for input_tag in tag("input"):
                if isinstance(input_tag, Tag):
                    input_ref = self._emit_input(input_tag, doc)
                    if input_ref is not None:
                        added_refs.append(input_ref)
            for checkbox_tag in tag.find_all(
                lambda item: isinstance(item, Tag)
                and self._is_custom_checkbox_tag(item)
            ):
                if isinstance(checkbox_tag, Tag):
                    checkbox_ref = self._emit_custom_checkbox(checkbox_tag, doc)
                    if checkbox_ref is not None:
                        added_refs.append(checkbox_ref)

        elif tag_name == "table":
            num_rows, num_cols = self.get_html_table_row_col(tag)
            data_e = TableData(num_rows=num_rows, num_cols=num_cols)
            table_prov = self._make_prov(text="", tag=tag)
            docling_table = doc.add_table(
                data=data_e,
                parent=self.parents[self.level],
                prov=table_prov,
                content_layer=self.content_layer,
            )
            added_refs.append(docling_table.get_ref())
            self.parse_table_data(tag, doc, docling_table, num_rows, num_cols)

        elif tag_name in {"stamp", "signature"}:
            _class_name = PictureClassificationLabel.STAMP.value
            if tag_name == "signature":
                _class_name = PictureClassificationLabel.SIGNATURE.value
            placeholder: PictureItem = doc.add_picture(
                parent=self.parents[self.level],
                content_layer=self.content_layer,
            )
            placeholder.meta = PictureMeta(
                classification=PictureClassificationMetaField(
                    predictions=[
                        PictureClassificationPrediction(
                            class_name=_class_name,
                        )
                    ],
                ),
            )
            text = HTMLDocumentBackend._clean_unicode(self.get_text(tag).strip())
            doc.add_text(label=DocItemLabel.TEXT, text=text, parent=placeholder)

        elif tag_name in {"pre"}:
            # handle monospace code snippets (pre).
            text_list = self._extract_text_and_hyperlink_recursively(
                tag, find_parent_annotation=True, keep_newlines=True
            )
            annotated_texts = text_list.simplify_text_elements()
            with self._use_inline_group(annotated_texts, doc) as inline_ref:
                for annotated_text in annotated_texts:
                    text_clean = HTMLDocumentBackend._clean_unicode(
                        annotated_text.text.strip()
                    )
                    prov = self._make_prov(
                        text=text_clean,
                        tag=tag,
                        source_tag_id=annotated_text.source_tag_id,
                    )
                    docling_code2 = doc.add_code(
                        parent=self.parents[self.level],
                        text=text_clean,
                        content_layer=self.content_layer,
                        formatting=annotated_text.formatting,
                        hyperlink=annotated_text.hyperlink,
                        prov=prov,
                    )
                    if inline_ref is None:
                        added_refs.append(docling_code2.get_ref())
            if inline_ref is not None:
                added_refs.append(inline_ref)

        elif tag_name == "footer":
            with self._use_footer(tag, doc):
                self._walk(tag, doc)

        elif tag_name == "details":
            with self._use_details(tag, doc):
                self._walk(tag, doc)
        return added_refs

    @staticmethod
    def _is_form_container(tag: Tag) -> bool:
        classes = tag.get("class")
        if not classes:
            return False
        class_values = [classes] if isinstance(classes, str) else classes
        return _FORM_CONTAINER_CLASS in class_values

    def _nearest_form_container_ancestor(self, tag: Tag) -> Optional[Tag]:
        for parent in tag.parents:
            if isinstance(parent, Tag) and self._is_form_container(parent):
                return parent
        return None

    def _is_tag_in_current_form_scope(self, tag: Tag, form_tag: Tag) -> bool:
        return self._nearest_form_container_ancestor(tag) is form_tag

    @staticmethod
    def _is_form_semantic_id(tag_id: Optional[str]) -> bool:
        if not tag_id:
            return False
        return bool(
            _FORM_KEY_ID_RE.match(tag_id)
            or _FORM_MARKER_ID_RE.match(tag_id)
            or _FORM_VALUE_ID_RE.match(tag_id)
        )

    def _should_flatten_info_text(self, tag: Tag) -> bool:
        if "info-text" not in self._get_tag_classes(tag):
            return False
        return not self._is_form_semantic_id(self._get_html_id(tag))

    def _emit_flattened_text_tag(
        self, tag: Tag, doc: DoclingDocument
    ) -> Optional[RefItem]:
        # Keep full textual payload of info-text blocks even when descendants
        # share ids with key/value tags consumed elsewhere in the same form.
        text_raw = self.get_text(tag)
        _, text_clean = self._normalize_form_text(text_raw)
        if not text_clean:
            return None
        prov = self._make_text_prov(
            text=text_clean,
            tag=tag,
        )
        text_item = doc.add_text(
            parent=self.parents[self.level],
            label=DocItemLabel.TEXT,
            text=text_clean,
            orig=text_raw,
            content_layer=self.content_layer,
            prov=prov,
        )
        return text_item.get_ref()

    def _ensure_tag_html_id(self, tag: Tag) -> str:
        existing = self._get_html_id(tag)
        if existing is not None:
            return existing
        self._generated_html_id_counter += 1
        generated = f"docling_auto_input_{self._generated_html_id_counter}"
        tag["id"] = generated
        return generated

    @staticmethod
    def _is_value_in_key_scope(key_tag: Tag, value_tag: Tag) -> bool:
        if key_tag is value_tag:
            return True
        if any(parent is key_tag for parent in value_tag.parents):
            return True
        key_parent = key_tag.parent
        value_parent = value_tag.parent
        if key_parent is not None and key_parent is value_parent:
            return True
        return False

    @staticmethod
    def _dom_distance_between_tags(left_tag: Tag, right_tag: Tag) -> int:
        if left_tag is right_tag:
            return 0

        left_chain: list[Tag] = [left_tag]
        left_chain.extend(
            parent for parent in left_tag.parents if isinstance(parent, Tag)
        )
        right_chain: list[Tag] = [right_tag]
        right_chain.extend(
            parent for parent in right_tag.parents if isinstance(parent, Tag)
        )

        left_positions = {id(tag): idx for idx, tag in enumerate(left_chain)}
        best_distance: Optional[int] = None
        for right_idx, right_ancestor in enumerate(right_chain):
            left_idx = left_positions.get(id(right_ancestor))
            if left_idx is None:
                continue
            distance = left_idx + right_idx
            if best_distance is None or distance < best_distance:
                best_distance = distance

        return best_distance if best_distance is not None else 10_000

    def _select_form_value_entries(
        self,
        key_tag: Optional[Tag],
        marker_entries: list[tuple[int, Tag]],
        value_entries: list[tuple[Optional[int], int, Tag]],
    ) -> list[tuple[Optional[int], int, Tag]]:
        if not value_entries:
            return []

        anchor_tag: Optional[Tag] = None
        if key_tag is not None:
            anchor_tag = key_tag
        elif marker_entries:
            anchor_tag = marker_entries[0][1]

        grouped_entries: dict[
            tuple[str, int], list[tuple[Optional[int], int, Tag]]
        ] = {}
        for value_index, dom_order, value_tag in value_entries:
            group_key = (
                ("idx", value_index) if value_index is not None else ("dom", dom_order)
            )
            grouped_entries.setdefault(group_key, []).append(
                (value_index, dom_order, value_tag)
            )

        selected_entries: list[tuple[Optional[int], int, Tag]] = []
        for entries in grouped_entries.values():
            ranked_entries = sorted(
                entries,
                key=lambda entry: (
                    (
                        0
                        if (
                            key_tag is not None
                            and self._is_value_in_key_scope(key_tag, entry[2])
                        )
                        else 1
                    )
                    if key_tag is not None
                    else 0,
                    (
                        self._dom_distance_between_tags(anchor_tag, entry[2])
                        if anchor_tag is not None
                        else 0
                    ),
                    (
                        0
                        if (
                            entry[2].name in {"input", "select", "textarea"}
                            or self._is_checkbox_like_tag(entry[2])
                        )
                        else 1
                    ),
                    entry[1],
                ),
            )
            selected_entries.append(ranked_entries[0])

        selected_entries.sort(
            key=lambda entry: (
                entry[0] is None,
                entry[0] if entry[0] is not None else entry[1],
                entry[1],
            )
        )
        return selected_entries

    @staticmethod
    def _get_table_cell(tag: Tag) -> Optional[Tag]:
        parent_cell = tag.find_parent(["td", "th"])
        return parent_cell if isinstance(parent_cell, Tag) else None

    @staticmethod
    def _is_bbox_within_any_table(
        value_bbox: BoundingBox, table_bboxes: list[BoundingBox], threshold: float = 0.9
    ) -> bool:
        for table_bbox in table_bboxes:
            if value_bbox.intersection_over_self(table_bbox) >= threshold:
                return True
        return False

    def _should_ignore_table_kv_link(
        self, key_tag: Tag, value_tag: Tag, table_bboxes: list[BoundingBox]
    ) -> bool:
        key_table = key_tag.find_parent("table")
        value_table = value_tag.find_parent("table")
        key_cell = self._get_table_cell(key_tag)
        value_cell = self._get_table_cell(value_tag)
        if key_table is not None or value_table is not None:
            if key_table is not value_table:
                return True
            if key_cell is None or value_cell is None:
                return True
            if key_cell is value_cell:
                return False
            key_row = key_cell.find_parent("tr")
            value_row = value_cell.find_parent("tr")
            if key_row is not None and key_row is value_row:
                return False
            if key_cell.parent is not None and key_cell.parent is value_cell.parent:
                return False
            if (
                self._dom_distance_between_tags(key_cell, value_cell) <= 4
                and key_table is value_table
            ):
                return False
            if key_cell is not value_cell:
                return True

        if key_table is None and value_table is None and table_bboxes:
            value_rendered = self._get_rendered_bbox_for_tag(value_tag)
            if value_rendered and self._is_bbox_within_any_table(
                value_rendered.bbox, table_bboxes
            ):
                return True

        return False

    @staticmethod
    def _extract_text_excluding_ids(tag: Tag, excluded_ids: set[str]) -> str:
        def _extract(node: PageElement) -> list[str]:
            if isinstance(node, NavigableString):
                return [str(node)]
            if isinstance(node, Tag):
                node_id = node.get("id")
                if node_id and node_id in excluded_ids:
                    return []
                parts: list[str] = []
                for child in node:
                    parts.extend(_extract(child))
                if node.name in {"p", "li"}:
                    parts.append(" ")
                return parts
            return []

        return "".join(_extract(tag))

    @staticmethod
    def _extract_direct_text(tag: Tag) -> str:
        parts: list[str] = []
        for child in tag.contents:
            if isinstance(child, NavigableString):
                parts.append(str(child))
        return "".join(parts)

    @staticmethod
    def _normalize_form_text(text: str) -> tuple[str, str]:
        raw = re.sub(r"\s+", " ", text).strip()
        return raw, HTMLDocumentBackend._clean_unicode(raw)

    @staticmethod
    def _infer_form_value_kind(value_tag: Tag) -> Literal["read_only", "fillable"]:
        if HTMLDocumentBackend._is_checkbox_like_tag(value_tag):
            return "fillable"
        if (
            value_tag.find(
                lambda item: isinstance(item, Tag)
                and HTMLDocumentBackend._is_checkbox_like_tag(item)
            )
            is not None
        ):
            return "fillable"

        classes = HTMLDocumentBackend._get_tag_classes(value_tag)
        fillable_class_hints = {
            "input",
            "input-box",
            "input-field",
            "input_field",
            "text-input",
            "text-box",
            "textbox",
            "form-control",
        }
        if classes & fillable_class_hints:
            return "fillable"
        if any(
            class_name.endswith(("-input", "_input", "-input-box"))
            for class_name in classes
        ):
            return "fillable"

        if value_tag.name in {"input", "select", "textarea"}:
            return "fillable"
        if value_tag.find(["input", "select", "textarea"]) is not None:
            return "fillable"
        return "read_only"

    @staticmethod
    def _get_tag_classes(tag: Tag) -> set[str]:
        classes = tag.get("class")
        if not classes:
            return set()
        if isinstance(classes, str):
            return {classes}
        return {str(value) for value in classes if isinstance(value, str)}

    @staticmethod
    def _has_inline_display_style(tag: Tag) -> bool:
        style_attr = tag.get("style")
        if not isinstance(style_attr, str):
            return False
        display_match = re.search(r"display\s*:\s*([^;]+)", style_attr, flags=re.I)
        if display_match is None:
            return False
        display_value = display_match.group(1).strip().lower()
        return display_value.startswith("inline") or display_value == "contents"

    def _should_buffer_tag_text_inline(self, tag: Tag) -> bool:
        tag_name = tag.name.lower()
        if tag_name in _INLINE_HTML_TAGS:
            return True
        # Treat explicit inline-styled divs like inline wrappers.
        if tag_name == "div" and self._has_inline_display_style(tag):
            return True
        return False

    @staticmethod
    def _is_input_checkbox_or_radio_tag(tag: Tag) -> bool:
        if tag.name != "input":
            return False
        input_type = str(tag.get("type", "")).strip().lower()
        return input_type in {"checkbox", "radio"}

    @staticmethod
    def _is_generic_text_input_candidate(tag: Tag) -> bool:
        if tag.name != "input":
            return False
        input_type = str(tag.get("type", "")).strip().lower()
        if input_type in {
            "hidden",
            "checkbox",
            "radio",
            "submit",
            "button",
            "reset",
            "file",
            "image",
            "color",
            "range",
            "date",
            "datetime-local",
            "month",
            "time",
            "week",
        }:
            return False
        return True

    @staticmethod
    def _is_custom_checkbox_tag(tag: Tag) -> bool:
        return bool(
            HTMLDocumentBackend._get_tag_classes(tag) & _CUSTOM_CHECKBOX_CLASSES
        )

    @staticmethod
    def _is_checkbox_like_tag(tag: Tag) -> bool:
        return HTMLDocumentBackend._is_input_checkbox_or_radio_tag(
            tag
        ) or HTMLDocumentBackend._is_custom_checkbox_tag(tag)

    @staticmethod
    def _extract_text_excluding_tag_obj_ids(
        tag: Tag, excluded_obj_ids: set[int]
    ) -> str:
        def _extract(node: PageElement) -> list[str]:
            if isinstance(node, NavigableString):
                return [str(node)]
            if isinstance(node, Tag):
                if id(node) in excluded_obj_ids:
                    return []
                parts: list[str] = []
                for child in node.contents:
                    parts.extend(_extract(child))
                if node.name in {"p", "li", "div", "label", "span", "td", "th"}:
                    parts.append(" ")
                return parts
            return []

        return "".join(_extract(tag))

    @staticmethod
    def _has_direct_checkbox_like_child(tag: Tag) -> bool:
        for child in tag.find_all(recursive=False):
            if isinstance(child, Tag) and HTMLDocumentBackend._is_checkbox_like_tag(
                child
            ):
                return True
        return False

    def _is_checkbox_label_container(self, tag: Tag) -> bool:
        classes = self._get_tag_classes(tag)
        if not (classes & _CHECKBOX_CONTAINER_CLASSES):
            return False
        return self._has_direct_checkbox_like_child(tag)

    def _is_checkbox_label_tag(self, tag: Tag) -> bool:
        if self._is_checkbox_like_tag(tag):
            return False
        if "checkbox-label" in self._get_tag_classes(tag):
            return True
        parent = tag.parent
        if isinstance(parent, Tag) and self._is_checkbox_label_container(parent):
            return True
        return False

    @staticmethod
    def _normalize_checkbox_text(text: str) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return ""
        if compact.lower() in _CHECKBOX_MARK_TEXTS:
            return ""
        return HTMLDocumentBackend._clean_unicode(compact)

    @staticmethod
    def _is_checkbox_checked(tag: Tag) -> bool:
        if HTMLDocumentBackend._is_input_checkbox_or_radio_tag(tag):
            if tag.has_attr("checked"):
                return True
            aria_checked = str(tag.get("aria-checked", "")).strip().lower()
            return aria_checked in {"true", "1", "yes", "on"}

        classes = HTMLDocumentBackend._get_tag_classes(tag)
        if "checked" in classes:
            return True

        aria_checked = str(tag.get("aria-checked", "")).strip().lower()
        if aria_checked in {"true", "1", "yes", "on"}:
            return True

        data_checked = str(tag.get("data-checked", "")).strip().lower()
        if data_checked in {"true", "1", "yes", "on"}:
            return True

        text = re.sub(r"\s+", "", HTMLDocumentBackend.get_text(tag))
        return text.lower() in _CHECKBOX_MARK_TEXTS

    @staticmethod
    def _get_checkbox_label_for_tag(tag: Tag) -> Optional[DocItemLabel]:
        if not HTMLDocumentBackend._is_checkbox_like_tag(tag):
            return None
        return (
            DocItemLabel.CHECKBOX_SELECTED
            if HTMLDocumentBackend._is_checkbox_checked(tag)
            else DocItemLabel.CHECKBOX_UNSELECTED
        )

    def _extract_checkbox_text_and_consumed_label_obj_ids(  # noqa: C901
        self, checkbox_tag: Tag
    ) -> tuple[str, set[int], list[Tag]]:
        consumed_tag_obj_ids: set[int] = set()
        consumed_label_tags: list[Tag] = []
        parent = checkbox_tag.parent if isinstance(checkbox_tag.parent, Tag) else None
        seen_label_obj_ids: set[int] = set()

        def _add_label_tag(label_tag: Tag) -> None:
            label_obj_id = id(label_tag)
            if label_obj_id in seen_label_obj_ids:
                return
            seen_label_obj_ids.add(label_obj_id)
            consumed_tag_obj_ids.add(label_obj_id)
            consumed_label_tags.append(label_tag)

        def _label_texts(tags: list[Tag]) -> list[str]:
            texts: list[str] = []
            for label_tag in tags:
                raw = self.get_text(label_tag)
                normalized = self._normalize_checkbox_text(raw)
                if normalized:
                    texts.append(normalized)
            return texts

        # Native checkbox/radio with explicit <label for="..."> in the same option container.
        if checkbox_tag.name == "input":
            input_id = self._get_html_id(checkbox_tag)
            if input_id:
                # Collect labels explicitly pointing to this checkbox/radio.
                if self.soup is not None:
                    for label_tag in self.soup.find_all(
                        "label", attrs={"for": input_id}
                    ):
                        if isinstance(label_tag, Tag):
                            _add_label_tag(label_tag)
                # Backward-compatible local search (same parent container).
                if parent is not None:
                    for sibling in parent.find_all("label", recursive=False):
                        if sibling.get("for") == input_id:
                            _add_label_tag(sibling)

            # Input wrapped by a <label> ... <input ...> ... </label>
            wrapping_label = checkbox_tag.find_parent("label")
            if isinstance(wrapping_label, Tag):
                _add_label_tag(wrapping_label)

            # ARIA linkage to label element(s), if present.
            aria_labelledby = checkbox_tag.get("aria-labelledby")
            if isinstance(aria_labelledby, str) and self.soup is not None:
                for labelledby_id in aria_labelledby.split():
                    labelledby_tag = self.soup.find(id=labelledby_id)
                    if isinstance(labelledby_tag, Tag):
                        _add_label_tag(labelledby_tag)

            explicit_texts = _label_texts(consumed_label_tags)
            if explicit_texts:
                return (
                    " ".join(explicit_texts),
                    consumed_tag_obj_ids,
                    consumed_label_tags,
                )

        if parent is not None:
            parent_classes = self._get_tag_classes(parent)

            # Pattern: checkbox + sibling span.checkbox-label inside .checkbox-container
            if "checkbox-container" in parent_classes:
                label_texts: list[str] = []
                for sibling in parent.find_all(recursive=False):
                    if not isinstance(sibling, Tag) or sibling is checkbox_tag:
                        continue
                    if "checkbox-label" not in self._get_tag_classes(sibling):
                        continue
                    text = self._normalize_checkbox_text(self.get_text(sibling))
                    _add_label_tag(sibling)
                    if text:
                        label_texts.append(text)
                if label_texts:
                    return (
                        " ".join(label_texts),
                        consumed_tag_obj_ids,
                        consumed_label_tags,
                    )

            # Pattern: checkbox + neighbour text in .checkbox-item or parent text in .checkbox-option/.option
            if parent_classes & {"checkbox-item", "checkbox-option", "option"}:
                has_direct_label_text = any(
                    isinstance(child, NavigableString)
                    and bool(self._normalize_checkbox_text(str(child)))
                    for child in parent.contents
                )
                for sibling in parent.find_all(recursive=False):
                    if not isinstance(sibling, Tag) or sibling is checkbox_tag:
                        continue
                    if self._is_checkbox_like_tag(sibling):
                        continue
                    _add_label_tag(sibling)
                raw = self._extract_text_excluding_tag_obj_ids(
                    parent, {id(checkbox_tag)}
                )
                text = self._normalize_checkbox_text(raw)
                if text:
                    if has_direct_label_text:
                        _add_label_tag(parent)
                    return text, consumed_tag_obj_ids, consumed_label_tags

        # Last fallback: custom checkbox text inside the element itself.
        if checkbox_tag.name != "input":
            raw = self.get_text(checkbox_tag)
            text = self._normalize_checkbox_text(raw)
            if text:
                return text, consumed_tag_obj_ids, consumed_label_tags

        return "", consumed_tag_obj_ids, consumed_label_tags

    @staticmethod
    def _extract_input_like_text(tag: Tag) -> str:
        if tag.name == "input":
            for attr in ("value", "placeholder", "name"):
                val = tag.get(attr)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            return ""

        if tag.name == "textarea":
            return HTMLDocumentBackend.get_text(tag).strip()

        if tag.name == "select":
            selected_option: Optional[Tag] = None
            for option in tag.find_all("option"):
                if option.has_attr("selected"):
                    selected_option = option
                    break
            if selected_option is None:
                first_option = tag.find("option")
                if isinstance(first_option, Tag):
                    selected_option = first_option
            if selected_option is not None:
                selected_text = HTMLDocumentBackend.get_text(selected_option).strip()
                if selected_text:
                    return selected_text
                selected_value = selected_option.get("value")
                if isinstance(selected_value, str) and selected_value.strip():
                    return selected_value.strip()
            select_value = tag.get("value")
            if isinstance(select_value, str) and select_value.strip():
                return select_value.strip()
            return ""

        return ""

    def _extract_form_value_text(self, value_tag: Tag) -> str:
        # Input elements carry their user-visible content in attributes, not inner text.
        if value_tag.name in {"input", "textarea", "select"}:
            return self._extract_input_like_text(value_tag)

        if value_tag.find(["input", "textarea", "select"]) is None:
            return HTMLDocumentBackend.get_text(value_tag)

        parts: list[str] = []

        def _collect(node: PageElement) -> None:
            if isinstance(node, NavigableString):
                parts.append(str(node))
                return
            if not isinstance(node, Tag):
                return
            if self._is_invisible_tag(node):
                return
            if node.name in {"input", "textarea", "select"}:
                input_text = self._extract_input_like_text(node)
                if input_text:
                    parts.append(input_text)
                parts.append(" ")
                return
            for child in node.contents:
                _collect(child)
            if node.name in {"p", "li", "div", "label", "span", "td", "th", "tr", "br"}:
                parts.append(" ")

        _collect(value_tag)
        return "".join(parts)

    @staticmethod
    def _extract_form_marker_text(marker_tag: Tag) -> str:
        return HTMLDocumentBackend.get_text(marker_tag)

    @staticmethod
    def _is_form_semantic_tag(tag: Tag) -> bool:
        return HTMLDocumentBackend._is_form_semantic_id(
            HTMLDocumentBackend._get_html_id(tag)
        )

    def _merge_adjacent_form_values(
        self, values: list[_ExtractedFormValue]
    ) -> list[_ExtractedFormValue]:
        if len(values) <= 1:
            return values

        merged_values: list[_ExtractedFormValue] = []
        current = values[0]
        current_is_char_run = len(current.text.strip()) == 1
        current_source_tag_ids: list[str] = [
            tag_id for tag_id in [self._get_tag_id(current.tag)] if tag_id is not None
        ]

        def _flush_current() -> None:
            nonlocal current, current_source_tag_ids
            if len(current_source_tag_ids) > 1 or (
                current.prov is None and current_source_tag_ids
            ):
                current.prov = self._make_text_prov_for_source_tag_ids(
                    text=current.text,
                    tag=current.tag,
                    source_tag_ids=current_source_tag_ids,
                )
            merged_values.append(current)

        for next_value in values[1:]:
            next_text = next_value.text.strip()
            next_is_char = len(next_text) == 1
            can_merge_text = (
                current_is_char_run
                and next_is_char
                and current.checkbox_label is None
                and next_value.checkbox_label is None
                and current.kind == next_value.kind
            )
            left_source = self._get_tag_id(current.tag)
            right_source = self._get_tag_id(next_value.tag)
            can_merge_bbox = (
                left_source is not None
                and right_source is not None
                and self._are_source_tag_ids_inline_neighbors(left_source, right_source)
            )
            can_merge_dom = (
                current.tag.parent is next_value.tag.parent
                and current_is_char_run
                and next_is_char
            )

            if can_merge_text and (can_merge_bbox or can_merge_dom):
                current.text = f"{current.text}{next_value.text}"
                current.orig = f"{current.orig}{next_value.orig}"
                current.order = min(current.order, next_value.order)
                if right_source is not None:
                    current_source_tag_ids.append(right_source)
                current_is_char_run = True
            else:
                _flush_current()
                current = next_value
                current_is_char_run = len(current.text.strip()) == 1
                current_source_tag_ids = [
                    tag_id
                    for tag_id in [self._get_tag_id(current.tag)]
                    if tag_id is not None
                ]

        _flush_current()
        return merged_values

    def _emit_custom_checkbox(
        self, checkbox_tag: Tag, doc: DoclingDocument
    ) -> Optional[RefItem]:
        if self._is_suppressed_tag(checkbox_tag):
            return None
        checkbox_label = self._get_checkbox_label_for_tag(checkbox_tag)
        if checkbox_label is None:
            return None
        (
            checkbox_text,
            _,
            checkbox_label_tags,
        ) = self._extract_checkbox_text_and_consumed_label_obj_ids(checkbox_tag)
        prov = self._make_checkbox_with_label_prov(
            text=checkbox_text,
            checkbox_tag=checkbox_tag,
            label_tags=checkbox_label_tags,
        )
        checkbox_item = doc.add_text(
            parent=self.parents[self.level],
            label=checkbox_label,
            text=checkbox_text,
            content_layer=self.content_layer,
            formatting=self._formatting,
            hyperlink=self.hyperlink,
            prov=prov,
        )
        return checkbox_item.get_ref()

    @contextmanager
    def _suppress_tag_ids(self, tag_ids: set[str]):
        if not tag_ids:
            yield None
            return
        self._suppressed_tag_ids_stack.append(tag_ids)
        try:
            yield None
        finally:
            self._suppressed_tag_ids_stack.pop()

    @contextmanager
    def _suppress_tag_obj_ids(self, tag_obj_ids: set[int]):
        if not tag_obj_ids:
            yield None
            return
        self._suppressed_tag_obj_ids_stack.append(tag_obj_ids)
        try:
            yield None
        finally:
            self._suppressed_tag_obj_ids_stack.pop()

    def _is_suppressed_tag(self, tag: Tag) -> bool:
        if self._is_invisible_tag(tag):
            return True
        if self._is_tag_outside_capture_area(tag):
            return True
        tag_obj_id = id(tag)
        if any(tag_obj_id in obj_ids for obj_ids in self._suppressed_tag_obj_ids_stack):
            return True
        tag_ids = set()
        if html_id := self._get_html_id(tag):
            tag_ids.add(html_id)
        if docling_id := self._get_tag_id(tag):
            tag_ids.add(docling_id)
        if not tag_ids:
            return False
        return any(bool(ids & tag_ids) for ids in self._suppressed_tag_ids_stack)

    @contextmanager
    def _use_form_fields_by_key_id(
        self, fields_by_key_id: dict[str, _ExtractedFormField]
    ):
        self._form_fields_by_key_id_stack.append(dict(fields_by_key_id))
        try:
            yield None
        finally:
            self._form_fields_by_key_id_stack.pop()

    def _has_pending_form_field_in_subtree(self, tag: Tag) -> bool:
        if not self._form_fields_by_key_id_stack:
            return False
        field_map = self._form_fields_by_key_id_stack[-1]
        if not field_map:
            return False

        tag_id = self._get_html_id(tag)
        if tag_id is not None and tag_id in field_map:
            return True

        for descendant in tag.find_all(True):
            descendant_id = descendant.get("id")
            if isinstance(descendant_id, str) and descendant_id in field_map:
                return True
        return False

    def _consume_form_field_for_tag(self, tag: Tag) -> Optional[_ExtractedFormField]:
        tag_id = self._get_html_id(tag)
        if tag_id is None:
            return None
        for field_map in reversed(self._form_fields_by_key_id_stack):
            field = field_map.pop(tag_id, None)
            if field is not None:
                for mapped_tag_id, mapped_field in list(field_map.items()):
                    if mapped_field is field:
                        field_map.pop(mapped_tag_id, None)
                return field
        return None

    def _consume_form_fields_in_subtree(self, tag: Tag) -> list[_ExtractedFormField]:
        if not self._form_fields_by_key_id_stack:
            return []
        field_map = self._form_fields_by_key_id_stack[-1]
        extracted_fields: list[_ExtractedFormField] = []
        consumed_field_ids: set[int] = set()
        for _, field in list(field_map.items()):
            field_obj_id = id(field)
            if field_obj_id in consumed_field_ids:
                continue
            field_tags: list[Tag] = []
            if field.key_tag is not None:
                field_tags.append(field.key_tag)
            if field.marker is not None:
                field_tags.append(field.marker.tag)
            field_tags.extend(value.tag for value in field.values)
            if not field_tags:
                continue
            if any(
                field_tag is tag or any(parent is tag for parent in field_tag.parents)
                for field_tag in field_tags
            ):
                extracted_fields.append(field)
                consumed_field_ids.add(field_obj_id)
                for pop_tag_id, pop_field in list(field_map.items()):
                    if pop_field is field:
                        field_map.pop(pop_tag_id, None)
        return extracted_fields

    def _is_lonely_key_covered_by_table(self, key_tag: Tag) -> bool:
        key_tag_id = self._get_html_id(key_tag)
        if key_tag_id is None:
            return False

        table_cell = self._get_table_cell(key_tag)
        if table_cell is None:
            return False

        remaining_raw = self._extract_text_excluding_ids(table_cell, {key_tag_id})
        _, remaining_clean = self._normalize_form_text(remaining_raw)
        if remaining_clean:
            return False

        for descendant in table_cell.descendants:
            if descendant is key_tag:
                continue
            if isinstance(descendant, Tag):
                if any(parent is key_tag for parent in descendant.parents):
                    continue
                return False
            if isinstance(descendant, NavigableString):
                if any(parent is key_tag for parent in descendant.parents):
                    continue
                if str(descendant).strip():
                    return False

        return True

    def _iter_form_scan_parents(self, tag: Tag, max_depth: int = 3) -> list[Tag]:
        parents: list[Tag] = []
        current: Optional[Tag] = tag.parent if isinstance(tag.parent, Tag) else None
        depth = 0
        while current is not None and depth < max_depth:
            parents.append(current)
            if self._is_form_container(current) or current.name in {
                "table",
                "tbody",
                "thead",
                "tfoot",
                "tr",
            }:
                break
            current = current.parent if isinstance(current.parent, Tag) else None
            depth += 1
        return parents

    def _is_simple_extra_text_candidate(self, tag: Tag) -> bool:
        if self._is_invisible_tag(tag):
            return False
        if self._is_form_semantic_id(self._get_html_id(tag)):
            return False
        classes = self._get_tag_classes(tag)
        if any(
            class_name in {"section-title", "grid-header", "line-items-header"}
            or class_name in {"title", "summary-title"}
            or class_name.endswith(("-header", "_header", "-title", "_title"))
            for class_name in classes
        ):
            return False
        if tag.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            return False
        if self._is_checkbox_like_tag(tag):
            return True
        if (
            tag.find(
                lambda item: isinstance(item, Tag)
                and item is not tag
                and self._is_form_semantic_id(self._get_html_id(item))
            )
            is not None
        ):
            return False

        direct_tag_children = [
            child for child in tag.find_all(recursive=False) if isinstance(child, Tag)
        ]
        has_direct_text = any(
            isinstance(child, NavigableString) and bool(str(child).strip())
            for child in tag.contents
        )
        if len(direct_tag_children) > 3 and not has_direct_text:
            return False

        raw = self._extract_text_excluding_tag_obj_ids(tag, set())
        _, clean = self._normalize_form_text(raw)
        if not clean:
            return False
        if len(clean) > 140 and len(direct_tag_children) > 1:
            return False
        return True

    def _add_field_item_from_extracted(
        self,
        field: _ExtractedFormField,
        doc: DoclingDocument,
        parent: Optional[Union[DocItem, GroupItem]],
    ) -> list[RefItem]:
        refs: list[RefItem] = []
        doc_with_fields = cast(Any, doc)
        field_item = doc_with_fields.add_field_item(
            parent=parent,
            content_layer=self.content_layer,
        )
        refs.append(field_item.get_ref())

        parts: list[tuple[int, Literal["key", "marker", "value", "text"], Any]] = []
        if field.key_tag is not None and field.key_text:
            parts.append((field.key_order, "key", field))
        if field.marker is not None:
            parts.append((field.marker.order, "marker", field.marker))
        for value in field.values:
            parts.append((value.order, "value", value))
        for extra_text in field.extra_texts:
            parts.append((extra_text.order, "text", extra_text))

        for _, part_type, payload in sorted(parts, key=lambda part: part[0]):
            if part_type == "key":
                field_key = doc_with_fields.add_field_key(
                    text=field.key_text,
                    orig=field.key_orig,
                    prov=field.key_prov,
                    parent=field_item,
                    content_layer=self.content_layer,
                )
                refs.append(field_key.get_ref())
            elif part_type == "marker":
                marker = cast(_ExtractedFormMarker, payload)
                marker_item = doc.add_text(
                    label=DocItemLabel.MARKER,
                    text=marker.text,
                    orig=marker.orig,
                    prov=marker.prov,
                    parent=field_item,
                    content_layer=self.content_layer,
                )
                refs.append(marker_item.get_ref())
            elif part_type == "text":
                extra_text = cast(_ExtractedFormText, payload)
                text_item = doc.add_text(
                    label=extra_text.label,
                    text=extra_text.text,
                    orig=extra_text.orig,
                    prov=extra_text.prov,
                    parent=field_item,
                    content_layer=self.content_layer,
                )
                refs.append(text_item.get_ref())
            else:
                value = cast(_ExtractedFormValue, payload)
                if value.checkbox_label is not None:
                    checkbox_item = doc.add_text(
                        label=value.checkbox_label,
                        text=value.text,
                        orig=value.orig,
                        prov=value.prov,
                        parent=field_item,
                        content_layer=self.content_layer,
                    )
                    refs.append(checkbox_item.get_ref())
                else:
                    field_value = doc_with_fields.add_field_value(
                        text=value.text,
                        orig=value.orig,
                        prov=value.prov,
                        parent=field_item,
                        content_layer=self.content_layer,
                        kind=value.kind,
                    )
                    refs.append(field_value.get_ref())

        return refs

    def _extract_form_region(self, form_tag: Tag) -> Optional[_ExtractedFormRegion]:  # noqa: C901
        key_tags: dict[str, Tag] = {}
        key_orders: dict[str, int] = {}
        first_order_by_key: dict[str, int] = {}
        markers_by_key: dict[str, list[tuple[int, Tag]]] = {}
        values_by_key: dict[str, list[tuple[Optional[int], int, Tag]]] = {}
        all_tags = [
            tag
            for tag in cast(list[Tag], form_tag.find_all(True))
            if self._is_tag_in_current_form_scope(tag, form_tag)
            and not self._is_invisible_tag(tag)
        ]
        tag_order_by_obj_id: dict[int, int] = {
            id(tag): idx for idx, tag in enumerate(all_tags, start=1)
        }

        for tag in all_tags:
            tag_id = tag.get("id")
            if not isinstance(tag_id, str):
                continue
            dom_order = tag_order_by_obj_id[id(tag)]

            value_match = _FORM_VALUE_ID_RE.match(tag_id)
            if value_match:
                key_id = value_match.group("key_id")
                value_id = value_match.group("value_id")
                value_index = int(value_id) if value_id.isdigit() else None
                values_by_key.setdefault(key_id, []).append(
                    (value_index, dom_order, tag)
                )
                first_order_by_key.setdefault(key_id, dom_order)
                continue

            marker_match = _FORM_MARKER_ID_RE.match(tag_id)
            if marker_match:
                key_id = marker_match.group("key_id")
                markers_by_key.setdefault(key_id, []).append((dom_order, tag))
                first_order_by_key.setdefault(key_id, dom_order)
                continue

            key_match = _FORM_KEY_ID_RE.match(tag_id)
            if key_match:
                key_id = key_match.group("key_id")
                if key_id not in key_tags:
                    key_tags[key_id] = tag
                    key_orders[key_id] = dom_order
                first_order_by_key.setdefault(key_id, dom_order)

        fields: list[_ExtractedFormField] = []
        consumed_tag_ids: set[str] = set()

        table_bboxes: list[BoundingBox] = []
        if self._rendered_bbox_by_id:
            for table_tag in form_tag.find_all("table"):
                if isinstance(table_tag, Tag):
                    rendered = self._get_rendered_bbox_for_tag(table_tag)
                    if rendered is not None:
                        table_bboxes.append(rendered.bbox)

        key_ids_in_order = sorted(
            first_order_by_key.keys(), key=lambda key_id: first_order_by_key[key_id]
        )
        assigned_extra_tag_obj_ids: set[int] = set()

        for key_id in key_ids_in_order:
            key_tag = key_tags.get(key_id)

            marker_entries = list(markers_by_key.get(key_id, []))
            if key_tag is not None:
                in_scope_markers = [
                    entry
                    for entry in marker_entries
                    if self._is_value_in_key_scope(key_tag, entry[1])
                ]
                if in_scope_markers:
                    marker_entries = in_scope_markers
            marker_entries.sort(key=lambda entry: entry[0])

            value_entries = list(values_by_key.get(key_id, []))
            if key_tag is not None:
                value_entries = [
                    entry
                    for entry in value_entries
                    if not self._should_ignore_table_kv_link(
                        key_tag, entry[2], table_bboxes
                    )
                ]
            value_entries = self._select_form_value_entries(
                key_tag=key_tag,
                marker_entries=marker_entries,
                value_entries=value_entries,
            )
            if (
                key_tag is not None
                and self._is_checkbox_like_tag(key_tag)
                and not any(entry[2] is key_tag for entry in value_entries)
            ):
                key_as_value_order = key_orders.get(
                    key_id, first_order_by_key.get(key_id, 0)
                )
                value_entries.append((0, key_as_value_order, key_tag))
                value_entries.sort(
                    key=lambda entry: (
                        entry[0] is None,
                        entry[0] if entry[0] is not None else entry[1],
                        entry[1],
                    )
                )

            marker: Optional[_ExtractedFormMarker] = None
            if marker_entries:
                marker_tag = marker_entries[0][1]
                marker_text_raw = self._extract_form_marker_text(marker_tag)
                marker_orig, marker_text = self._normalize_form_text(marker_text_raw)
                if marker_text:
                    marker = _ExtractedFormMarker(
                        tag=marker_tag,
                        order=marker_entries[0][0],
                        orig=marker_orig,
                        text=marker_text,
                        prov=self._make_text_prov(text=marker_text, tag=marker_tag),
                    )
                    marker_tag_id = self._get_html_id(marker_tag)
                    if marker_tag_id is not None:
                        consumed_tag_ids.add(marker_tag_id)
                else:
                    marker_tag_id = self._get_html_id(marker_tag)
                    if marker_tag_id is not None:
                        consumed_tag_ids.add(marker_tag_id)

            value_tags = [entry[2] for entry in value_entries]
            excluded_ids = {
                tag_id
                for tag_id in (tag.get("id") for tag in value_tags)
                if isinstance(tag_id, str)
            }
            if marker is not None:
                marker_id = self._get_html_id(marker.tag)
                if marker_id is not None:
                    excluded_ids.add(marker_id)

            key_orig = ""
            key_text = ""
            key_prov: Optional[ProvenanceItem] = None
            if key_tag is not None:
                if key_tag.name in {"input", "select", "textarea"}:
                    key_text_raw = self._extract_form_value_text(key_tag)
                else:
                    descendant_value_count = len(
                        [
                            descendant
                            for descendant in key_tag.find_all(True)
                            if self._is_form_semantic_id(self._get_html_id(descendant))
                            and _FORM_VALUE_ID_RE.match(
                                self._get_html_id(descendant) or ""
                            )
                        ]
                    )
                    if descendant_value_count > 3:
                        key_text_raw = self._extract_direct_text(key_tag)
                    else:
                        key_text_raw = self._extract_text_excluding_ids(
                            key_tag, excluded_ids
                        )
                key_orig, key_text = self._normalize_form_text(key_text_raw)
                key_prov = self._make_text_prov(text=key_text, tag=key_tag)
                if len(value_tags) > 8 and len(key_text) > 120:
                    key_orig = ""
                    key_text = ""
                    key_prov = None

            if key_tag is None and marker is None and not value_tags:
                continue
            if (
                key_tag is not None
                and marker is None
                and not key_text
                and not value_tags
            ):
                continue

            values: list[_ExtractedFormValue] = []
            for _, value_order, value_tag in value_entries:
                checkbox_label = self._get_checkbox_label_for_tag(value_tag)
                consumed_label_tag_obj_ids: set[int] = set()
                checkbox_label_tags: list[Tag] = []
                value_kind = self._infer_form_value_kind(value_tag)
                if checkbox_label is not None:
                    (
                        value_text,
                        consumed_label_tag_obj_ids,
                        checkbox_label_tags,
                    ) = self._extract_checkbox_text_and_consumed_label_obj_ids(
                        value_tag
                    )
                    value_orig = value_text
                else:
                    value_text_raw = self._extract_form_value_text(value_tag)
                    value_orig, value_text = self._normalize_form_text(value_text_raw)
                values.append(
                    _ExtractedFormValue(
                        tag=value_tag,
                        order=value_order,
                        orig=value_orig,
                        text=value_text,
                        prov=(
                            self._make_checkbox_with_label_prov(
                                text=value_text,
                                checkbox_tag=value_tag,
                                label_tags=checkbox_label_tags,
                            )
                            if checkbox_label is not None
                            else self._make_text_prov(text=value_text, tag=value_tag)
                        ),
                        kind=value_kind,
                        checkbox_label=checkbox_label,
                        consumed_label_tag_obj_ids=consumed_label_tag_obj_ids,
                        checkbox_label_tags=checkbox_label_tags,
                    )
                )
                value_tag_id = self._get_html_id(value_tag)
                if value_tag_id is not None:
                    consumed_tag_ids.add(value_tag_id)
            values = self._merge_adjacent_form_values(values)

            if key_text and key_prov is None:
                if marker is not None and marker.prov is not None:
                    key_prov = marker.prov
                elif values and values[0].prov is not None:
                    key_prov = values[0].prov

            component_tag_obj_ids: set[int] = {id(value.tag) for value in values}
            component_tag_obj_ids.update(id(tag) for tag in value_tags)
            if key_tag is not None:
                component_tag_obj_ids.add(id(key_tag))
            if marker is not None:
                component_tag_obj_ids.add(id(marker.tag))
            consumed_label_obj_ids: set[int] = set()
            for value in values:
                consumed_label_obj_ids.update(value.consumed_label_tag_obj_ids)
            seen_extra_tag_obj_ids: set[int] = set()
            extra_texts: list[_ExtractedFormText] = []
            parent_tags_to_scan: list[Tag] = []
            if key_tag is not None:
                parent_tags_to_scan.extend(self._iter_form_scan_parents(key_tag))
            if marker is not None:
                parent_tags_to_scan.extend(self._iter_form_scan_parents(marker.tag))
            for value in values:
                parent_tags_to_scan.extend(self._iter_form_scan_parents(value.tag))

            seen_parent_tag_obj_ids: set[int] = set()
            unique_parent_tags_to_scan: list[Tag] = []
            for parent_tag in parent_tags_to_scan:
                parent_tag_obj_id = id(parent_tag)
                if parent_tag_obj_id in seen_parent_tag_obj_ids:
                    continue
                seen_parent_tag_obj_ids.add(parent_tag_obj_id)
                unique_parent_tags_to_scan.append(parent_tag)

            for parent_tag in unique_parent_tags_to_scan:
                component_direct_child_indices: list[int] = []
                direct_children = [
                    child
                    for child in parent_tag.find_all(recursive=False)
                    if isinstance(child, Tag)
                ]
                for idx, child in enumerate(direct_children):
                    child_obj_id = id(child)
                    if child_obj_id in component_tag_obj_ids:
                        component_direct_child_indices.append(idx)
                        continue
                    if any(
                        descendant_obj_id in component_tag_obj_ids
                        for descendant_obj_id in [
                            id(descendant) for descendant in child.find_all(True)
                        ]
                    ):
                        component_direct_child_indices.append(idx)

                if component_direct_child_indices:
                    min_idx = min(component_direct_child_indices)
                    max_idx = max(component_direct_child_indices)
                else:
                    min_idx = 0
                    max_idx = len(direct_children) - 1

                for idx, sibling_tag in enumerate(direct_children):
                    if idx < (min_idx - 1) or idx > (max_idx + 1):
                        continue
                    sibling_obj_id = id(sibling_tag)
                    if sibling_obj_id in component_tag_obj_ids:
                        continue
                    if self._is_invisible_tag(sibling_tag):
                        continue
                    if any(
                        id(descendant) in component_tag_obj_ids
                        for descendant in sibling_tag.find_all(True)
                    ):
                        continue
                    if sibling_obj_id in seen_extra_tag_obj_ids:
                        continue
                    if sibling_obj_id in assigned_extra_tag_obj_ids:
                        continue
                    if sibling_obj_id in consumed_label_obj_ids:
                        continue
                    sibling_html_id = self._get_html_id(sibling_tag)
                    if sibling_html_id is not None and self._is_form_semantic_id(
                        sibling_html_id
                    ):
                        continue
                    if not self._is_simple_extra_text_candidate(sibling_tag):
                        continue
                    sibling_text_raw = self._extract_text_excluding_tag_obj_ids(
                        sibling_tag, component_tag_obj_ids | consumed_label_obj_ids
                    )
                    sibling_orig, sibling_text = self._normalize_form_text(
                        sibling_text_raw
                    )
                    sibling_label = DocItemLabel.TEXT
                    if self._is_custom_checkbox_tag(sibling_tag):
                        sibling_label = (
                            DocItemLabel.CHECKBOX_SELECTED
                            if self._is_checkbox_checked(sibling_tag)
                            else DocItemLabel.CHECKBOX_UNSELECTED
                        )
                        sibling_text = self._normalize_checkbox_text(sibling_text_raw)
                        sibling_orig = sibling_text
                    if not sibling_text and sibling_label == DocItemLabel.TEXT:
                        continue
                    sibling_order = tag_order_by_obj_id.get(sibling_obj_id)
                    if sibling_order is None:
                        continue
                    extra_texts.append(
                        _ExtractedFormText(
                            tag=sibling_tag,
                            order=sibling_order,
                            orig=sibling_orig,
                            text=sibling_text,
                            prov=self._make_text_prov(
                                text=sibling_text, tag=sibling_tag
                            ),
                            label=sibling_label,
                        )
                    )
                    seen_extra_tag_obj_ids.add(sibling_obj_id)
                    assigned_extra_tag_obj_ids.add(sibling_obj_id)

            fields.append(
                _ExtractedFormField(
                    key_tag=key_tag,
                    key_order=key_orders.get(key_id, first_order_by_key.get(key_id, 0)),
                    key_orig=key_orig,
                    key_text=key_text,
                    key_prov=key_prov,
                    marker=marker,
                    values=values,
                    extra_texts=extra_texts,
                )
            )
            key_tag_id = self._get_html_id(key_tag)
            if key_tag_id is not None:
                consumed_tag_ids.add(key_tag_id)

        extracted_value_tag_obj_ids = {
            id(value.tag) for field in fields for value in field.values
        }
        generic_clusters: list[list[tuple[int, Tag]]] = []
        current_cluster: list[tuple[int, Tag]] = []
        current_parent_obj_id: Optional[int] = None
        previous_order: Optional[int] = None
        for tag in all_tags:
            if id(tag) in extracted_value_tag_obj_ids:
                continue
            if not self._is_generic_text_input_candidate(tag):
                continue
            if self._get_html_id(tag) is not None:
                continue
            tag_order = tag_order_by_obj_id.get(id(tag))
            if tag_order is None:
                continue
            parent_obj_id = id(tag.parent) if isinstance(tag.parent, Tag) else None
            if (
                current_cluster
                and parent_obj_id == current_parent_obj_id
                and previous_order is not None
                and tag_order - previous_order <= 3
            ):
                current_cluster.append((tag_order, tag))
            else:
                if current_cluster:
                    generic_clusters.append(current_cluster)
                current_cluster = [(tag_order, tag)]
                current_parent_obj_id = parent_obj_id
            previous_order = tag_order
        if current_cluster:
            generic_clusters.append(current_cluster)

        for cluster in generic_clusters:
            generic_values: list[_ExtractedFormValue] = []
            for order, value_tag in cluster:
                value_text_raw = self._extract_form_value_text(value_tag)
                value_orig, value_text = self._normalize_form_text(value_text_raw)
                generic_values.append(
                    _ExtractedFormValue(
                        tag=value_tag,
                        order=order,
                        orig=value_orig,
                        text=value_text,
                        prov=self._make_text_prov(text=value_text, tag=value_tag),
                        kind="fillable",
                    )
                )
                consumed_tag_ids.add(self._ensure_tag_html_id(value_tag))
            generic_values = self._merge_adjacent_form_values(generic_values)
            if not generic_values:
                continue
            fields.append(
                _ExtractedFormField(
                    key_tag=None,
                    key_order=generic_values[0].order,
                    key_orig="",
                    key_text="",
                    key_prov=None,
                    marker=None,
                    values=generic_values,
                    extra_texts=[],
                )
            )

        if not fields:
            return None
        return _ExtractedFormRegion(fields=fields, consumed_tag_ids=consumed_tag_ids)

    def _extract_form_graph(self, form_tag: Tag) -> Optional[GraphData]:
        extracted = self._extract_form_region(form_tag)
        if extracted is None:
            return None

        cells: list[GraphCell] = []
        links: list[GraphLink] = []
        cell_id_seq = 0
        for field in extracted.fields:
            if not field.key_text:
                continue
            key_cell = GraphCell(
                cell_id=cell_id_seq,
                label=GraphCellLabel.KEY,
                text=field.key_text,
                orig=field.key_orig,
                prov=field.key_prov,
            )
            cells.append(key_cell)
            cell_id_seq += 1

            for value in field.values:
                value_cell = GraphCell(
                    cell_id=cell_id_seq,
                    label=GraphCellLabel.VALUE,
                    text=value.text,
                    orig=value.orig,
                    prov=value.prov,
                )
                cells.append(value_cell)
                links.append(
                    GraphLink(
                        label=GraphLinkLabel.TO_VALUE,
                        source_cell_id=key_cell.cell_id,
                        target_cell_id=value_cell.cell_id,
                    )
                )
                cell_id_seq += 1

        if not cells:
            return None
        return GraphData(cells=cells, links=links)

    def _handle_form_container(self, tag: Tag, doc: DoclingDocument) -> list[RefItem]:
        added_refs: list[RefItem] = []
        supports_field_kv = all(
            hasattr(doc, method_name)
            for method_name in (
                "add_field_region",
                "add_field_item",
                "add_field_key",
                "add_field_value",
            )
        )

        if supports_field_kv:
            doc_with_fields = cast(Any, doc)
            form_region = self._extract_form_region(tag)
            consumed_tag_ids: set[str] = set()
            consumed_tag_obj_ids: set[int] = set()
            fields_by_key_id: dict[str, _ExtractedFormField] = {}
            if form_region is not None:
                consumed_tag_ids.update(form_region.consumed_tag_ids)
                for field in form_region.fields:
                    key_tag_id = self._get_html_id(field.key_tag)
                    if not field.values:
                        if (
                            field.key_tag is not None
                            and key_tag_id is not None
                            and self._is_lonely_key_covered_by_table(field.key_tag)
                        ):
                            consumed_tag_ids.add(key_tag_id)
                        continue

                    anchor_tag_ids: set[str] = set()
                    if key_tag_id is not None:
                        anchor_tag_ids.add(key_tag_id)
                    if field.marker is not None:
                        marker_tag_id = self._get_html_id(field.marker.tag)
                        if marker_tag_id is not None:
                            anchor_tag_ids.add(marker_tag_id)
                    for value in field.values:
                        value_tag_id = self._get_html_id(value.tag)
                        if value_tag_id is not None:
                            anchor_tag_ids.add(value_tag_id)
                    for anchor_tag_id in anchor_tag_ids:
                        consumed_tag_ids.add(anchor_tag_id)
                        fields_by_key_id[anchor_tag_id] = field
                    for extra_text in field.extra_texts:
                        consumed_tag_obj_ids.add(id(extra_text.tag))
                    for value in field.values:
                        consumed_tag_obj_ids.update(value.consumed_label_tag_obj_ids)

            if not fields_by_key_id:
                if tag.name.lower() == "table":
                    added_refs.extend(self._handle_block(tag, doc))
                else:
                    with self._suppress_tag_ids(consumed_tag_ids):
                        with self._suppress_tag_obj_ids(consumed_tag_obj_ids):
                            added_refs.extend(self._walk(tag, doc))
                return added_refs

            region_prov = self._make_prov(text="", tag=tag)
            field_region = doc_with_fields.add_field_region(
                prov=region_prov,
                parent=self.parents[self.level],
            )
            field_region.content_layer = self.content_layer
            added_refs.append(field_region.get_ref())

            with self._use_form_container(field_region):
                with self._use_form_fields_by_key_id(fields_by_key_id):
                    if tag.name.lower() == "table":
                        # For table-form containers, keep cell content visible to rich-cell parsing.
                        added_refs.extend(self._handle_block(tag, doc))
                    else:
                        with self._suppress_tag_ids(consumed_tag_ids):
                            with self._suppress_tag_obj_ids(consumed_tag_obj_ids):
                                added_refs.extend(self._walk(tag, doc))
            return added_refs

        form_graph = self._extract_form_graph(tag)
        form_data = form_graph if form_graph is not None else GraphData()
        form_prov = self._make_prov(text="", tag=tag)
        form_item = doc.add_form(
            graph=deepcopy(form_data),
            prov=form_prov,
            parent=self.parents[self.level],
        )
        form_item.content_layer = self.content_layer
        added_refs.append(form_item.get_ref())

        if form_graph is not None:
            kv_item = doc.add_key_values(
                graph=form_graph,
                prov=None,
                parent=form_item,
            )
            kv_item.content_layer = self.content_layer
            added_refs.append(kv_item.get_ref())

        with self._use_form_container(form_item):
            if tag.name.lower() == "table":
                added_refs.extend(self._handle_block(tag, doc))
            else:
                added_refs.extend(self._walk(tag, doc))
        return added_refs

    def _emit_image(self, img_tag: Tag, doc: DoclingDocument) -> Optional[RefItem]:
        figure = img_tag.find_parent("figure")
        caption: AnnotatedTextList = AnnotatedTextList()
        caption_prov_tag: Optional[Tag] = None

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
            caption_prov_tag = img_tag

        if isinstance(figure, Tag):
            caption_tag = figure.find("figcaption", recursive=False)
            if isinstance(caption_tag, Tag):
                caption = self._extract_text_and_hyperlink_recursively(
                    caption_tag, find_parent_annotation=True
                )
                caption_prov_tag = caption_tag
        if not caption and img_tag.get("alt"):
            caption = AnnotatedTextList([AnnotatedText(text=img_tag.get("alt"))])
            caption_prov_tag = img_tag

        caption_anno_text = caption.to_single_text_element()

        caption_item: Optional[TextItem] = None
        if caption_anno_text.text:
            text_clean = HTMLDocumentBackend._clean_unicode(
                caption_anno_text.text.strip()
            )
            prov = self._make_prov(
                text=text_clean,
                tag=caption_prov_tag or img_tag,
                source_tag_id=caption_anno_text.source_tag_id,
            )
            caption_item = doc.add_text(
                label=DocItemLabel.CAPTION,
                text=text_clean,
                orig=caption_anno_text.text,
                content_layer=self.content_layer,
                formatting=caption_anno_text.formatting,
                hyperlink=caption_anno_text.hyperlink,
                prov=prov,
            )

        src_loc: str = self._get_attr_as_string(img_tag, "src")
        pic_prov = self._make_prov(text="", tag=img_tag)
        if not cast(HTMLBackendOptions, self.options).fetch_images or not src_loc:
            # Do not fetch the image, just add a placeholder
            placeholder: PictureItem = doc.add_picture(
                caption=caption_item,
                parent=parent,
                content_layer=self.content_layer,
                prov=pic_prov,
            )
            return placeholder.get_ref()

        src_loc = self._resolve_relative_path(src_loc)
        img_ref = self._create_image_ref(src_loc)

        docling_pic = doc.add_picture(
            image=img_ref,
            caption=caption_item,
            parent=parent,
            content_layer=self.content_layer,
            prov=pic_prov,
        )
        return docling_pic.get_ref()

    def _emit_input(self, input_tag: Tag, doc: DoclingDocument) -> Optional[RefItem]:
        if self._is_suppressed_tag(input_tag):
            return None
        input_type = self._get_attr_as_string(input_tag, "type").lower()
        if input_type == "hidden":
            return None

        label = DocItemLabel.TEXT
        checkbox_label = self._get_checkbox_label_for_tag(input_tag)
        if checkbox_label is not None:
            label = checkbox_label
            (
                text_clean,
                _,
                checkbox_label_tags,
            ) = self._extract_checkbox_text_and_consumed_label_obj_ids(input_tag)
        else:
            text = self._get_attr_as_string(input_tag, "value").strip()
            if not text:
                text = self._get_attr_as_string(input_tag, "placeholder").strip()
            if not text:
                text = self._get_attr_as_string(input_tag, "name").strip()
            text_clean = HTMLDocumentBackend._clean_unicode(text) if text else ""
        prov = (
            self._make_checkbox_with_label_prov(
                text=text_clean,
                checkbox_tag=input_tag,
                label_tags=checkbox_label_tags,
            )
            if checkbox_label is not None
            else self._make_prov(text=text_clean, tag=input_tag)
        )
        input_item = doc.add_text(
            parent=self.parents[self.level],
            label=label,
            text=text_clean,
            content_layer=self.content_layer,
            formatting=self._formatting,
            hyperlink=self.hyperlink,
            prov=prov,
        )
        return input_item.get_ref()

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
                    "".join(parts) + " "
                    if tag.name in {"p", "li", "th", "td"}
                    else "".join(parts)
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
