from __future__ import annotations

import logging
import re
import warnings
import zipfile
from contextlib import contextmanager
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Final
from urllib.parse import urlparse

from docling_core.types.doc import (
    CodeItem,
    ContentLayer,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupLabel,
    ImageRef,
    ListGroup,
    NodeItem,
    PictureClassificationLabel,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureMeta,
    RefItem,
    RichTableCell,
    TableCell,
    TableData,
    TableItem,
    TabularChartMetaField,
)
from docling_core.types.doc.document import FineRef, Formatting, Script
from lxml import etree
from PIL import Image, UnidentifiedImageError
from pydantic import AnyUrl, ValidationError
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.docx.drawingml.utils import (
    convert_to_modern_format,
    get_docx_to_pdf_converter,
    get_pil_from_dml_docx,
)
from docling.backend.docx.latex.omml import oMath2Latex
from docling.datamodel.backend_options import MsWordBackendOptions
from docling.datamodel.base_models import FormatToMimeType, InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError, SecurityError

_log = logging.getLogger(__name__)

_DOCX_AVAILABLE: bool = False
_DOCX_IMPORT_ERROR: ImportError | None = None
try:  # pragma: no cover - import-time guard
    from docx import Document
    from docx.document import Document as DocxDocument
    from docx.oxml.simpletypes import ST_Merge
    from docx.oxml.xmlchemy import BaseOxmlElement
    from docx.styles.style import ParagraphStyle
    from docx.table import Table, _Cell
    from docx.text.hyperlink import Hyperlink
    from docx.text.paragraph import Paragraph
    from docx.text.run import Run

    _DOCX_AVAILABLE = True
except ImportError as e:  # pragma: no cover - import-time guard
    _DOCX_IMPORT_ERROR = e

_INSTALL_HINT = (
    "The 'python-docx' package is required to process Word files. "
    "Install it with `pip install 'docling-slim[format-docx]'`."
)

_CHART_RENDER_HINT = (
    "LibreOffice is required to render Word charts as images "
    "(render_chart_images=True). Install LibreOffice and make sure `soffice` is "
    "on PATH. Charts still keep their classification and reconstructed tabular "
    "data without it."
)

# Safe XML parser for chart parts — prevents XXE, DTD-over-network, and
# entity-expansion attacks when parsing untrusted ``chartN.xml`` payloads.
_SAFE_XML_PARSER: Final = etree.XMLParser(
    resolve_entities=False,
    load_dtd=False,
    no_network=True,
    dtd_validation=False,
)

# Maps a DrawingML chart plot element name (e.g. "barChart", the child of
# ``c:plotArea``) to the docling picture-classification label the emitted chart
# PictureItem is tagged with. These tagnames are shared across the OOXML Office
# formats; chart types not listed fall back to OTHER_CHART.
_CHART_TAGNAME_TO_CLASSIFICATION: Final[dict[str, PictureClassificationLabel]] = {
    "barChart": PictureClassificationLabel.BAR_CHART,
    "bar3DChart": PictureClassificationLabel.BAR_CHART,
    "lineChart": PictureClassificationLabel.LINE_CHART,
    "line3DChart": PictureClassificationLabel.LINE_CHART,
    "pieChart": PictureClassificationLabel.PIE_CHART,
    "pie3DChart": PictureClassificationLabel.PIE_CHART,
    "doughnutChart": PictureClassificationLabel.PIE_CHART,
    "scatterChart": PictureClassificationLabel.SCATTER_CHART,
    "areaChart": PictureClassificationLabel.OTHER_CHART,
    "area3DChart": PictureClassificationLabel.OTHER_CHART,
}

_STRICT_OOXML_NS_PREFIX: Final[str] = "http://purl.oclc.org/ooxml/"
_TRANSITIONAL_NS_HOST: Final[str] = "http://schemas.openxmlformats.org/"

_STRICT_OOXML_MARKER: Final[bytes] = b"purl.oclc.org/ooxml"
"""Byte string present in every Strict OOXML part that carries a Strict namespace URI."""

_OOXML_ROOT_RELS: Final[str] = "_rels/.rels"
"""OPC root relationships part; its ``officeDocument`` type identifies Strict vs Transitional."""

_MAX_ROOT_RELS_SIZE: Final[int] = 64 * 1024  # 64 KiB
"""Read cap for ``_rels/.rels`` during Strict detection (the part is typically ~500 bytes)."""

_MAX_MEMBER_UNCOMPRESSED_SIZE: Final[int] = 512 * 1024 * 1024  # 512 MiB
"""Per-member uncompressed size cap applied during Strict-to-Transitional rewriting."""

_MAX_TOTAL_UNCOMPRESSED_SIZE: Final[int] = 2 * 1024 * 1024 * 1024  # 2 GiB
"""Total uncompressed size cap for all members during Strict-to-Transitional rewriting."""

_STRICT_OOXML_NS_OVERRIDES: Final[dict[str, str]] = {
    "http://purl.oclc.org/ooxml/descriptions/base": "http://descriptions.openxmlformats.org/description/base",
    "http://purl.oclc.org/ooxml/descriptions/full": "http://descriptions.openxmlformats.org/description/full",
    "http://purl.oclc.org/ooxml/officeDocument/relationships/customXml": "http://schemas.openxmlformats.org/officeDocument/2006/customXml",
    "http://purl.oclc.org/ooxml/officeDocument/relationships/metadata/thumbnail": "http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail",
}
"""Strict namespace URIs whose Transitional equivalents are irregular (Open XML SDK NamespaceIdMap)."""

_STRICT_OOXML_NS_RE: Final = re.compile(
    r"http://purl\.oclc\.org/ooxml/[A-Za-z0-9_./-]+"
)
"""Matches Strict OOXML namespace/relationship URIs."""

_VISIBLE_NUMBERING_FORMATS: Final[frozenset[str]] = frozenset(
    {
        "decimal",
        "lowerRoman",
        "upperRoman",
        "lowerLetter",
        "upperLetter",
        "decimalZero",
    }
)
"""OOXML numFmt values that produce visible list/heading markers."""


def _strict_ns_to_transitional(strict_ns: str) -> str:
    """Map a single Strict OOXML namespace/relationship URI to its Transitional form."""
    if strict_ns in _STRICT_OOXML_NS_OVERRIDES:
        return _STRICT_OOXML_NS_OVERRIDES[strict_ns]
    rest = strict_ns[len(_STRICT_OOXML_NS_PREFIX) :]
    rest = rest.replace("extendedProperties", "extended-properties")
    rest = rest.replace("customProperties", "custom-properties")
    segment, separator, tail = rest.partition("/")
    if not separator:
        return f"{_TRANSITIONAL_NS_HOST}{segment}/2006"
    return f"{_TRANSITIONAL_NS_HOST}{segment}/2006/{tail}"


def _is_strict_ooxml(archive: zipfile.ZipFile) -> bool:
    """Cheaply decide whether an open .docx archive is a Strict OOXML package.

    Only the small package root relationships part (``_rels/.rels``) is read, so
    the common Transitional case does not pay for decompressing the whole file.
    """
    try:
        with archive.open(_OOXML_ROOT_RELS) as root_rels:
            return _STRICT_OOXML_MARKER in root_rels.read(_MAX_ROOT_RELS_SIZE)
    except KeyError:
        # No root relationships: not a well-formed OOXML package. Let python-docx
        # deal with it on the unchanged path.
        return False


def _is_safe_zip_member(name: str) -> bool:
    """Return whether a zip member name stays inside the archive root.

    Guards against zip-slip: absolute paths, drive-letter paths and ``..``
    traversal are rejected.
    """
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
        return False
    return not any(part == ".." for part in normalized.split("/"))


def _normalize_strict_ooxml(archive: zipfile.ZipFile) -> BytesIO:
    """Rewrite an open Strict OOXML package to Transitional namespaces in memory.

    Only XML/relationship parts that actually carry a Strict namespace are
    decoded and rewritten; every other member (images, fonts, ...) is copied
    through with its original compression, avoiding a needless decode pass. Each
    member is decompressed exactly once. The archive is validated against
    zip-slip and zip-bomb attacks while it is read.
    """
    normalized = BytesIO()
    total_uncompressed = 0
    with zipfile.ZipFile(normalized, "w", zipfile.ZIP_DEFLATED) as target:
        for info in archive.infolist():
            if not _is_safe_zip_member(info.filename):
                raise SecurityError(f"ZIP slip attempt: {info.filename}")
            if info.file_size > _MAX_MEMBER_UNCOMPRESSED_SIZE:
                raise SecurityError(
                    f"Refusing to expand oversized OOXML part: {info.filename}"
                )
            total_uncompressed += info.file_size
            if total_uncompressed > _MAX_TOTAL_UNCOMPRESSED_SIZE:
                raise SecurityError(
                    "Refusing to expand OOXML package exceeding the uncompressed size limit"
                )
            content = archive.read(info.filename)
            if (
                info.filename.endswith((".xml", ".rels"))
                and _STRICT_OOXML_MARKER in content
            ):
                content = _STRICT_OOXML_NS_RE.sub(
                    lambda match: _strict_ns_to_transitional(match.group(0)),
                    content.decode("utf-8"),
                ).encode("utf-8")
            target.writestr(info, content)
    normalized.seek(0)
    return normalized


class MsWordDocumentBackend(DeclarativeDocumentBackend):
    """Backend for parsing Word documents (DOCX and DOC files).

    Both Transitional (ISO/IEC 29500-4) and Strict (ISO/IEC 29500-1) ``.docx``
    packages are supported. Strict packages use ``purl.oclc.org`` namespace URIs
    that ``python-docx`` does not recognise; they are normalised to their
    Transitional equivalents in memory before parsing. Transitional files are
    handed to ``python-docx`` unchanged.

    Legacy ``.doc`` files (binary Word 97-2004 format) are first converted to
    ``.docx`` via LibreOffice before parsing.

    Note:
        Images with a total area (width x height) less than or equal to
        `SPACER_IMAGE_AREA_THRESHOLD` (default: 25 px2) are treated as invisible
        layout spacers and discarded during parsing.
    """

    _W_NS: Final[str] = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    _W_NS_CLARK: Final[str] = f"{{{_W_NS}}}"

    _BLIP_NAMESPACES: Final = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "w": _W_NS,
        "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
        "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
        "v": "urn:schemas-microsoft-com:vml",
        "wps": "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
        "w10": "urn:schemas-microsoft-com:office:word",
        "a14": "http://schemas.microsoft.com/office/drawing/2010/main",
        "w14": "http://schemas.microsoft.com/office/word/2010/wordml",
    }

    SPACER_IMAGE_AREA_THRESHOLD: Final[int] = 25
    """Images with an area (w*h) below this are dropped as layout artifacts."""

    # Style names/ids that mark code, per pandoc's docx reader ("Source
    # Code") and LibreOffice/HTML equivalents. Matched exactly (case-folded),
    # never by substring: "Unicode" or "Area Code" are not code, and caption
    # styles like "Listing" are left out.
    _CODE_STYLE_NAMES: Final[frozenset[str]] = frozenset(
        {
            "source code",
            "code",
            "code block",
            "code listing",
            "html preformatted",
            "preformatted text",
            "preformatted",
            "verbatim",
        }
    )
    _CODE_STYLE_IDS: Final[frozenset[str]] = frozenset(
        {
            "sourcecode",
            "source_code",
            "code",
            "codeblock",
            "codelisting",
            "htmlpreformatted",
            "preformattedtext",
            "preformatted",
            "verbatim",
        }
    )
    # Defensive cap against malformed/cyclic base_style inheritance chains.
    _MAX_STYLE_INHERITANCE_DEPTH: Final[int] = 10

    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: BytesIO | Path,
        options: MsWordBackendOptions | None = None,
    ) -> None:
        if not _DOCX_AVAILABLE:
            raise ImportError(_INSTALL_HINT) from _DOCX_IMPORT_ERROR
        if options is None:
            options = MsWordBackendOptions()
        if in_doc.format == InputFormat.DOC:
            path_or_stream = convert_to_modern_format(path_or_stream, "doc", "docx")
        super().__init__(in_doc, path_or_stream, options)
        self.XML_KEY = f"{self._W_NS_CLARK}val"
        self.xml_namespaces = {
            "w": "http://schemas.microsoft.com/office/word/2003/wordml"
        }
        self.blip_xpath_expr = etree.XPath(
            ".//a:blip", namespaces=MsWordDocumentBackend._BLIP_NAMESPACES
        )
        self.vml_imagedata_xpath_expr = etree.XPath(
            ".//v:imagedata", namespaces=MsWordDocumentBackend._BLIP_NAMESPACES
        )
        # self.initialise(path_or_stream)
        # Word file:
        self.path_or_stream: BytesIO | Path = path_or_stream
        self.valid: bool = False
        # Initialise the parents for the hierarchy
        self.max_levels: int = 10
        self.level_at_new_list: int | None = None
        self.parents: dict[int, NodeItem | None] = {}
        self.numbered_headers: dict[int, int] = {}
        self.equation_bookends: str = "<eq>{EQ}</eq>"
        # Track processed textbox elements to avoid duplication
        self.processed_textbox_elements: list[int] = []
        self.docx_to_pdf_converter: Callable | None = None
        self.docx_to_pdf_converter_init = False
        self.display_drawingml_warning = True

        for i in range(-1, self.max_levels):
            self.parents[i] = None

        self.level = 0
        self.listIter = 0
        # Track list counters per numId and ilvl
        self.list_counters: dict[tuple[int, int], int] = {}
        # Track the last numId to handle list continuation after interruptions
        self.last_numid: int | None = None
        # Track the last list group and its parent to reuse only in same context
        self.last_list_group: ListGroup | None = None
        self.last_list_group_numid: int | None = None
        self.last_list_group_parent: NodeItem | None = None
        # Set starting content layer
        self.content_layer = ContentLayer.BODY

        self.history: dict[str, Any] = {
            "names": [None],
            "levels": [None],
            "numids": [None],
            "indents": [None],
        }

        # Track comment mappings: comment_id -> comment object
        self.comment_map: dict[str, Any] = {}
        # Track paragraph elements to their comment IDs
        self.paragraph_comment_map: dict[int, list[str]] = {}
        # Track text items created from each paragraph element
        self.paragraph_to_items: dict[int, list[RefItem]] = {}
        # Forces the next code paragraph to start a fresh block; set at
        # boundaries that add no item of their own (1x1 tables, headers).
        self._force_new_code_block: bool = False
        # Blank code paragraphs are buffered so a block never ends in them.
        self._pending_code_blank_lines: int = 0

        self.docx_obj = self.load_msword_file(
            path_or_stream=self.path_or_stream, document_hash=self.document_hash
        )
        if self.docx_obj:
            self.valid = True
            self.current_part = self.docx_obj.part
            # Build comment mappings after loading document
            self._extract_comment_ranges()

    @override
    def is_valid(self) -> bool:
        return self.valid

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
        return {InputFormat.DOCX, InputFormat.DOC}

    @override
    def convert(self) -> DoclingDocument:
        """Parses the DOCX into a structured document model.

        Returns:
            The parsed document.
        """

        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype=FormatToMimeType[self.input_format][0],
            binary_hash=self.document_hash,
        )

        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)
        if self.is_valid():
            assert self.docx_obj is not None
            # Reset mappings for a fresh conversion pass
            self.paragraph_comment_map.clear()
            self.paragraph_to_items.clear()
            doc, _ = self._walk_linear(self.docx_obj.element.body, doc)
            self._add_header_footer(self.docx_obj, doc)
            # Add comments and link them to annotated paragraphs
            self._add_comments(self.docx_obj, doc)

            return doc
        else:
            raise RuntimeError(
                f"Cannot convert doc with {self.document_hash} because the backend failed to init."
            )

    @staticmethod
    def load_msword_file(
        path_or_stream: BytesIO | Path, document_hash: str
    ) -> DocxDocument:
        try:
            if isinstance(path_or_stream, Path):
                with zipfile.ZipFile(path_or_stream) as archive:
                    if _is_strict_ooxml(archive):
                        return Document(_normalize_strict_ooxml(archive))
                return Document(str(path_or_stream))
            elif isinstance(path_or_stream, BytesIO):
                with zipfile.ZipFile(path_or_stream) as archive:
                    if _is_strict_ooxml(archive):
                        return Document(_normalize_strict_ooxml(archive))
                path_or_stream.seek(0)
                return Document(path_or_stream)
            else:
                return None
        except SecurityError:
            raise
        except Exception as e:
            raise DocumentLoadError(
                f"MsWordDocumentBackend could not load document with hash {document_hash}"
            ) from e

    def _update_history(
        self,
        name: str,
        level: int | None,
        numid: int | None,
        ilevel: int | None,
    ):
        self.history["names"].append(name)
        self.history["levels"].append(level)

        self.history["numids"].append(numid)
        self.history["indents"].append(ilevel)

    def _prev_name(self) -> str | None:
        return self.history["names"][-1]

    def _prev_level(self) -> int | None:
        return self.history["levels"][-1]

    def _prev_numid(self) -> int | None:
        return self.history["numids"][-1]

    def _prev_indent(self) -> int | None:
        return self.history["indents"][-1]

    def _get_level(self) -> int:
        """Return the first None index."""
        for k, v in self.parents.items():
            if k >= 0 and v is None:
                return k
        return 0

    def _can_reuse_list_group(self, numid: int, parent: NodeItem | None) -> bool:
        """Check if we can reuse the cached list group.

        Returns True only if we're continuing the same numId in the same parent context.
        """
        return (
            self.last_numid == numid
            and self.last_list_group is not None
            and self.last_list_group_numid == numid
            and self.last_list_group_parent == parent
        )

    def _get_or_create_list_group(
        self,
        doc: DoclingDocument,
        numid: int,
        parent: NodeItem | None,
        elem_ref: list[RefItem],
    ) -> ListGroup:
        """Get existing list group if reusable, otherwise create a new one.

        Args:
            doc: The DoclingDocument being constructed.
            numid: The numbering ID for this list.
            parent: The parent node for the list group.
            elem_ref: List to append new group reference to (if created).

        Returns:
            The list group to use (either reused or newly created).
        """
        if self._can_reuse_list_group(numid, parent):
            # When reusing a list group, remove any empty text item that was added
            # between the last list item and this one (from closing the list)
            if doc.texts and len(doc.texts) > 0:
                last_text = doc.texts[-1]
                if not last_text.text or not last_text.text.strip():
                    doc.delete_items(node_items=[last_text])
            return self.last_list_group

        list_gr = doc.add_list_group(
            name="list",
            parent=parent,
            content_layer=self.content_layer,
        )
        elem_ref.append(list_gr.get_ref())

        # Update cache for potential future reuse
        self.last_list_group = list_gr
        self.last_list_group_numid = numid
        self.last_list_group_parent = parent

        return list_gr

    def _clear_list_group_cache(self) -> None:
        """Clear the cached list group to prevent reuse across contexts."""
        self.last_list_group = None
        self.last_list_group_numid = None
        self.last_list_group_parent = None

    @contextmanager
    def _isolated_list_context(self):
        """Preserve list state during table cell processing.

        This context manager saves the list-related state (history, level_at_new_list,
        and parents) before entering a table cell, and restores it after processing.
        This ensures that lists in different table cells are treated independently,
        even when they share the same numId.
        """
        saved_history = {
            "names": self.history["names"].copy(),
            "levels": self.history["levels"].copy(),
            "numids": self.history["numids"].copy(),
            "indents": self.history["indents"].copy(),
        }
        saved_level_at_new_list = self.level_at_new_list
        saved_parents = self.parents.copy()
        # Save and clear list group cache to prevent reuse across table cells
        saved_last_list_group = self.last_list_group
        saved_last_list_group_numid = self.last_list_group_numid
        saved_last_list_group_parent = self.last_list_group_parent
        self._clear_list_group_cache()

        try:
            yield
        finally:
            self.history = saved_history
            self.level_at_new_list = saved_level_at_new_list
            self.parents = saved_parents
            self.last_list_group = saved_last_list_group
            self.last_list_group_numid = saved_last_list_group_numid
            self.last_list_group_parent = saved_last_list_group_parent

    def _walk_linear(
        self,
        body: BaseOxmlElement,
        doc: DoclingDocument,
        # parent:
    ) -> tuple[DoclingDocument, list[RefItem]]:
        added_elements = []
        for element in body:
            tag_name = etree.QName(element).localname
            # Check for Inline Images (blip elements)
            _raw_drawing_blip = self.blip_xpath_expr(element)
            _raw_drawingml_els = element.findall(
                ".//w:drawing", namespaces=MsWordDocumentBackend._BLIP_NAMESPACES
            )
            _raw_vml_images = self.vml_imagedata_xpath_expr(element)

            # Filter out images inside textboxes to prevent double-extraction
            # (they will be properly extracted by _handle_textbox_content instead)
            def _in_textbox(elem):
                return any(
                    etree.QName(anc).localname in ["txbxContent", "textbox"]
                    for anc in elem.iterancestors()
                )

            drawing_blip = [x for x in _raw_drawing_blip if not _in_textbox(x)]
            drawingml_els = [x for x in _raw_drawingml_els if not _in_textbox(x)]
            vml_images = [x for x in _raw_vml_images if not _in_textbox(x)]

            # Check for textbox content - check multiple textbox formats
            # Only process if the element hasn't been processed before
            element_id = id(element)
            if element_id not in self.processed_textbox_elements:
                # Modern Word textboxes
                txbx_xpath = etree.XPath(
                    ".//w:txbxContent|.//v:textbox//w:p",
                    namespaces=MsWordDocumentBackend._BLIP_NAMESPACES,
                )
                textbox_elements = txbx_xpath(element)

                # No modern textboxes found, check for alternate/legacy textbox formats
                if not textbox_elements:
                    # Additional checks for textboxes in DrawingML and VML formats
                    alt_txbx_xpath = etree.XPath(
                        ".//wps:txbx//w:p|.//w10:wrap//w:p|.//v:textbox//w:txbxContent//w:p",
                        namespaces=MsWordDocumentBackend._BLIP_NAMESPACES,
                    )
                    textbox_elements = alt_txbx_xpath(element)

                    # Check for shape text that's not in a standard textbox
                    if not textbox_elements:
                        shape_text_xpath = etree.XPath(
                            ".//a:bodyPr/ancestor::*//a:t|.//a:txBody//a:t",
                            namespaces=MsWordDocumentBackend._BLIP_NAMESPACES,
                        )
                        shape_text_elements = shape_text_xpath(element)
                        if shape_text_elements:
                            # Create custom text elements from shape text
                            text_content = " ".join(
                                [t.text for t in shape_text_elements if t.text]
                            )
                            if text_content.strip():
                                _log.debug(f"Found shape text: {text_content[:50]}...")
                                # Create a paragraph-like element to process with standard handler
                                level = self._get_level()
                                shape_group = doc.add_group(
                                    label=GroupLabel.SECTION,
                                    parent=self.parents[level - 1],
                                    name="shape-text",
                                    content_layer=self.content_layer,
                                )
                                added_elements.append(shape_group.get_ref())
                                doc.add_text(
                                    label=DocItemLabel.TEXT,
                                    parent=shape_group,
                                    text=text_content,
                                    content_layer=self.content_layer,
                                )

                if textbox_elements:
                    # Mark the parent element as processed
                    self.processed_textbox_elements.append(element_id)
                    # Also mark all found textbox elements as processed
                    for tb_element in textbox_elements:
                        self.processed_textbox_elements.append(id(tb_element))

                    _log.debug(
                        f"Found textbox content with {len(textbox_elements)} elements"
                    )
                    tbc = self._handle_textbox_content(textbox_elements, doc)
                    added_elements.extend(tbc)

            # Check for Tables
            if tag_name == "tbl":
                try:
                    t = self._handle_tables(element, doc)
                    added_elements.extend(t)
                except Exception:
                    _log.debug("could not parse a table, broken docx table")
            # Check for Image
            elif drawing_blip:
                pics = self._handle_pictures(drawing_blip, doc)
                added_elements.extend(pics)
                # Check for Text after the Image
                if (
                    tag_name == "p"
                    and element.find(
                        ".//w:t", namespaces=MsWordDocumentBackend._BLIP_NAMESPACES
                    )
                    is not None
                ):
                    te1 = self._handle_text_elements(element, doc)
                    added_elements.extend(te1)
            # Check for VML images (legacy format, e.g., embedded Visio drawings)
            elif vml_images:
                vml_pics = self._handle_vml_pictures(vml_images, doc)
                added_elements.extend(vml_pics)
                # Check for Text after the VML Image
                if (
                    tag_name == "p"
                    and element.find(
                        ".//w:t", namespaces=MsWordDocumentBackend._BLIP_NAMESPACES
                    )
                    is not None
                ):
                    te2 = self._handle_text_elements(element, doc)
                    added_elements.extend(te2)
            # Check for DrawingML elements
            elif drawingml_els:
                # Native charts (graphicFrames referencing word/charts/chartN.xml)
                # are parsed into classified pictures carrying their reconstructed
                # data, without LibreOffice. Any remaining DrawingML (shapes,
                # SmartArt, ...) still needs LibreOffice to be rasterized.
                chart_els: list[Any] = []
                other_els: list[Any] = []
                for el in drawingml_els:
                    (chart_els if self._is_chart_drawing(el) else other_els).append(el)

                for chart_el in chart_els:
                    chart_ref = self._handle_chart(doc=doc, chart_el=chart_el)
                    if chart_ref is not None:
                        added_elements.append(chart_ref)

                if other_els:
                    if (
                        self.docx_to_pdf_converter is None
                        and self.docx_to_pdf_converter_init is False
                    ):
                        self.docx_to_pdf_converter = get_docx_to_pdf_converter()
                        self.docx_to_pdf_converter_init = True

                    if self.docx_to_pdf_converter is None:
                        if self.display_drawingml_warning:
                            _log.warning(
                                "Found DrawingML elements in document, but no DOCX to PDF converters. "
                                "If you want these exported, make sure you have "
                                "LibreOffice binary in PATH or specify its path with DOCLING_LIBREOFFICE_CMD."
                            )
                            self.display_drawingml_warning = False
                    else:
                        self._handle_drawingml(doc=doc, drawingml_els=other_els)

                # Always process text in paragraph
                if (
                    tag_name == "p"
                    and element.find(
                        ".//w:t", namespaces=MsWordDocumentBackend._BLIP_NAMESPACES
                    )
                    is not None
                ):
                    te = self._handle_text_elements(element, doc, skip_empty_text=True)
                    added_elements.extend(te)
            # Check for the sdt containers, like table of contents
            elif tag_name == "sdt":
                sdt_content = element.find(
                    "./w:sdtContent", namespaces=MsWordDocumentBackend._BLIP_NAMESPACES
                )
                if sdt_content is not None:
                    # Recursively walk the SDT content to catch textboxes, tables, and nested structures
                    _, te = self._walk_linear(sdt_content, doc)
                    added_elements.extend(te)
            # Check for Text
            elif tag_name == "p":
                # "tcPr", "sectPr"
                te = self._handle_text_elements(element, doc)
                added_elements.extend(te)
            else:
                _log.debug(f"Ignoring element in DOCX with tag: {tag_name}")

        return doc, added_elements

    def _is_invisible_spacer(self, pil_image: Image.Image | None) -> bool:
        """Check if an image is an invisible layout spacer rather than a meaningful graphic."""
        if pil_image is None:
            return False

        # Filter tiny spacer images
        if pil_image.width * pil_image.height <= self.SPACER_IMAGE_AREA_THRESHOLD:
            return True

        try:
            extrema = pil_image.getextrema()
            if extrema is not None:
                if pil_image.mode in ("RGBA", "LA"):
                    # extrema[-1] is the Alpha channel. If max alpha is 0, it is 100% invisible.
                    if extrema[-1][1] == 0:
                        return True
                elif pil_image.mode == "RGB":
                    # If all channels are exactly 255, it is a pure white spacing box.
                    if extrema == ((255, 255), (255, 255), (255, 255)):
                        return True
        except Exception:
            pass  # pragma: no cover

        return False

    def _str_to_int(self, s: str | None, default: int | None = 0) -> int | None:
        if s is None:
            return None
        try:
            return int(s)
        except ValueError:
            return default

    def _split_text_and_number(self, input_string: str) -> list[str]:
        match = re.match(r"(\D+)(\d+)$|^(\d+)(\D+)", input_string)
        if match:
            parts = list(filter(None, match.groups()))
            return parts
        else:
            return [input_string]

    def _get_numId_and_ilvl(
        self, paragraph: Paragraph
    ) -> tuple[int | None, int | None]:
        # Access the XML element of the paragraph
        numPr = paragraph._element.find(
            ".//w:numPr", namespaces=paragraph._element.nsmap
        )

        if numPr is not None:
            # Get the numId element and extract the value
            numId_elem = numPr.find("w:numId", namespaces=paragraph._element.nsmap)
            ilvl_elem = numPr.find("w:ilvl", namespaces=paragraph._element.nsmap)
            numId = numId_elem.get(self.XML_KEY) if numId_elem is not None else None
            ilvl = ilvl_elem.get(self.XML_KEY) if ilvl_elem is not None else None

            return self._str_to_int(numId, None), self._str_to_int(ilvl, None)

        # If not found directly in paragraph, check if the style defines numbering
        if paragraph.style is not None:
            style_elem = getattr(paragraph.style, "element", None)
            if style_elem is not None:
                style_numPr = style_elem.find(f".//{self._W_NS_CLARK}numPr")
                if style_numPr is not None:
                    numId_elem = style_numPr.find(f"{self._W_NS_CLARK}numId")
                    ilvl_elem = style_numPr.find(f"{self._W_NS_CLARK}ilvl")
                    numId = (
                        numId_elem.get(self.XML_KEY) if numId_elem is not None else None
                    )
                    ilvl = (
                        ilvl_elem.get(self.XML_KEY) if ilvl_elem is not None else None
                    )

                    # If numId is found but ilvl is not specified, default to level 0
                    if numId is not None and ilvl is None:
                        ilvl = "0"

                    return self._str_to_int(numId, None), self._str_to_int(ilvl, None)

        return None, None  # If the paragraph is not part of a list

    def _get_level_element(self, numid: int, ilvl: int) -> BaseOxmlElement | None:
        """Find the level element from the numbering XML for a given numId and ilvl."""
        try:
            if not hasattr(self.docx_obj, "part") or not hasattr(
                self.docx_obj.part, "package"
            ):
                return None

            numbering_part = None
            for part in self.docx_obj.part.package.parts:
                if "numbering" in part.partname:
                    numbering_part = part
                    break

            if numbering_part is None:
                return None

            numbering_root = numbering_part.element
            namespaces = {"w": self._W_NS}

            num_element = numbering_root.find(
                f".//w:num[@w:numId='{numid}']", namespaces=namespaces
            )
            if num_element is None:
                return None

            abstract_num_id_elem = num_element.find(
                ".//w:abstractNumId", namespaces=namespaces
            )
            if abstract_num_id_elem is None:
                return None

            abstract_num_id = abstract_num_id_elem.get(self.XML_KEY)
            if abstract_num_id is None:
                return None

            abstract_num_element = numbering_root.find(
                f".//w:abstractNum[@w:abstractNumId='{abstract_num_id}']",
                namespaces=namespaces,
            )
            if abstract_num_element is None:
                return None

            return abstract_num_element.find(
                f".//w:lvl[@w:ilvl='{ilvl}']", namespaces=namespaces
            )
        except Exception as e:
            _log.debug(f"Error finding level element: {e}")
            return None

    def _get_start_value(self, numid: int, ilvl: int) -> int:
        """Read the start value from the abstractNum definition."""
        lvl_element = self._get_level_element(numid, ilvl)
        if lvl_element is not None:
            namespaces = {"w": self._W_NS}
            start_element = lvl_element.find(".//w:start", namespaces=namespaces)
            if start_element is not None:
                val = start_element.get(self.XML_KEY)
                if val is not None:
                    return int(val)
        return 1

    def _get_list_counter(self, numid: int, ilvl: int) -> int:
        """Get and increment the counter for a specific numId and ilvl combination."""
        key = (numid, ilvl)
        if key not in self.list_counters:
            start = self._get_start_value(numid, ilvl)
            self.list_counters[key] = start - 1
        self.list_counters[key] += 1
        # Reset sub-level counters since parent level advanced
        for k in [k for k in self.list_counters if k[0] == numid and k[1] > ilvl]:
            self.list_counters[k] = 0
        return self.list_counters[key]

    def _reset_list_counters_for_new_sequence(self, numid: int):
        """Reset counters when starting a new numbering sequence."""
        # Reset all counters for this numid
        keys_to_reset = [key for key in self.list_counters.keys() if key[0] == numid]
        for key in keys_to_reset:
            self.list_counters[key] = 0

    def _build_enum_marker(self, numid: int, ilvl: int) -> str:
        """Build enumeration marker from the lvlText template (e.g. 'Proposal %1:').

        Uses lvlText when it contains a text prefix/suffix beyond simple
        placeholders and separators.  Falls back to the default '1.2.3.'
        pattern for plain numeric markers.
        """
        lvl_element = self._get_level_element(numid, ilvl)
        namespaces = {"w": self._W_NS}
        lvl_text = None
        if lvl_element is not None:
            lt = lvl_element.find(".//w:lvlText", namespaces=namespaces)
            if lt is not None:
                lvl_text = lt.get(self.XML_KEY)

        # Use lvlText as template only when it contains %N placeholders
        # alongside non-trivial text (e.g. "Proposal %1:", "Table %1").
        # Skip when lvlText is a bare bullet symbol like "o" or "•".
        if lvl_text and re.search(r"%\d+", lvl_text):
            stripped = re.sub(r"%\d+", "", lvl_text)
            stripped = stripped.strip(" .)(:[]")
            if stripped:

                def _replace(match):
                    lvl_idx = int(match.group(1)) - 1
                    counter = self.list_counters.get((numid, lvl_idx))
                    if counter is None:
                        counter = self._get_start_value(numid, lvl_idx)
                    return str(counter)

                return re.sub(r"%(\d+)", _replace, lvl_text)

        # Fallback: default hierarchical '1.2.3.' pattern
        parts = []
        for lvl in range(ilvl + 1):
            counter = self.list_counters.get((numid, lvl))
            if counter is None:
                counter = self._get_start_value(numid, lvl)
            parts.append(str(counter))
        return ".".join(parts) + "."

    def _has_visible_numbering_format(self, numId: int, ilvl: int) -> bool:
        """Return True when numbering.xml defines a visible marker for numId/ilvl."""
        try:
            lvl_element = self._get_level_element(numId, ilvl)
            if lvl_element is None:
                return False

            namespaces = {"w": self._W_NS}
            num_fmt_element = lvl_element.find(".//w:numFmt", namespaces=namespaces)
            if num_fmt_element is None:
                return False

            num_fmt = num_fmt_element.get(self.XML_KEY)

            return num_fmt in _VISIBLE_NUMBERING_FORMATS

        except Exception as e:
            _log.debug(f"Error determining visible numbering format: {e}")
            return False

    def _is_numbered_heading(self, paragraph: Paragraph) -> bool:
        """Return True when heading numbering would render a visible marker."""
        numid, ilvl = self._get_numId_and_ilvl(paragraph)
        return numid is not None and self._has_visible_numbering_format(
            numid, ilvl or 0
        )

    def _get_outline_level_from_style(self, paragraph: Paragraph) -> int | None:
        """Extract outlineLvl from paragraph's style definition.

        In OOXML, outlineLvl is 0-indexed (0-8 for heading levels 1-9).
        This method returns the 1-indexed heading level (outlineLvl + 1).
        """
        if paragraph.style is None:
            return None

        style_elem = getattr(paragraph.style, "element", None)
        if style_elem is None:
            return None

        # Look for outlineLvl in the style's paragraph properties
        outline_elem = style_elem.find(f".//{self._W_NS_CLARK}outlineLvl")
        if outline_elem is not None:
            val = outline_elem.get(f"{self._W_NS_CLARK}val")
            if val is not None:
                try:
                    # Convert 0-indexed outlineLvl to 1-indexed heading level
                    return int(val) + 1
                except ValueError:
                    pass
        return None

    def _get_heading_and_level(self, style_label: str) -> tuple[str, int | None]:
        parts = self._split_text_and_number(style_label)

        if len(parts) == 2:
            parts.sort()
            label_str: str = ""
            label_level: int | None = 0
            if parts[0].strip().lower() == "heading":
                label_str = "Heading"
                label_level = self._str_to_int(parts[1], None)
            if parts[1].strip().lower() == "heading":
                label_str = "Heading"
                label_level = self._str_to_int(parts[0], None)
            # Ensure heading level is at least 1 (e.g., custom "Heading 0" styles)
            if isinstance(label_level, int) and label_level < 1:
                label_level = 1
            return label_str, label_level

        return style_label, None

    def _last_child_item(
        self, doc: DoclingDocument, parent: NodeItem | None
    ) -> NodeItem | None:
        """Return the last child of a container.

        Args:
            doc: The document being built.
            parent: The container to inspect; None means the document body.

        Returns:
            The resolved last child item, or None if the container is empty.
        """
        container = parent if parent is not None else doc.body
        if not container.children:
            return None
        return container.children[-1].resolve(doc)

    def _is_code_style(self, style: ParagraphStyle | None) -> bool:
        """Return True if a style marks its paragraphs as code.

        Mirrors pandoc's docx reader: the style itself or any ancestor in its
        ``base_style`` chain may carry the code style name/id.

        Args:
            style: The already-resolved paragraph style (resolving it is
                expensive, so callers do it once).

        Returns:
            True if the style chain marks the paragraph as code.
        """
        depth = 0
        while style is not None and depth < self._MAX_STYLE_INHERITANCE_DEPTH:
            name = (style.name or "").strip().lower()
            style_id = (style.style_id or "").strip().lower()
            if name in self._CODE_STYLE_NAMES or style_id in self._CODE_STYLE_IDS:
                return True
            # A malformed basedOn chain can hop to a style type (e.g. a
            # numbering style) that lacks this attribute; getattr keeps
            # the walk safe.
            style = getattr(style, "base_style", None)
            depth += 1
        return False

    def _get_label_and_level(self, paragraph: Paragraph) -> tuple[str, int | None]:
        # Resolve the style once: python-docx's ``paragraph.style`` scans all
        # styles on every access, so re-reading it per predicate is costly.
        style = paragraph.style
        if style is None:
            return "Normal", None

        label: str = style.style_id
        name: str = style.name or ""
        base_style_label: str | None = None
        base_style_name: str | None = None
        if isinstance(base_style := getattr(style, "base_style", None), ParagraphStyle):
            base_style_label = base_style.style_id
            base_style_name = base_style.name

        if not label:
            return "Normal", None

        if ":" in label:
            parts = label.split(":")
            if len(parts) == 2:
                return parts[0], self._str_to_int(parts[1], None)

        # Check if this is a heading style
        is_heading = (
            "heading" in label.lower()
            or "heading" in name.lower()
            or (base_style_label and "heading" in base_style_label.lower())
            or (base_style_name and "heading" in base_style_name.lower())
        )

        if is_heading:
            # First try to get the level from outlineLvl (authoritative source)
            outline_level = self._get_outline_level_from_style(paragraph)
            if outline_level is not None:
                return "Heading", outline_level

            # Fall back to parsing level from style name
            if "heading" in label.lower():
                return self._get_heading_and_level(label)
            if "heading" in name.lower():
                return self._get_heading_and_level(name)
            if base_style_label and "heading" in base_style_label.lower():
                return self._get_heading_and_level(base_style_label)
            if base_style_name and "heading" in base_style_name.lower():
                return self._get_heading_and_level(base_style_name)

        if self._is_code_style(style):
            return "Code", None

        return label, None

    @classmethod
    def _get_format_from_run(
        cls, run: Run, paragraph: Paragraph | None = None
    ) -> Formatting | None:
        is_bold = run.bold

        if not is_bold:
            try:
                # Check the raw XML of the run itself for <w:b> tags
                if hasattr(run, "_element") and run._element is not None:
                    b_tags = run._element.xpath(".//w:b | .//w:bCs")
                    for b in b_tags:
                        val = b.get(
                            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"
                        )
                        if val not in ["0", "false"]:
                            is_bold = True
                            break

                # Check the paragraph's direct formatting properties
                if (
                    not is_bold
                    and hasattr(run, "_parent")
                    and hasattr(run._parent, "_element")
                ):
                    pPr_b = run._parent._element.xpath(
                        "./w:pPr/w:rPr/w:b | ./w:pPr/w:rPr/w:bCs"
                    )
                    for b in pPr_b:
                        val = b.get(
                            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"
                        )
                        if val not in ["0", "false"]:
                            is_bold = True
                            break

                # Recursively climb the paragraph's Master Style Sheet
                if (
                    not is_bold
                    and paragraph is not None
                    and getattr(paragraph, "style", None)
                ):
                    current_style = paragraph.style
                    while current_style is not None:
                        if hasattr(current_style, "font") and current_style.font.bold:
                            is_bold = True
                            break

                        current_style = getattr(current_style, "base_style", None)
            except Exception:
                pass

        is_bold = is_bold or False

        is_italic = run.italic or False
        is_strikethrough = run.font.strike or False
        # Convert any non-None underline value to True
        is_underline = bool(run.underline is not None and run.underline)
        is_sub = run.font.subscript or False
        is_sup = run.font.superscript or False
        script = Script.SUB if is_sub else Script.SUPER if is_sup else Script.BASELINE

        return Formatting(
            bold=is_bold,
            italic=is_italic,
            underline=is_underline,
            strikethrough=is_strikethrough,
            script=script,
        )

    def _get_hyperlink_target(self, hyperlink: Hyperlink) -> AnyUrl | Path | None:
        """Resolve a hyperlink's address to a URL or a local path.

        Addresses without a URL scheme are treated as (relative) filesystem
        paths. Addresses with a scheme are parsed as URLs.

        Malformed URLs (e.g. an address containing spaces) are handled
        gracefully: the invalid target is dropped and ``None`` is returned so
        that the link text is still preserved and conversion of the rest of the
        document continues. This avoids a single bad hyperlink aborting the
        whole conversion.

        Args:
            hyperlink: The DOCX hyperlink whose address should be resolved.

        Returns:
            An ``AnyUrl`` for a valid URL, a ``Path`` for a scheme-less address,
            or ``None`` when there is no address or the URL is malformed.
        """
        if hyperlink.address:
            if not urlparse(hyperlink.address).scheme:
                return Path(hyperlink.address)
            try:
                return AnyUrl(hyperlink.address)
            except ValidationError:
                _log.warning(
                    "Skipping malformed hyperlink address: %r", hyperlink.address
                )
                return None

        return None

    def _iter_paragraph_content(
        self, paragraph: Paragraph
    ) -> list[tuple[str, Formatting | None, AnyUrl | Path | None]]:
        if not hasattr(paragraph, "_p"):
            return []

        content: list[tuple[str, Formatting | None, AnyUrl | Path | None]] = []

        def _get_children_recursive(node):
            for child in node:
                tag_name = etree.QName(child).localname
                if tag_name in {"smartTag", "customXml", "ins", "fldSimple"}:
                    yield from _get_children_recursive(child)
                else:
                    yield child

        for child in _get_children_recursive(paragraph._p):
            tag_name = etree.QName(child).localname

            if tag_name == "sdt":
                text = "".join(
                    child.xpath(
                        ".//w:sdtContent//w:t/text()",
                        namespaces=MsWordDocumentBackend._BLIP_NAMESPACES,
                    )
                )
                if len(text) == 0:
                    continue

                runs = child.xpath(
                    ".//w:sdtContent//w:r",
                    namespaces=MsWordDocumentBackend._BLIP_NAMESPACES,
                )
                fmt = (
                    self._get_format_from_run(Run(runs[0], paragraph), paragraph)
                    if runs
                    else None
                )
                content.append((text, fmt, None))
                continue

            if tag_name not in {"r", "hyperlink"}:
                continue

            item = (
                Run(child, paragraph)
                if tag_name == "r"
                else Hyperlink(child, paragraph)
            )

            if isinstance(item, Hyperlink):
                content.append(
                    (
                        item.text,
                        (
                            self._get_format_from_run(item.runs[0], paragraph)
                            if item.runs and len(item.runs) > 0
                            else None
                        ),
                        self._get_hyperlink_target(item),
                    )
                )
            elif isinstance(item, Run):
                content.append(
                    (item.text, self._get_format_from_run(item, paragraph), None)
                )

        return content

    def _get_paragraph_text(self, paragraph: Paragraph) -> str:
        if not hasattr(paragraph, "iter_inner_content") or not hasattr(paragraph, "_p"):
            return paragraph.text

        return "".join(
            text
            for text, _format, _hyperlink in self._iter_paragraph_content(paragraph)
        )

    def _get_paragraph_elements(self, paragraph: Paragraph):
        """
        Extract paragraph elements along with their formatting and hyperlink
        """

        # for now retain empty paragraphs for backwards compatibility:
        if self._get_paragraph_text(paragraph).strip() == "":
            return [("", None, None)]

        paragraph_elements: list[
            tuple[str, Formatting | None, AnyUrl | Path | None]
        ] = []
        group_text = ""
        previous_format = None

        # Iterate over the runs of the paragraph and group them by format
        for text, format, hyperlink in self._iter_paragraph_content(paragraph):
            if (len(text.strip()) and format != previous_format) or (
                hyperlink is not None
            ):
                # If the style changes for a non empty text, add the previous group
                if len(group_text.strip()) > 0:
                    paragraph_elements.append(
                        (group_text.strip(), previous_format, None)
                    )
                group_text = ""

                # If there is a hyperlink, add it immediately
                if hyperlink is not None:
                    paragraph_elements.append((text.strip(), format, hyperlink))
                    text = ""
                else:
                    previous_format = format

            group_text += text

        # Format the last group
        if len(group_text.strip()) > 0:
            paragraph_elements.append((group_text.strip(), previous_format, None))

        return paragraph_elements

    def _has_checkbox(self, element: BaseOxmlElement) -> bool:
        """Check if a paragraph element contains a checkbox.

        Args:
            element: The paragraph element to check for checkbox presence.

        Returns:
            True if the element contains a checkbox, False otherwise.
        """
        try:
            checkboxes = element.findall(
                f".//{{{self._BLIP_NAMESPACES['w14']}}}checkbox"
            )
            return len(checkboxes) > 0
        except (AttributeError, TypeError):
            return False

    def _is_checkbox_checked(self, element: BaseOxmlElement) -> bool:
        """Check if a checkbox in the paragraph is checked.

        Args:
            element: The paragraph element containing the checkbox.

        Returns:
            True if checked (w14:checked val="1"), False if unchecked
                (val="0" or missing).
        """
        w14_ns = self._BLIP_NAMESPACES["w14"]
        checkboxes = element.findall(f".//{{{w14_ns}}}checkbox")
        if not checkboxes:
            return False

        checkbox = checkboxes[0]
        checked_elem = checkbox.find(f".//{{{w14_ns}}}checked")

        if checked_elem is not None:
            val = checked_elem.get(f"{{{w14_ns}}}val")
            return val == "1"

        return False

    def _get_checkbox_label(self, element: BaseOxmlElement) -> DocItemLabel | None:
        """Get the appropriate checkbox label for a paragraph element.

        Args:
            element: The paragraph element to check for checkbox.

        Returns:
            DocItemLabel.CHECKBOX_SELECTED if checked, DocItemLabel.CHECKBOX_UNSELECTED
                if unchecked, or None if no checkbox is present.
        """
        if not self._has_checkbox(element):
            return None

        if self._is_checkbox_checked(element):
            return DocItemLabel.CHECKBOX_SELECTED
        else:
            return DocItemLabel.CHECKBOX_UNSELECTED

    def _clean_checkbox_symbols(self, text: str) -> str:
        """Remove checkbox symbols from text.

        Removes common checkbox symbols like ☐, ☑, ☒ from the beginning of text.

        Args:
            text: The text string to clean.

        Returns:
            The text with checkbox symbols removed from the beginning.
        """
        # Common checkbox symbols in docx documents
        checkbox_symbols = ["☐", "☑", "☒", "□", "■", "▪", "▫"]

        text = text.strip()
        for symbol in checkbox_symbols:
            if text.startswith(symbol):
                text = text[len(symbol) :].strip()
                break

        return text

    def _get_paragraph_position(self, paragraph_element):
        """Extract vertical position information from paragraph element."""
        # First try to directly get the index from w:p element that has an order-related attribute
        if (
            hasattr(paragraph_element, "getparent")
            and paragraph_element.getparent() is not None
        ):
            parent = paragraph_element.getparent()
            # Get all paragraph siblings
            paragraphs = [
                p for p in parent.getchildren() if etree.QName(p).localname == "p"
            ]
            # Find index of current paragraph within its siblings
            try:
                paragraph_index = paragraphs.index(paragraph_element)
                return paragraph_index  # Use index as position for consistent ordering
            except ValueError:
                pass

        # Look for position hints in element attributes and ancestor elements
        for elem in (*[paragraph_element], *paragraph_element.iterancestors()):
            # Check for direct position attributes
            for attr_name in ["y", "top", "positionY", "y-position", "position"]:
                value = elem.get(attr_name)
                if value:
                    try:
                        # Remove any non-numeric characters (like 'pt', 'px', etc.)
                        clean_value = re.sub(r"[^0-9.]", "", value)
                        if clean_value:
                            return float(clean_value)
                    except (ValueError, TypeError):
                        pass

            # Check for position in transform attribute
            transform = elem.get("transform")
            if transform:
                # Extract translation component from transform matrix
                match = re.search(r"translate\([^,]+,\s*([0-9.]+)", transform)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        pass

            # Check for anchors or relative position indicators in Word format
            # 'dist' attributes can indicate relative positioning
            for attr_name in ["distT", "distB", "anchor", "relativeFrom"]:
                if elem.get(attr_name) is not None:
                    return elem.sourceline  # Use the XML source line number as fallback

        # For VML shapes, look for specific attributes
        for ns_uri in paragraph_element.nsmap.values():
            if "vml" in ns_uri:
                # Try to extract position from style attribute
                style = paragraph_element.get("style")
                if style:
                    match = re.search(r"top:([0-9.]+)pt", style)
                    if match:
                        try:
                            return float(match.group(1))
                        except ValueError:
                            pass

        # If no better position indicator found, use XML source line number as proxy for order
        return (
            paragraph_element.sourceline
            if hasattr(paragraph_element, "sourceline")
            else None
        )

    def _collect_textbox_paragraphs(self, textbox_elements):
        """Collect and organize paragraphs from textbox elements."""
        processed_paragraphs = []
        container_paragraphs = {}

        for element in textbox_elements:
            element_id = id(element)
            # Skip if we've already processed this exact element
            if element_id in processed_paragraphs:
                continue

            tag_name = etree.QName(element).localname
            processed_paragraphs.append(element_id)

            # Handle paragraphs directly found (VML textboxes)
            if tag_name == "p":
                # Find the containing textbox or shape element
                container_id = None
                for ancestor in element.iterancestors():
                    if any(ns in ancestor.tag for ns in ["textbox", "shape", "txbx"]):
                        container_id = id(ancestor)
                        break

                if container_id not in container_paragraphs:
                    container_paragraphs[container_id] = []
                container_paragraphs[container_id].append(
                    (element, self._get_paragraph_position(element))
                )

            # Handle txbxContent elements (Word DrawingML textboxes)
            elif tag_name == "txbxContent":
                paragraphs = element.findall(".//w:p", namespaces=element.nsmap)
                container_id = id(element)
                if container_id not in container_paragraphs:
                    container_paragraphs[container_id] = []

                for p in paragraphs:
                    p_id = id(p)
                    if p_id not in processed_paragraphs:
                        processed_paragraphs.append(p_id)
                        container_paragraphs[container_id].append(
                            (p, self._get_paragraph_position(p))
                        )
            else:
                # Try to extract any paragraphs from unknown elements
                paragraphs = element.findall(".//w:p", namespaces=element.nsmap)
                container_id = id(element)
                if container_id not in container_paragraphs:
                    container_paragraphs[container_id] = []

                for p in paragraphs:
                    p_id = id(p)
                    if p_id not in processed_paragraphs:
                        processed_paragraphs.append(p_id)
                        container_paragraphs[container_id].append(
                            (p, self._get_paragraph_position(p))
                        )

        return container_paragraphs

    def _handle_textbox_content(
        self,
        textbox_elements: list,
        doc: DoclingDocument,
    ) -> list[RefItem]:
        """Process textbox content and add it to the document structure."""
        elem_ref: list[RefItem] = []
        level = self._get_level()
        # Create a textbox group to contain all text from the textbox
        textbox_group = doc.add_group(
            label=GroupLabel.SECTION,
            parent=self.parents[level - 1],
            name="textbox",
            content_layer=self.content_layer,
        )
        elem_ref.append(textbox_group.get_ref())
        # Set this as the current parent to ensure textbox content
        # is properly nested in document structure
        original_parent = self.parents[level]
        self.parents[level] = textbox_group

        # Collect and organize paragraphs
        container_paragraphs = self._collect_textbox_paragraphs(textbox_elements)

        # Process all paragraphs
        all_paragraphs = []

        # Sort paragraphs within each container, then process containers
        for paragraphs in container_paragraphs.values():
            # Sort by vertical position within each container
            sorted_container_paragraphs = sorted(
                paragraphs,
                key=lambda x: (
                    x[1] is None,
                    x[1] if x[1] is not None else float("inf"),
                ),
            )

            # Add the sorted paragraphs to our processing list
            all_paragraphs.extend(sorted_container_paragraphs)

        # Track processed paragraphs to avoid duplicates
        processed_paragraphs = set()

        # Process all the paragraphs
        for p, position in all_paragraphs:
            # Create paragraph object to get text content
            paragraph = Paragraph(p, self.docx_obj)
            text_content = paragraph.text.strip()

            if text_content:
                # Deduplicate AlternateContent by exact text natively
                if text_content in processed_paragraphs:
                    continue
                processed_paragraphs.add(text_content)
            else:
                # Preserve empty regions (images) based on position
                paragraph_id = (text_content, position)
                if paragraph_id in processed_paragraphs:
                    continue
                processed_paragraphs.add(paragraph_id)

            elem_ref.extend(self._handle_text_elements(p, doc))

            # Extract embedded images inside the text box
            tb_drawing_blip = self.blip_xpath_expr(p)
            tb_vml_images = self.vml_imagedata_xpath_expr(p)
            tb_drawingml_els = p.findall(
                ".//w:drawing", namespaces=MsWordDocumentBackend._BLIP_NAMESPACES
            )

            if tb_drawing_blip:
                pics = self._handle_pictures(tb_drawing_blip, doc)
                elem_ref.extend(pics)
            elif tb_vml_images:
                vml_pics = self._handle_vml_pictures(tb_vml_images, doc)
                elem_ref.extend(vml_pics)
            elif tb_drawingml_els:
                self._handle_drawingml(doc=doc, drawingml_els=tb_drawingml_els)

        # Restore original parent
        self.parents[level] = original_parent
        return elem_ref

    def _handle_equations_in_text(self, element, text):
        only_texts = []
        only_equations = []
        texts_and_equations = []

        # Collect oMath elements and text runs from the paragraph.
        # Use direct children iteration first; fall back to deep iteration
        # only if no oMath elements are found at the direct level.
        direct_omaths = [
            child
            for child in element
            if "oMath" in child.tag and "oMathPara" not in child.tag
        ]

        if direct_omaths:
            # Iterate direct children to preserve sibling order and avoid
            # processing nested oMath descendants of an already-converted node.
            for child in element:
                if "oMath" in child.tag and "oMathPara" not in child.tag:
                    latex_equation = str(oMath2Latex(child)).strip()
                    if len(latex_equation) > 0:
                        only_equations.append(
                            self.equation_bookends.format(EQ=latex_equation)
                        )
                        texts_and_equations.append(
                            self.equation_bookends.format(EQ=latex_equation)
                        )
                else:
                    # Collect text from non-math children (e.g. <w:r> runs)
                    for t_elem in child.iter():
                        t_tag = etree.QName(t_elem).localname
                        if t_tag == "t" and "math" not in t_elem.tag:
                            if isinstance(t_elem.text, str):
                                only_texts.append(t_elem.text)
                                texts_and_equations.append(t_elem.text)
        else:
            # Original deep-iteration fallback for nested oMath (e.g.
            # inside oMathPara or other wrapper elements).
            for subt in element.iter():
                tag_name = etree.QName(subt).localname
                if tag_name == "t" and "math" not in subt.tag:
                    if isinstance(subt.text, str):
                        only_texts.append(subt.text)
                        texts_and_equations.append(subt.text)
                elif "oMath" in subt.tag and "oMathPara" not in subt.tag:
                    latex_equation = str(oMath2Latex(subt)).strip()
                    if len(latex_equation) > 0:
                        only_equations.append(
                            self.equation_bookends.format(EQ=latex_equation)
                        )
                        texts_and_equations.append(
                            self.equation_bookends.format(EQ=latex_equation)
                        )

        if len(only_equations) < 1:
            return text, []

        if (
            re.sub(r"\s+", "", "".join(only_texts)).strip()
            != re.sub(r"\s+", "", text).strip()
        ):
            # If we are not able to reconstruct the initial raw text
            # do not try to parse equations and return the original
            return text, []

        # Insert equations into original text
        # This is done to preserve white space structure
        output_text = ""
        text_pos = 0

        for substr in texts_and_equations:
            if len(substr) == 0:
                continue
            if substr.startswith("<eq>"):
                # This is an equation - insert it directly
                output_text += substr
            else:
                # This is a text fragment - find it in original text
                pos = text.find(substr, text_pos)
                if pos >= 0:
                    output_text += substr
                    text_pos = pos + len(substr)
                else:
                    # Fallback: if not found, just append it
                    output_text += substr

        return output_text, only_equations

    def _create_or_reuse_parent(
        self,
        *,
        doc: DoclingDocument,
        prev_parent: NodeItem | None,
        paragraph_elements: list,
    ) -> NodeItem | None:
        return (
            doc.add_inline_group(parent=prev_parent, content_layer=self.content_layer)
            if len(paragraph_elements) > 1
            else prev_parent
        )

    def _handle_text_elements(
        self,
        element: BaseOxmlElement,
        doc: DoclingDocument,
        skip_empty_text: bool = False,
    ) -> list[RefItem]:
        elem_ref: list[RefItem] = []
        paragraph = Paragraph(element, self.docx_obj)
        paragraph_elements = self._get_paragraph_elements(paragraph)
        text, equations = self._handle_equations_in_text(
            element=element, text=self._get_paragraph_text(paragraph)
        )

        if text is None:
            return elem_ref
        # Kept unstripped: code blocks preserve leading indentation.
        raw_paragraph_text = text
        text = text.strip()

        # Track the paragraph element ID for comment linking
        para_element_id = id(element)
        comment_ids = self._get_comment_ids_for_element(element)

        # Check if this paragraph contains a checkbox
        checkbox_label = self._get_checkbox_label(element)

        # Common styles for bullet and numbered lists.
        # "List Bullet", "List Number", "List Paragraph"
        # Identify whether list is a numbered list or not
        p_style_id, p_level = self._get_label_and_level(paragraph)
        numid, ilevel = self._get_numId_and_ilvl(paragraph)

        if numid == 0:
            numid = None

        # Handle lists
        if (
            numid is not None
            and ilevel is not None
            and p_style_id not in ["Title", "Heading", "Code"]
        ):
            # Check if this is actually a numbered list by examining the numFmt
            is_numbered = self._has_visible_numbering_format(numid, ilevel)

            # If there are equations in the list item, handle them specially
            if len(equations) > 0:
                li = self._add_list_item_with_equations(
                    doc=doc,
                    numid=numid,
                    ilevel=ilevel,
                    text=text,
                    equations=equations,
                    is_numbered=is_numbered,
                )
            else:
                li = self._add_list_item(
                    doc=doc,
                    numid=numid,
                    ilevel=ilevel,
                    elements=paragraph_elements,
                    is_numbered=is_numbered,
                )
            elem_ref.extend(li)  # MUST BE REF!!!
            self._update_history(p_style_id, p_level, numid, ilevel)
            return elem_ref
        elif (
            self._prev_numid() is not None
            and p_style_id not in ["Title", "Heading"]
            and (numid is None or p_style_id == "Code")
        ):  # Close list. A Code paragraph after a list must close it even if it
            # carries a stray/inherited numId, then be re-parented at body level
            # by the Code branch below (otherwise it nests inside the ListGroup).
            self.last_numid = self._prev_numid()
            if text and text.strip():
                # Substantive body text breaks list continuity
                self._clear_list_group_cache()
            else:
                # Empty/whitespace paragraph — cache the group for potential reuse
                if self.level_at_new_list and self.level_at_new_list in self.parents:
                    parent_item = self.parents.get(self.level_at_new_list)
                    if isinstance(parent_item, ListGroup):
                        self.last_list_group = parent_item
                        self.last_list_group_numid = self.last_numid
                        self.last_list_group_parent = self.parents.get(
                            self.level_at_new_list - 1
                        )

            if self.level_at_new_list:
                for key in range(len(self.parents)):
                    if key >= self.level_at_new_list:
                        self.parents[key] = None
                self.level = self.level_at_new_list - 1
                self.level_at_new_list = None
            else:
                for key in range(len(self.parents)):
                    self.parents[key] = None
                self.level = 0

        if p_style_id in ["Title"]:
            for key in range(len(self.parents)):
                self.parents[key] = None
            te = doc.add_text(
                parent=None,
                label=DocItemLabel.TITLE,
                text=text,
                content_layer=self.content_layer,
            )
            self.parents[0] = te
            elem_ref.append(te.get_ref())
        elif "Heading" in p_style_id:
            is_numbered_style = self._is_numbered_heading(paragraph)
            h1 = self._add_heading(doc, p_level, text, is_numbered_style)
            elem_ref.extend(h1)

        elif len(equations) > 0:
            if (paragraph.text is None or len(paragraph.text.strip()) == 0) and len(
                text
            ) > 0:
                # Standalone equation(s) — emit each as a separate formula
                level = self._get_level()
                parent = self.parents[level - 1]
                if len(equations) > 1:
                    for eq in equations:
                        eq_text = eq.replace("<eq>", "").replace("</eq>", "").strip()
                        if len(eq_text) > 0:
                            t1 = doc.add_text(
                                label=DocItemLabel.FORMULA,
                                parent=parent,
                                text=eq_text,
                                content_layer=self.content_layer,
                            )
                            elem_ref.append(t1.get_ref())
                else:
                    t1 = doc.add_text(
                        label=DocItemLabel.FORMULA,
                        parent=parent,
                        text=text.replace("<eq>", "").replace("</eq>", ""),
                        content_layer=self.content_layer,
                    )
                    elem_ref.append(t1.get_ref())
            else:
                # Inline equation
                level = self._get_level()
                inline_equation = doc.add_inline_group(
                    parent=self.parents[level - 1], content_layer=self.content_layer
                )
                elem_ref.append(inline_equation.get_ref())

                self._add_inline_equations_to_parent(
                    doc=doc,
                    parent=inline_equation,
                    text=text,
                    equations=equations,
                    elem_ref=elem_ref,
                )

        elif p_style_id == "Code" and not checkbox_label:
            # Code-styled checkboxes fall through to the text branch,
            # keeping their label.
            level = self._get_level()
            parent = self.parents[level - 1]
            code_text = raw_paragraph_text.rstrip()
            # Merge into the previous code block only when it is the parent's
            # last child (any intervening element breaks the block), the most
            # recent text item (a resumed list adds items without touching
            # this parent), and in the same content layer.
            last_item = self._last_child_item(doc, parent)
            merge_target = None if self._force_new_code_block else last_item
            if (
                isinstance(merge_target, CodeItem)
                and merge_target.content_layer == self.content_layer
                and doc.texts
                and doc.texts[-1] is merge_target
            ):
                if code_text:
                    joiner = "\n" * (self._pending_code_blank_lines + 1)
                    merge_target.text = f"{merge_target.text}{joiner}{code_text}"
                    merge_target.orig = f"{merge_target.orig}{joiner}{code_text}"
                    self._pending_code_blank_lines = 0
                else:
                    # Buffered: written only if more code follows, so a
                    # block never ends in blank lines.
                    self._pending_code_blank_lines += 1
                elem_ref.append(merge_target.get_ref())
                self._force_new_code_block = False
            elif text:
                # Start a new block, but never on a leading blank paragraph.
                self._pending_code_blank_lines = 0
                code_item = doc.add_code(
                    text=code_text,
                    orig=code_text,
                    parent=parent,
                    content_layer=self.content_layer,
                )
                elem_ref.append(code_item.get_ref())
                self._force_new_code_block = False
            # A blank that neither starts nor extends a block leaves the
            # barrier armed.

        else:
            level = self._get_level()
            parent = self._create_or_reuse_parent(
                doc=doc,
                prev_parent=self.parents.get(level - 1),
                paragraph_elements=paragraph_elements,
            )
            for text, format, hyperlink in paragraph_elements:
                # Clean checkbox symbols from text if this is a checkbox item
                clean_text = (
                    self._clean_checkbox_symbols(text) if checkbox_label else text
                )

                # Skip empty text items
                if skip_empty_text and (not clean_text or not clean_text.strip()):
                    continue

                text_item = doc.add_text(
                    label=checkbox_label if checkbox_label else DocItemLabel.TEXT,
                    parent=parent,
                    text=clean_text,
                    formatting=format,
                    hyperlink=hyperlink,
                    content_layer=self.content_layer,
                )
                elem_ref.append(text_item.get_ref())

        self._update_history(p_style_id, p_level, numid, ilevel)

        # Store mapping of paragraph element to created items for comment linking
        if elem_ref and para_element_id:
            self.paragraph_to_items[para_element_id] = elem_ref
            if comment_ids:
                self.paragraph_comment_map[para_element_id] = list(comment_ids)

        return elem_ref

    def _add_heading(
        self,
        doc: DoclingDocument,
        curr_level: int | None,
        text: str,
        is_numbered_style: bool = False,
    ) -> list[RefItem]:
        elem_ref: list[RefItem] = []
        level = self._get_level()
        if isinstance(curr_level, int):
            if curr_level > level:
                # add invisible group
                for i in range(level, curr_level):
                    gr1 = doc.add_group(
                        parent=self.parents[i - 1],
                        label=GroupLabel.SECTION,
                        name=f"header-{i}",
                    )
                    elem_ref.append(gr1.get_ref())
                    self.parents[i] = gr1

            elif curr_level < level:
                # remove the tail
                for key in range(len(self.parents)):
                    if key >= curr_level:
                        self.parents[key] = None

            # Defense in depth: ensure level is at least 1
            curr_level = max(1, curr_level)
            current_level = curr_level
            parent_level = curr_level - 1
            add_level = curr_level
        else:
            current_level = self.level
            parent_level = self.level - 1
            add_level = 1

        if is_numbered_style:
            if add_level in self.numbered_headers:
                self.numbered_headers[add_level] += 1
            else:
                self.numbered_headers[add_level] = 1
            text = f"{self.numbered_headers[add_level]} {text}"

            # Reset deeper levels
            next_level = add_level + 1
            while next_level in self.numbered_headers:
                self.numbered_headers[next_level] = 0
                next_level += 1

            # Scan upper levels
            previous_level = add_level - 1
            while previous_level in self.numbered_headers:
                # MSWord convention: no empty sublevels
                # I.e., sub-sub section (2.0.1) without a sub-section (2.1)
                # is processed as 2.1.1
                if self.numbered_headers[previous_level] == 0:
                    self.numbered_headers[previous_level] += 1

                text = f"{self.numbered_headers[previous_level]}.{text}"
                previous_level -= 1

        hd = doc.add_heading(
            parent=self.parents[parent_level],
            text=text,
            level=add_level,
        )
        self.parents[current_level] = hd
        elem_ref.append(hd.get_ref())
        return elem_ref

    def _add_formatted_list_item(
        self,
        doc: DoclingDocument,
        elements: list,
        marker: str,
        enumerated: bool,
        level: int,
    ) -> list[RefItem]:
        elem_ref: list[RefItem] = []
        # This should not happen by construction
        if not isinstance(self.parents[level], ListGroup):
            _log.warning(
                "Parent element of the list item is not a ListGroup. The list item will be ignored."
            )
            return elem_ref
        if not elements:
            return elem_ref

        if len(elements) == 1:
            text, format, hyperlink = elements[0]
            if text:
                doc.add_list_item(
                    marker=marker,
                    enumerated=enumerated,
                    parent=self.parents[level],
                    text=text,
                    formatting=format,
                    hyperlink=hyperlink,
                )
        else:
            new_item = doc.add_list_item(
                marker=marker,
                enumerated=enumerated,
                parent=self.parents[level],
                text="",
            )
            new_parent = doc.add_inline_group(parent=new_item)
            for text, format, hyperlink in elements:
                if text:
                    doc.add_text(
                        label=DocItemLabel.TEXT,
                        parent=new_parent,
                        text=text,
                        formatting=format,
                        hyperlink=hyperlink,
                        content_layer=self.content_layer,
                    )
        return elem_ref

    def _add_list_item_with_marker(
        self,
        doc: DoclingDocument,
        elements: list,
        numid: int,
        ilevel: int,
        is_numbered: bool,
        level: int,
    ) -> None:
        """Resolve enumeration marker and add a formatted list item."""
        if is_numbered:
            self._get_list_counter(numid, ilevel)
            enum_marker = self._build_enum_marker(numid, ilevel)
        else:
            enum_marker = ""
        self._add_formatted_list_item(doc, elements, enum_marker, is_numbered, level)

    def _add_inline_equations_to_parent(
        self,
        *,
        doc: DoclingDocument,
        parent: NodeItem,
        text: str,
        equations: list[str],
        elem_ref: list[RefItem] | None = None,
    ) -> None:
        """Add text and inline equations as children of a parent element.

        This helper method splits text by equation markers and adds alternating
        TEXT and FORMULA elements as children of the given parent. This logic
        is shared between regular paragraphs with inline equations and list items
        with inline equations.

        Args:
            doc: The DoclingDocument being constructed.
            parent: The parent element (inline_group) to add children to.
            text: The paragraph text with equation placeholders (e.g., "<eq>formula</eq>").
            equations: List of equation strings with markers (e.g., ["<eq>A=B</eq>", ...]).
            elem_ref: Optional list to append created element references to.
        """
        text_tmp = text
        for eq in equations:
            if len(text_tmp) == 0:
                break

            split_text_tmp = text_tmp.split(eq.strip(), maxsplit=1)

            pre_eq_text = split_text_tmp[0]
            text_tmp = "" if len(split_text_tmp) == 1 else split_text_tmp[1]

            if len(pre_eq_text) > 0:
                e1 = doc.add_text(
                    label=DocItemLabel.TEXT,
                    parent=parent,
                    text=pre_eq_text,
                    content_layer=self.content_layer,
                )
                if elem_ref is not None:
                    elem_ref.append(e1.get_ref())

            e2 = doc.add_text(
                label=DocItemLabel.FORMULA,
                parent=parent,
                text=eq.replace("<eq>", "").replace("</eq>", ""),
                content_layer=self.content_layer,
            )
            if elem_ref is not None:
                elem_ref.append(e2.get_ref())

        if len(text_tmp) > 0:
            e3 = doc.add_text(
                label=DocItemLabel.TEXT,
                parent=parent,
                text=text_tmp.strip(),
                content_layer=self.content_layer,
            )
            if elem_ref is not None:
                elem_ref.append(e3.get_ref())

    def _manage_list_structure(
        self,
        *,
        doc: DoclingDocument,
        numid: int,
        ilevel: int,
    ) -> tuple[list[RefItem], int]:
        """Manage list structure and return elem_ref and use_level.

        This helper method handles the list group creation and level management
        that is common to both regular list items and list items with equations.
        It determines whether to open a new list, continue an existing one, handle
        indentation changes, or close lists based on the numbering context.

        Args:
            doc: The DoclingDocument being constructed.
            numid: The numbering ID from the DOCX paragraph properties.
            ilevel: The indentation level from the DOCX paragraph properties.

        Returns:
            A tuple containing the list of references to created list groups and
            the level at which the list item should be added.
        """
        elem_ref: list[RefItem] = []
        level = self._get_level()
        prev_indent = self._prev_indent()

        if self._prev_numid() is None or (
            self._prev_numid() == numid and self.level_at_new_list is None
        ):  # Open new list
            self.level_at_new_list = level
            # Only reset counters if this is a truly new list (different numId)
            if self.last_numid != numid:
                self._reset_list_counters_for_new_sequence(numid)

            list_gr = self._get_or_create_list_group(
                doc=doc,
                numid=numid,
                parent=self.parents[level - 1],
                elem_ref=elem_ref,
            )

            self.parents[level] = list_gr
            use_level = level
            self.last_numid = numid

        elif (
            self._prev_numid() == numid
            and self.level_at_new_list is not None
            and prev_indent is not None
            and prev_indent < ilevel
        ):  # Open indented list
            for i in range(
                self.level_at_new_list + prev_indent + 1,
                self.level_at_new_list + ilevel + 1,
            ):
                list_gr1 = doc.add_list_group(
                    name="list",
                    parent=self.parents[i - 1],
                    content_layer=self.content_layer,
                )
                self.parents[i] = list_gr1
                elem_ref.append(list_gr1.get_ref())
            use_level = self.level_at_new_list + ilevel

        elif (
            self._prev_numid() == numid
            and self.level_at_new_list is not None
            and prev_indent is not None
            and ilevel < prev_indent
        ):  # Close list
            for k in self.parents:
                if k > self.level_at_new_list + ilevel:
                    self.parents[k] = None
            use_level = self.level_at_new_list + ilevel

        elif self._prev_numid() == numid and isinstance(
            self.parents.get(level - 1), ListGroup
        ):
            # Continue existing list
            use_level = level - 1

        elif self._prev_numid() != numid or not isinstance(
            self.parents.get(level - 1), ListGroup
        ):
            # New list sequence
            if self.level_at_new_list is not None:
                use_level = self.level_at_new_list + ilevel
                for k in list(self.parents.keys()):
                    if k > use_level:
                        self.parents[k] = None
            else:
                use_level = level
                self.level_at_new_list = use_level

            # Only reset counters if this is a different numId
            if self.last_numid != numid:
                self._reset_list_counters_for_new_sequence(numid)

            list_gr = self._get_or_create_list_group(
                doc=doc,
                numid=numid,
                parent=self.parents[use_level - 1],
                elem_ref=elem_ref,
            )

            self.parents[use_level] = list_gr
            self.last_numid = numid
        else:
            use_level = level - 1

        return elem_ref, use_level

    def _add_list_item(
        self,
        *,
        doc: DoclingDocument,
        numid: int,
        ilevel: int,
        elements: list,
        is_numbered: bool = False,
    ) -> list[RefItem]:
        """Add a regular list item without inline equations.

        Args:
            doc: The DoclingDocument being constructed.
            numid: The numbering ID from the DOCX paragraph properties.
            ilevel: The indentation level from the DOCX paragraph properties.
            elements: List of (text, formatting, hyperlink) tuples representing the paragraph content.
            is_numbered: Whether this is a numbered list (True) or bulleted list (False).

        Returns:
            List of references to created document elements.
        """
        if not elements:
            return []

        elem_ref, use_level = self._manage_list_structure(
            doc=doc, numid=numid, ilevel=ilevel
        )

        if is_numbered:
            self._get_list_counter(numid, ilevel)
            enum_marker = self._build_enum_marker(numid, ilevel)
        else:
            enum_marker = ""

        self._add_formatted_list_item(
            doc, elements, enum_marker, is_numbered, use_level
        )
        return elem_ref

    def _add_list_item_with_equations(
        self,
        *,
        doc: DoclingDocument,
        numid: int,
        ilevel: int,
        text: str,
        equations: list[str],
        is_numbered: bool = False,
    ) -> list[RefItem]:
        """Add a list item that contains inline equations.

        This method handles list items with inline formulas by creating an inline_group
        structure similar to how non-list paragraphs with equations are handled. The text
        is split by equation markers, and alternating TEXT and FORMULA elements are added
        as children of the inline_group.

        Args:
            doc: The DoclingDocument being constructed.
            numid: The numbering ID from the DOCX paragraph properties.
            ilevel: The indentation level from the DOCX paragraph properties.
            text: The paragraph text with equation placeholders (e.g., "<eq>formula</eq>").
            equations: List of equation strings with markers (e.g., ["<eq>A=B</eq>", ...]).
            is_numbered: Whether this is a numbered list (True) or bulleted list (False).

        Returns:
            List of references to created document elements.
        """
        elem_ref, use_level = self._manage_list_structure(
            doc=doc, numid=numid, ilevel=ilevel
        )

        if is_numbered:
            self._get_list_counter(numid, ilevel)
            enum_marker = self._build_enum_marker(numid, ilevel)
        else:
            enum_marker = ""

        if not isinstance(self.parents[use_level], ListGroup):
            _log.warning(
                "Parent element of the list item is not a ListGroup. The list item will be ignored."
            )
            return elem_ref

        list_item = doc.add_list_item(
            marker=enum_marker,
            enumerated=is_numbered,
            parent=self.parents[use_level],
            text="",
        )

        inline_group = doc.add_inline_group(
            parent=list_item,
            content_layer=self.content_layer,
        )

        self._add_inline_equations_to_parent(
            doc=doc,
            parent=inline_group,
            text=text,
            equations=equations,
        )

        return elem_ref

    @staticmethod
    def _group_cell_elements(
        group_name: str,
        doc: DoclingDocument,
        provs_in_cell: list[RefItem],
        docling_table: TableItem,
        content_layer: ContentLayer = ContentLayer.BODY,
    ) -> RefItem:
        group_element = doc.add_group(
            label=GroupLabel.UNSPECIFIED,
            name=group_name,
            parent=docling_table,
            content_layer=content_layer,
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

    def _handle_tables(
        self,
        element: BaseOxmlElement,
        doc: DoclingDocument,
    ) -> list[RefItem]:
        """Convert a ``w:tbl`` element into a Docling table.

        Cells are emitted in a single pass over the rows. A row may start late
        (``w:gridBefore``) or end early (``w:gridAfter``), so a cell's position
        within its row does not identify its layout-grid column; the grid column
        is therefore tracked explicitly while walking the row. A cell with
        ``vMerge="continue"`` holds no content of its own and instead extends the
        cell still open at the same grid column.

        Args:
            element: The ``w:tbl`` element to convert.
            doc: The DoclingDocument being constructed.

        Returns:
            References to the items created for this table.
        """
        elem_ref: list[RefItem] = []
        table: Table = Table(element, self.docx_obj)
        num_rows = len(table.rows)
        num_cols = len(table.columns)
        _log.debug(f"Table grid with {num_rows} rows and {num_cols} columns")

        if num_rows == 1 and num_cols == 1:
            cell_element = table.rows[0].cells[0]
            # In case we have a table of only 1 cell, we consider it furniture
            # And proceed processing the content of the cell as though it's in the document body
            self._clear_list_group_cache()
            # Still a code-block boundary: outside code must not merge with
            # code inside the cell.
            self._force_new_code_block = True
            self._walk_linear(cell_element._element, doc)
            self._force_new_code_block = True
            return elem_ref

        data = TableData(num_rows=num_rows, num_cols=num_cols)
        level = self._get_level()
        docling_table = doc.add_table(
            data=data, parent=self.parents[level - 1], content_layer=self.content_layer
        )
        elem_ref.append(docling_table.get_ref())

        open_cells: dict[int, TableCell] = {}
        for row_idx, row in enumerate(table.rows):
            grid_col = row.grid_cols_before
            for tc in row._tr.tc_lst:
                if grid_col >= num_cols:
                    break
                col_span = tc.grid_span

                spanned = open_cells.get(grid_col)
                if tc.vMerge == ST_Merge.CONTINUE and spanned is not None:
                    spanned.end_row_offset_idx = row_idx + 1
                    spanned.row_span = (
                        spanned.end_row_offset_idx - spanned.start_row_offset_idx
                    )
                    grid_col += col_span
                    continue

                cell = _Cell(tc, table)

                # Detect equations in cell text
                text, equations = self._handle_equations_in_text(
                    element=cell._element, text=cell.text
                )
                if len(equations) == 0:
                    text = cell.text
                else:
                    text = text.replace("<eq>", "$").replace("</eq>", "$")

                provs_in_cell: list[RefItem] = []
                ref_for_rich_cell: RefItem | None = None
                rich_table_cell: bool = self._is_rich_table_cell(cell)

                if rich_table_cell:
                    with self._isolated_list_context():
                        _, provs_in_cell = self._walk_linear(cell._element, doc)
                _log.debug(f"Table cell {row_idx},{grid_col} rich? {rich_table_cell}")

                new_cell: TableCell
                if len(provs_in_cell) > 0:
                    # Cell has multiple elements, we need to group them
                    group_name = (
                        f"rich_cell_group_{len(doc.tables)}_{grid_col}_{row_idx}"
                    )
                    ref_for_rich_cell = MsWordDocumentBackend._group_cell_elements(
                        group_name,
                        doc,
                        provs_in_cell,
                        docling_table,
                        content_layer=self.content_layer,
                    )

                if rich_table_cell and ref_for_rich_cell is not None:
                    new_cell = RichTableCell(
                        text=text,
                        row_span=1,
                        col_span=col_span,
                        start_row_offset_idx=row_idx,
                        end_row_offset_idx=row_idx + 1,
                        start_col_offset_idx=grid_col,
                        end_col_offset_idx=grid_col + col_span,
                        column_header=row_idx == 0,
                        row_header=False,
                        ref=ref_for_rich_cell,  # points to an artificial group around children
                    )
                else:
                    new_cell = TableCell(
                        text=text,
                        row_span=1,
                        col_span=col_span,
                        start_row_offset_idx=row_idx,
                        end_row_offset_idx=row_idx + 1,
                        start_col_offset_idx=grid_col,
                        end_col_offset_idx=grid_col + col_span,
                        column_header=row_idx == 0,
                        row_header=False,
                    )

                doc.add_table_cell(table_item=docling_table, cell=new_cell)
                open_cells[grid_col] = new_cell
                grid_col += col_span
        return elem_ref

    def _has_blip(self, element: BaseOxmlElement) -> bool:
        """Check if a docx element holds any BLIP as a child.

        Args:
            element: a docx element

        Returns:
            Whether the element contains a BLIP as a direct child.
        """

        for item in element:
            if self.blip_xpath_expr(item):
                return True
            if item.findall(
                ".//w:drawing", namespaces=MsWordDocumentBackend._BLIP_NAMESPACES
            ):
                return True

        return False

    def _is_rich_table_cell(self, cell: _Cell) -> bool:
        """Determine whether a docx cell should be parsed as a Docling RichTableCell.

        A docx cell can hold rich content and be parsed with a Docling RichTableCell.
        However, this requires walking through the lxml elements and creating
        node items. If the cell holds only plain text, a TableCell, the parsing
        is simpler and using a TableCell is prefered.

        Plain text means:
        - The cell has only one paragraph
        - The paragraph consists solely of runs with no run properties
          (no need of Docling formatting).
        - No other block-level elements are present inside the cell element.

        Args:
            cell: A docx cell

        Returns:
            Whether the docx cell should be parsed as RichTableCell
        """
        tc = cell._tc

        # must contain only one paragraph
        paragraphs = list(tc.iterchildren(f"{self._W_NS_CLARK}p"))
        if len(paragraphs) > 1:
            return True

        # no other content
        allowed_tags = {"p", "tcPr"}  # paragraph or table-cell properties
        for child in tc:
            tag = child.tag.split("}")[-1]
            if tag not in allowed_tags:
                return True
        if self._has_blip(tc):
            return True

        # paragraph must contain runs with no run-properties
        for para in paragraphs:
            runs = list(para.iterchildren(f"{self._W_NS_CLARK}r"))
            for rn in runs:
                item: Run = Run(rn, self.docx_obj)
                if item is not None:
                    fm = MsWordDocumentBackend._get_format_from_run(item)
                    if fm != Formatting():
                        return True

        # Walk a non-empty code-styled cell as rich content so it can emit a
        # CodeItem; the font fallback never fires inside cells.
        if paragraphs:
            first_para = Paragraph(paragraphs[0], self.docx_obj)
            if first_para.text.strip() and self._is_code_style(first_para.style):
                return True

        # All checks passed: plain text only
        return False

    def _get_image_from_relationship(
        self, element: Any, rel_attr: str, image_type: str = "image"
    ) -> bytes | None:
        """Get image data from a relationship ID.

        Args:
            element: The XML element containing the relationship reference
            rel_attr: The attribute name for the relationship ID
            image_type: Type of image for warning messages

        Returns:
            Image data as bytes, or None if not found or external
        """
        image_data: bytes | None = None
        rId = element.get(rel_attr)
        active_part = getattr(self, "current_part", self.docx_obj.part)
        if rId and rId in active_part.rels:
            rel = active_part.rels[rId]
            if rel.is_external:
                warnings.warn(
                    f"Skipping external {image_type} reference: {rel.target_ref}",
                    UserWarning,
                    stacklevel=2,
                )
                return None
            # Access the image part using the relationship ID
            image_part = rel.target_part
            image_data = image_part.blob  # Get the binary image data
        return image_data

    def _add_picture_to_doc(
        self,
        doc: DoclingDocument,
        parent: NodeItem | None,
        pil_image: Image.Image | None,
        is_spacer: bool = False,
    ) -> RefItem:
        """Add a picture element to the document.

        Args:
            doc: The DoclingDocument being constructed
            parent: Parent node for the picture
            pil_image: PIL Image object, or None for placeholder
            is_spacer: Flag indicating whether the parsed image is a layout spacer artifact to be excluded

        Returns:
            Reference to the added picture element
        """
        target_layer = ContentLayer.INVISIBLE if is_spacer else self.content_layer
        if pil_image is not None:
            p = doc.add_picture(
                parent=parent,
                image=ImageRef.from_pil(image=pil_image, dpi=72),
                caption=None,
                content_layer=target_layer,
            )
        else:
            p = doc.add_picture(
                parent=parent,
                caption=None,
                content_layer=target_layer,
            )
        return p.get_ref()

    def _convert_elements_via_docx(
        self, elements: Any | list[Any], element_tag: str | list[str] | None = None
    ) -> Image.Image | None:
        """Convert XML element(s) to image by rendering through DOCX->PDF->PNG.

        This method creates a temporary DOCX with the specified element(s),
        converts it to PDF using LibreOffice, then renders to PNG.

        Args:
            elements: Single XML element or list of elements to convert
            element_tag: Tag name(s) to look for in parent hierarchy when processing a
                single element. If None, elements are added directly without searching
                for parent.

        Returns:
            PIL Image object, or None if conversion failed
        """
        # Initialize converter if needed
        if (
            self.docx_to_pdf_converter is None
            and self.docx_to_pdf_converter_init is False
        ):
            self.docx_to_pdf_converter = get_docx_to_pdf_converter()
            self.docx_to_pdf_converter_init = True

        if self.docx_to_pdf_converter is None:
            return None

        try:
            # Create a temporary document with just these elements
            temp_doc = self.load_msword_file(self.path_or_stream, self.document_hash)
            body = temp_doc._element.body
            for child in list(body):
                body.remove(child)

            # Add elements to empty document
            new_para = temp_doc.add_paragraph()
            new_r = new_para.add_run()

            # Handle list of elements (e.g., multiple DrawingML elements)
            if isinstance(elements, list):
                for elem in elements:
                    new_r._r.append(deepcopy(elem))
            else:
                # Handle single element - find parent if element_tag specified
                if element_tag is not None:
                    if isinstance(element_tag, str):
                        element_tag = [element_tag]

                    parent_elem = elements
                    while parent_elem is not None:
                        tag = etree.QName(parent_elem).localname
                        if tag in element_tag:
                            new_r._r.append(deepcopy(parent_elem))
                            break
                        parent_elem = parent_elem.getparent()

                    if parent_elem is None:
                        return None
                else:
                    # Add element directly
                    new_r._r.append(deepcopy(elements))

            # Convert DOCX->PDF->PNG
            pil_image = get_pil_from_dml_docx(
                temp_doc, converter=self.docx_to_pdf_converter
            )
            return pil_image
        except Exception as e:
            _log.debug(f"Element conversion via DOCX failed: {e}")
            return None

    def _handle_pictures(
        self, drawing_blip: Any, doc: DoclingDocument
    ) -> list[RefItem]:
        """Handle DrawingML pictures with blip elements."""
        elem_ref: list[RefItem] = []
        if drawing_blip:
            level = self._get_level()
            parent: NodeItem | None = (
                self.parents[level - 1]
                if len(drawing_blip) == 1
                else doc.add_group(
                    label=GroupLabel.PICTURE_AREA,
                    parent=self.parents[level - 1],
                    content_layer=self.content_layer,
                )
            )
            for image in drawing_blip:
                image_data = self._get_image_from_relationship(
                    image,
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed",
                    "image",
                )
                pil_image: Image.Image | None = None

                if image_data is None:
                    _log.warning("Warning: image cannot be found")
                else:
                    try:
                        image_bytes = BytesIO(image_data)
                        pil_image = Image.open(image_bytes)
                        # Try to ensure the image is usable by converting to PNG
                        # This will fail for WMF/EMF files that PIL can't render
                        test_bytes = BytesIO()
                        pil_image.save(test_bytes, format="PNG")
                        test_bytes.seek(0)
                        pil_image = Image.open(test_bytes)
                    except (UnidentifiedImageError, OSError) as e:
                        _log.warning(f"Warning: image cannot be loaded by Pillow: {e}")
                        pil_image = None

                if pil_image is None and image is not None and image_data is not None:
                    _log.debug(
                        "Direct PIL loading failed, trying DOCX conversion via LibreOffice"
                    )
                    pil_image = self._convert_elements_via_docx(
                        image, ["drawing", "pict"]
                    )

                is_spacer = self._is_invisible_spacer(pil_image)
                elem_ref.append(
                    self._add_picture_to_doc(
                        doc, parent, pil_image, is_spacer=is_spacer
                    )
                )

        return elem_ref

    def _handle_vml_pictures(
        self, vml_imagedatas: Any, doc: DoclingDocument
    ) -> list[RefItem]:
        """Handle VML (Vector Markup Language) images.

        VML images are legacy format images often used for embedded objects like Visio
        drawings, charts, etc. They use v:imagedata elements instead of a:blip
        elements.

        For EMF/WMF formats that PIL cannot render, we use the same approach as
        DrawingML: create a temporary DOCX with the VML element, convert to PDF via
        LibreOffice, then render to PNG.

        Args:
            vml_imagedatas: List of v:imagedata elements
            doc: The DoclingDocument being constructed

        Returns:
            List of RefItem references to the added picture elements
        """
        elem_ref: list[RefItem] = []
        if vml_imagedatas:
            level = self._get_level()
            parent: NodeItem | None = (
                self.parents[level - 1]
                if len(vml_imagedatas) == 1
                else doc.add_group(
                    label=GroupLabel.PICTURE_AREA,
                    parent=self.parents[level - 1],
                    content_layer=self.content_layer,
                )
            )
            for imagedata in vml_imagedatas:
                image_data = self._get_image_from_relationship(
                    imagedata,
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id",
                    "VML image",
                )
                pil_image: Image.Image | None = None

                if image_data is None:
                    _log.warning("Warning: VML image cannot be found")
                else:
                    try:
                        image_bytes = BytesIO(image_data)
                        pil_image = Image.open(image_bytes)
                        test_bytes = BytesIO()
                        pil_image.save(test_bytes, format="PNG")
                        test_bytes.seek(0)
                        pil_image = Image.open(test_bytes)
                    except (UnidentifiedImageError, OSError, Exception) as e:
                        _log.debug(
                            f"Direct PIL loading failed: {e}, trying DOCX conversion"
                        )
                        pil_image = None

                    if pil_image is None and image_data is not None:
                        pil_image = self._convert_elements_via_docx(
                            imagedata, ["object", "pict"]
                        )
                        if pil_image is None:
                            _log.warning(
                                "Warning: VML image cannot be loaded. "
                                "Install LibreOffice for better VML/EMF/WMF support."
                            )

                is_spacer = self._is_invisible_spacer(pil_image)
                elem_ref.append(
                    self._add_picture_to_doc(
                        doc, parent, pil_image, is_spacer=is_spacer
                    )
                )

        return elem_ref

    def _handle_drawingml(self, doc: DoclingDocument, drawingml_els: Any):
        """Handle DrawingML elements by converting to image via DOCX->PDF->PNG.

        Blip-less DrawingML shapes (e.g., SmartArt, WordprocessingML shapes) need
        to be rendered through LibreOffice to extract as images. Native charts are
        handled separately by the _handle_chart method.

        Args:
            doc: The DoclingDocument being constructed
            drawingml_els: List of DrawingML elements to process
        """
        level = self._get_level()
        parent = self.parents[level - 1]

        try:
            pil_image = self._convert_elements_via_docx(drawingml_els, element_tag=None)
            if pil_image is None:
                raise UnidentifiedImageError

            is_spacer = self._is_invisible_spacer(pil_image)
            self._add_picture_to_doc(doc, parent, pil_image, is_spacer=is_spacer)

        except (UnidentifiedImageError, OSError):
            _log.warning("Warning: DrawingML image cannot be loaded by Pillow")
            self._add_picture_to_doc(doc, parent, None)

        return

    def _is_chart_drawing(self, drawing_el: Any) -> bool:
        """Return True when a ``w:drawing`` embeds a native chart.

        A charted drawing holds a graphic frame whose ``c:chart`` element
        references a ``word/charts/chartN.xml`` part (rather than an ``a:blip``
        image or a shape).
        """
        return (
            drawing_el.find(".//c:chart", namespaces=self._BLIP_NAMESPACES) is not None
        )

    def _resolve_chart_root(self, drawing_el: Any) -> Any | None:
        """Resolve a charted ``w:drawing`` to the root of its chart part.

        The drawing's ``c:chart@r:id`` is a relationship into the main document
        part; we follow it to the ``chartN.xml`` payload and parse it with the
        hardened XML parser. Returns None when the relationship or payload is
        missing or malformed.
        """
        chart_ref = drawing_el.find(".//c:chart", namespaces=self._BLIP_NAMESPACES)
        if chart_ref is None:
            return None
        rid = chart_ref.get(f"{{{self._BLIP_NAMESPACES['r']}}}id")
        if not rid:
            return None
        try:
            part = self.docx_obj.part.related_parts[rid]
        except KeyError:
            _log.debug("chart relationship %r not found in document part", rid)
            return None
        try:
            return etree.fromstring(part.blob, parser=_SAFE_XML_PARSER)
        except etree.XMLSyntaxError:
            _log.debug("could not parse chart part for relationship %r", rid)
            return None

    def _classify_chart(self, chart_root: Any) -> PictureClassificationLabel:
        """Classify a chart from the first plot element under ``c:plotArea``.

        The plot element name (``barChart``, ``lineChart``, ``pieChart``, ...) is
        mapped to a docling classification label; unknown or combination charts
        fall back to OTHER_CHART.
        """
        plot_area = chart_root.find(".//c:plotArea", namespaces=self._BLIP_NAMESPACES)
        if plot_area is not None:
            for child in plot_area:
                label = _CHART_TAGNAME_TO_CLASSIFICATION.get(
                    etree.QName(child).localname
                )
                if label is not None:
                    return label
        return PictureClassificationLabel.OTHER_CHART

    @staticmethod
    def _chart_cell_text(value: str | None) -> str:
        """Format a cached chart value as cell text.

        Numeric cache values (``c:numCache``) round-trip through float so they
        read like the source data (``"4.4000000000000004"`` -> ``"4.4"``,
        ``"2"`` -> ``"2"``). Non-numeric labels are returned unchanged; None
        becomes an empty string.
        """
        if value is None:
            return ""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return value
        if number.is_integer():
            return str(int(number))
        return str(number)

    def _read_chart_cache(self, node: Any | None) -> list[str]:
        """Read the cached ``c:pt`` values under a chart data source.

        Handles both cached references (``c:numCache``/``c:strCache``) and inline
        literals (``c:numLit``/``c:strLit``). Points are indexed by ``@idx`` and
        gaps are filled with empty strings so every series aligns to the same rows.
        """
        if node is None:
            return []
        ns = self._BLIP_NAMESPACES
        cache = None
        for tag in ("numCache", "strCache", "numLit", "strLit"):
            cache = node.find(f".//c:{tag}", namespaces=ns)
            if cache is not None:
                break
        if cache is None:
            return []

        points: dict[int, str] = {}
        for pt in cache.findall("c:pt", namespaces=ns):
            try:
                idx = int(pt.get("idx", "0"))
            except ValueError:
                continue
            value = pt.find("c:v", namespaces=ns)
            points[idx] = self._chart_cell_text(
                value.text if value is not None else None
            )
        if not points:
            return []

        count = 0
        count_el = cache.find("c:ptCount", namespaces=ns)
        if count_el is not None and count_el.get("val"):
            try:
                count = int(count_el.get("val"))
            except ValueError:
                count = 0
        length = max([count] + [idx + 1 for idx in points])
        return [points.get(idx, "") for idx in range(length)]

    def _chart_series_name(self, series: Any) -> str:
        """Return a chart series' name from its ``c:tx`` (cached ref or literal)."""
        tx = series.find("c:tx", namespaces=self._BLIP_NAMESPACES)
        if tx is None:
            return ""
        cached = self._read_chart_cache(tx)
        if cached:
            return cached[0]
        literal = tx.find("c:v", namespaces=self._BLIP_NAMESPACES)
        return self._chart_cell_text(literal.text) if literal is not None else ""

    def _chart_title_text(self, chart_root: Any) -> str | None:
        """Extract the chart's title text, or None when it has no title.

        A chart title is DrawingML rich text (``a:t`` runs) under ``c:chart/
        c:title``; some charts instead reference a cell, cached in a ``c:strRef``.
        """
        ns = self._BLIP_NAMESPACES
        chart = chart_root.find("c:chart", namespaces=ns)
        if chart is None:
            return None
        title = chart.find("c:title", namespaces=ns)
        if title is None:
            return None
        runs = [t.text for t in title.findall(".//a:t", namespaces=ns) if t.text]
        text = "".join(runs).strip()
        if not text:
            cached = self._read_chart_cache(title)
            text = cached[0].strip() if cached else ""
        return text or None

    def _chart_to_table_data(self, chart_root: Any) -> TableData | None:
        """Reconstruct a chart's underlying data grid as a TableData.

        Layout produced (categories down the first column, one column per series):

            | <blank> | <series 0 name> | <series 1 name> | ...
            | cat_0   | val_0,0         | val_1,0         | ...
            | cat_1   | val_0,1         | val_1,1         | ...

        The plotted numbers and labels come from each series' inline cache
        (``c:numCache``/``c:strCache``), so no workbook reference resolution is
        needed. Scatter charts use ``c:xVal``/``c:yVal`` in place of ``c:cat``/
        ``c:val``.

        Args:
            chart_root: The parsed root of a ``chartN.xml`` part.

        Returns:
            A TableData, or None if the chart exposes no usable series.
        """
        ns = self._BLIP_NAMESPACES
        series_list = chart_root.findall(".//c:ser", namespaces=ns)
        if not series_list:
            return None

        categories: list[str] = []
        for series in series_list:
            cat = series.find("c:cat", namespaces=ns)
            if cat is None:
                cat = series.find("c:xVal", namespaces=ns)
            resolved = self._read_chart_cache(cat)
            if resolved:
                categories = resolved
                break

        columns: list[tuple[str, list[str]]] = []
        for series in series_list:
            val = series.find("c:val", namespaces=ns)
            if val is None:
                val = series.find("c:yVal", namespaces=ns)
            values = self._read_chart_cache(val)
            columns.append((self._chart_series_name(series), values))

        num_data_rows = max([len(categories)] + [len(values) for _, values in columns])
        if num_data_rows == 0:
            return None

        num_rows = num_data_rows + 1
        num_cols = 1 + len(columns)
        cells: list[TableCell] = []

        header_labels = [""] + [name for name, _ in columns]
        for col_idx, label in enumerate(header_labels):
            cells.append(
                TableCell(
                    text=label,
                    row_span=1,
                    col_span=1,
                    start_row_offset_idx=0,
                    end_row_offset_idx=1,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + 1,
                    column_header=True,
                    row_header=False,
                )
            )
        for data_row in range(num_data_rows):
            row_idx = data_row + 1
            category = categories[data_row] if data_row < len(categories) else ""
            row_texts = [category] + [
                (values[data_row] if data_row < len(values) else "")
                for _, values in columns
            ]
            for col_idx, text in enumerate(row_texts):
                cells.append(
                    TableCell(
                        text=text,
                        row_span=1,
                        col_span=1,
                        start_row_offset_idx=row_idx,
                        end_row_offset_idx=row_idx + 1,
                        start_col_offset_idx=col_idx,
                        end_col_offset_idx=col_idx + 1,
                        column_header=False,
                        row_header=(col_idx == 0),
                    )
                )

        return TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells)

    def _render_chart_image(self, drawing_el: Any) -> Image.Image | None:
        """Render a charted drawing to an image via LibreOffice.

        Reuses the shared DOCX->PDF->PNG path, isolating just the chart drawing.
        Returns None (and warns once) when LibreOffice is unavailable or the
        render fails, so the chart still keeps its classification and data.
        """
        image = self._convert_elements_via_docx([drawing_el], element_tag=None)
        if image is None and self.display_drawingml_warning:
            _log.warning(_CHART_RENDER_HINT)
            self.display_drawingml_warning = False
        return image

    def _handle_chart(self, doc: DoclingDocument, chart_el: Any) -> RefItem | None:
        """Add a native chart drawing as a classified PictureItem with its data.

        The chart becomes a PictureItem whose meta carries (a) the chart-type
        classification and (b) the chart's plotted numbers reconstructed as a
        TableData — both derived from the chart part without LibreOffice. The
        chart title, if any, becomes the picture caption. When
        ``render_chart_images`` is enabled and LibreOffice is available, a
        rendered image is attached; on any failure the picture keeps its
        classification and data without an image.

        Args:
            doc: The DoclingDocument being constructed.
            chart_el: The ``w:drawing`` element embedding the chart.

        Returns:
            A reference to the added picture, or None if it could not be added.
        """
        level = self._get_level()
        parent = self.parents[level - 1]

        chart_root = self._resolve_chart_root(chart_el)
        classification: PictureClassificationLabel | None = None
        table_data: TableData | None = None
        caption_text: str | None = None
        if chart_root is not None:
            classification = self._classify_chart(chart_root)
            table_data = self._chart_to_table_data(chart_root)
            caption_text = self._chart_title_text(chart_root)

        render_charts = (
            isinstance(self.options, MsWordBackendOptions)
            and self.options.render_chart_images
        )
        image_ref = None
        if render_charts:
            chart_image = self._render_chart_image(chart_el)
            if chart_image is not None:
                image_ref = ImageRef.from_pil(image=chart_image, dpi=72)

        caption_item = (
            doc.add_text(
                label=DocItemLabel.CAPTION,
                text=caption_text,
                content_layer=self.content_layer,
            )
            if caption_text
            else None
        )
        picture = doc.add_picture(
            parent=parent,
            image=image_ref,
            caption=caption_item,
            content_layer=self.content_layer,
        )
        if classification is not None:
            picture.meta = PictureMeta(
                classification=PictureClassificationMetaField(
                    predictions=[
                        PictureClassificationPrediction(class_name=classification)
                    ]
                ),
                tabular_chart=(
                    TabularChartMetaField(chart_data=table_data)
                    if table_data is not None
                    else None
                ),
            )
        return picture.get_ref()

    def _add_header_footer(self, docx_obj: DocxDocument, doc: DoclingDocument) -> None:
        """Add section headers and footers.

        Headers and footers are added in the furniture content and only the text paragraphs
        are parsed. The paragraphs are attached to a single group item for the header or the
        footer. If the document has a section with new header and footer, they will be parsed
        in new group items.

        Args:
            docx_obj: A docx Document object to be parsed.
            doc: A DoclingDocument object to add the header and footer from docx_obj.
        """
        current_layer = self.content_layer
        base_parent = self.parents[0]
        self.content_layer = ContentLayer.FURNITURE

        txbx_xpath = etree.XPath(
            ".//w:txbxContent|.//v:textbox//w:p|.//wps:txbx//w:p|.//a:p//a:t",
            namespaces=self._BLIP_NAMESPACES,
        )
        for sec_idx, section in enumerate(docx_obj.sections):
            if sec_idx > 0 and not section.different_first_page_header_footer:
                continue

            hdr = (
                section.first_page_header
                if section.different_first_page_header_footer
                else section.header
            )
            par = [txt for txt in (par.text.strip() for par in hdr.paragraphs) if txt]
            tables = hdr.tables
            has_blip = self._has_blip(hdr._element)
            has_txbx = len(txbx_xpath(hdr._element)) > 0

            if par or tables or has_blip or has_txbx:
                self.parents[0] = doc.add_group(
                    label=GroupLabel.SECTION,
                    name="page header",
                    content_layer=self.content_layer,
                )
                # Each header/footer part is its own code-block scope.
                self._force_new_code_block = True
                self._pending_code_blank_lines = 0
                self.current_part = hdr.part
                self._walk_linear(hdr._element, doc)
                self.current_part = self.docx_obj.part

            ftr = (
                section.first_page_footer
                if section.different_first_page_header_footer
                else section.footer
            )
            par = [txt for txt in (par.text.strip() for par in ftr.paragraphs) if txt]
            tables = ftr.tables
            has_blip = self._has_blip(ftr._element)
            has_txbx = len(txbx_xpath(ftr._element)) > 0

            if par or tables or has_blip or has_txbx:
                self.parents[0] = doc.add_group(
                    label=GroupLabel.SECTION,
                    name="page footer",
                    content_layer=self.content_layer,
                )
                self._force_new_code_block = True
                self._pending_code_blank_lines = 0
                self.current_part = ftr.part
                self._walk_linear(ftr._element, doc)
                self.current_part = self.docx_obj.part

        self._force_new_code_block = True
        self._pending_code_blank_lines = 0
        self.content_layer = current_layer
        self.parents[0] = base_parent

    def _add_comments(self, docx_obj: DocxDocument, doc: DoclingDocument) -> None:
        """Add document comments (reviewer annotations) and link to annotated items.

        Comments are added in the NOTES content layer using the add_comment API.
        Comments are linked to their annotated text items via the comments field
        (FineRef references).

        The comment text format is:
            [author: Name (initials), time: ISO-timestamp]: comment text

        Examples:
            [author: Jane Editor (JE), time: 2026-01-04T05:48:07.000+00:00]: Review this.
            [author: John Doe]: Simple comment without timestamp.

        Args:
            docx_obj: A docx Document object to be parsed.
            doc: A DoclingDocument object to add the comments from docx_obj.
        """
        # Check if document has any comments
        if not hasattr(docx_obj, "comments") or len(docx_obj.comments) == 0:
            return

        # Process each comment and link to target items
        for comment in docx_obj.comments:
            # Build comment text with metadata prefix
            metadata_parts = []
            if comment.author:
                author_str = f"author: {comment.author}"
                if comment.initials:
                    author_str += f" ({comment.initials})"
                metadata_parts.append(author_str)
            if comment.timestamp:
                metadata_parts.append(
                    f"time: {comment.timestamp.isoformat(timespec='milliseconds')}"
                )

            metadata_prefix = ", ".join(metadata_parts)
            comment_text = comment.text.strip() if comment.text else ""

            if metadata_prefix and comment_text:
                full_text = f"[{metadata_prefix}]: {comment_text}"
            elif metadata_prefix:
                full_text = f"[{metadata_prefix}]"
            elif comment_text:
                full_text = comment_text
            else:
                # Skip empty comments
                continue

            # Collect target items for this comment
            targets: list[Any] = []
            comment_id = str(comment.comment_id)

            # Find paragraphs that have this comment
            for para_id, comment_ids in self.paragraph_comment_map.items():
                if comment_id in comment_ids:
                    # Get the text items created from this paragraph
                    if para_id in self.paragraph_to_items:
                        for item_ref in self.paragraph_to_items[para_id]:
                            try:
                                item = item_ref.resolve(doc)
                                if item not in targets:
                                    targets.append(item)
                            except Exception as e:
                                _log.debug(f"Error resolving item ref: {e}")

            # Create a group for this comment in NOTES and add the comment there
            comment_group = doc.add_group(
                label=GroupLabel.COMMENT_SECTION,
                name=f"comment-{comment_id}",
                content_layer=ContentLayer.NOTES,
            )
            # Add the comment text without targets first
            doc.add_comment(
                text=full_text,
                targets=None,
                parent=comment_group,
            )

            # Manually link targets to the comment GROUP (not the text item)
            # This allows multiple comment replies to be grouped together
            if targets:
                group_ref = FineRef(cref=comment_group.self_ref)
                for target in targets:
                    # Only DocItem has a 'comments' field; GroupItem does not,
                    # so skip non-DocItem targets (fixes #2955).
                    if isinstance(target, DocItem):
                        target.comments.append(group_ref)

            _log.debug(
                f"Added comment {comment_id} in group with {len(targets)} linked item(s)"
            )

    def _extract_comment_ranges(self) -> None:
        """Extract comment range markers from the document.

        Parses the DOCX XML to find commentRangeStart and commentRangeEnd markers
        and builds a map of paragraph elements to their associated comment IDs.
        """
        if not self.docx_obj or not hasattr(self.docx_obj, "element"):
            return

        # Parse the document body for comment range markers
        namespaces = {
            "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        }

        try:
            # Find all paragraphs with comment ranges
            body = self.docx_obj.element.body
            for paragraph in body.findall(".//w:p", namespaces):
                para_id = id(paragraph)

                # Find comment range start markers in this paragraph
                comment_starts = paragraph.findall(".//w:commentRangeStart", namespaces)
                comment_ends = paragraph.findall(".//w:commentRangeEnd", namespaces)

                # Collect comment IDs from both start and end markers
                comment_ids = set()

                for start_marker in comment_starts:
                    comment_id = start_marker.get(
                        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}id"
                    )
                    if comment_id:
                        comment_ids.add(comment_id)

                for end_marker in comment_ends:
                    comment_id = end_marker.get(
                        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}id"
                    )
                    if comment_id:
                        comment_ids.add(comment_id)

                # Store the mapping if comments were found
                if comment_ids:
                    self.paragraph_comment_map[para_id] = list(comment_ids)
                    _log.debug(
                        f"Found {len(comment_ids)} comment(s) for paragraph {para_id}"
                    )

        except Exception as e:
            _log.debug(f"Error extracting comment ranges: {e}")

    def _get_comment_ids_for_element(self, element: BaseOxmlElement) -> set[str]:
        """Return the set of comment IDs attached to a paragraph element."""
        namespaces = {
            "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        }
        comment_ids: set[str] = set()

        for marker in element.findall(".//w:commentRangeStart", namespaces):
            comment_id = marker.get(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}id"
            )
            if comment_id:
                comment_ids.add(comment_id)

        for marker in element.findall(".//w:commentRangeEnd", namespaces):
            comment_id = marker.get(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}id"
            )
            if comment_id:
                comment_ids.add(comment_id)

        # Some documents only contain commentReference nodes without range markers
        for marker in element.findall(".//w:commentReference", namespaces):
            comment_id = marker.get(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}id"
            )
            if comment_id:
                comment_ids.add(comment_id)

        return comment_ids
