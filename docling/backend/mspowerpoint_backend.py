# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import math
import re
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from tempfile import mkdtemp
from typing import (
    Any,
    Callable,
    Final,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Union,
)
from urllib.parse import urlparse

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupItem,
    GroupLabel,
    ImageRef,
    ListGroup,
    ListItem,
    PictureClassificationLabel,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureMeta,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
    TabularChartMetaField,
    TextItem,
)
from docling_core.types.doc.document import ContentLayer
from lxml import etree
from PIL import Image, UnidentifiedImageError
from pydantic import AnyUrl, ValidationError
from typing_extensions import override

from docling.backend.abstract_backend import (
    DeclarativeDocumentBackend,
    PaginatedDocumentBackend,
)
from docling.backend.docx.drawingml.utils import (
    convert_to_modern_format,
    crop_whitespace,
    get_docx_to_pdf_converter,
)
from docling.backend.docx.latex.omml import oMath2Latex
from docling.backend.utils.charts import build_chart_table, datetime_to_iso
from docling.backend.utils.image import normalize_image_for_png
from docling.datamodel.backend_options import MsPowerpointBackendOptions
from docling.datamodel.base_models import FormatToMimeType, InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError

_log = logging.getLogger(__name__)

_PPTX_AVAILABLE: bool = False
_PPTX_IMPORT_ERROR: ImportError | None = None
try:  # pragma: no cover - import-time guard
    from pptx import Presentation, presentation
    from pptx.chart.series import BarSeries, XySeries
    from pptx.enum.action import PP_ACTION
    from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER
    from pptx.exc import InvalidXmlError
    from pptx.oxml.ns import qn
    from pptx.oxml.text import CT_TextLineBreak
    from pptx.shapes.autoshape import Shape
    from pptx.shapes.base import BaseShape
    from pptx.shapes.picture import Picture

    _PPTX_AVAILABLE = True
except ImportError as e:  # pragma: no cover - import-time guard
    _PPTX_IMPORT_ERROR = e

# pypdfium2 ships with the PDF extras, not with format-pptx, and is only reached
# after `get_docx_to_pdf_converter` returns a converter. That shared factory
# returns None when pypdfium2 is missing, so the rendering paths already degrade
# to "no image"; this guard only keeps the module itself importable.
# See https://github.com/docling-project/docling/issues/3613.
try:  # pragma: no cover - import-time guard
    import pypdfium2
except ImportError:  # pragma: no cover - import-time guard
    pass

_INSTALL_HINT = (
    "The 'python-pptx' package is required to process PowerPoint files. "
    "Install it with `pip install 'docling-slim[format-pptx]'`."
)

_CHART_RENDER_HINT = (
    "LibreOffice is required to render PowerPoint charts as images "
    "(render_chart_images=True). Install LibreOffice and make sure `soffice` is "
    "on PATH. Charts still keep their classification and reconstructed tabular "
    "data without it."
)

_IMAGE_RENDER_HINT = (
    "LibreOffice is required to rasterize EMF/WMF pictures embedded in slides. "
    "Install LibreOffice and make sure `soffice` is on PATH. Such pictures are "
    "still recorded with their position on the slide without it, but carry no "
    "image data."
)

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
"""Maps the tag-local name of a chart plot element, such as ``"bar3DChart"``, to
a docling picture-classification label. The Word and Excel backends use the same
mapping. Plot types not listed fall back to ``OTHER_CHART``."""

_XY_PLOT_TAGNAMES: Final = frozenset({"scatterChart", "bubbleChart"})
"""Plot types whose series keep their values in ``c:yVal`` instead of ``c:val``."""

_EXCEL_EPOCH: Final = datetime(1899, 12, 30)
_EXCEL_1904_EPOCH: Final = datetime(1904, 1, 1)
"""Day zero of the date serials in a chart's data cache, for the 1900 and the
1904 date systems. ``c:date1904`` on the chart selects the second one."""

_NUMBER_FORMAT_LITERALS: Final = re.compile(r'"[^"]*"|\\.|\[[^\]]*\]|[_*].')
"""Parts of an Excel number format that are not format codes: quoted text,
escaped characters, bracketed colours, locales and conditions, and padding."""

# Windows metafile signatures. A placeable WMF opens with the Aldus magic and an
# EMF carries " EMF" in the dSignature field of its EMR_HEADER, 40 bytes in.
# These are the two Pillow itself identifies, so a bare WMF never reaches here.
_WMF_PLACEABLE_MAGIC: Final = b"\xd7\xcd\xc6\x9a"
_EMF_SIGNATURE: Final = b" EMF"
_EMF_SIGNATURE_OFFSET: Final = 40


# The markup-compatibility namespace. python-pptx registers this URI under the
# ``ve`` prefix rather than the ``mc`` prefix files actually use, so the tag is
# spelled out here instead of going through ``qn``.
_MC_ALTERNATE_CONTENT: Final = (
    "{http://schemas.openxmlformats.org/markup-compatibility/2006}AlternateContent"
)
_MATH_NAMESPACE: Final = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_DRAWING_2010_NAMESPACE: Final = "http://schemas.microsoft.com/office/drawing/2010/main"
_MC_NAMESPACE: Final = "http://schemas.openxmlformats.org/markup-compatibility/2006"


def _iter_paragraph_elements(element: etree._Element) -> Iterator[etree._Element]:
    """Read text and OMML in order, selecting one markup-compatibility branch."""
    for child in element:
        if child.tag == _MC_ALTERNATE_CONTENT:
            branch = child.find(f"{{{_MC_NAMESPACE}}}Fallback")
            for choice in child.findall(f"{{{_MC_NAMESPACE}}}Choice"):
                required = choice.get("Requires", "").split()
                if required and all(
                    choice.nsmap.get(prefix)
                    in {_MATH_NAMESPACE, _DRAWING_2010_NAMESPACE}
                    for prefix in required
                ):
                    branch = choice
                    break
            if branch is not None:
                yield from _iter_paragraph_elements(branch)
        elif child.tag in {
            f"{{{_DRAWING_2010_NAMESPACE}}}m",
            f"{{{_MATH_NAMESPACE}}}oMathPara",
        }:
            yield from _iter_paragraph_elements(child)
        elif child.tag in {
            qn("a:r"),
            qn("a:br"),
            qn("a:fld"),
            f"{{{_MATH_NAMESPACE}}}oMath",
        }:
            yield child


def _paragraph_fragments(element: etree._Element) -> list[tuple[DocItemLabel, str]]:
    """Coalesce adjacent text runs while keeping equations as separate fragments."""
    fragments: list[tuple[DocItemLabel, str]] = []
    text = ""
    for child in _iter_paragraph_elements(element):
        if child.tag == f"{{{_MATH_NAMESPACE}}}oMath":
            if text:
                fragments.append((DocItemLabel.TEXT, text))
                text = ""
            equation = str(oMath2Latex(child)).strip()
            if equation:
                fragments.append((DocItemLabel.FORMULA, equation))
        elif isinstance(child, CT_TextLineBreak):
            text += " "
        else:
            run_text = child.find(qn("a:t"))
            if run_text is not None:
                text += run_text.text or ""
    if text:
        fragments.append((DocItemLabel.TEXT, text))
    return fragments


_SAFE_XML_PARSER: Final = etree.XMLParser(
    resolve_entities=False,
    load_dtd=False,
    no_network=True,
    dtd_validation=False,
)
"""Safe XML parser to prevent XXE, DTD-over-network and entity-expansion attacks."""


@dataclass
class _OpenList:
    """A list group that is still accepting items while a text frame is walked.

    Attributes:
        group: The list group that items at ``level`` are added to.
        level: The paragraph level (``a:pPr/@lvl``) of the group's items.
        counter: The number of enumerated items added to the group so far.
        last_item: The most recent item in the group, which parents any list
            nested below it.
    """

    group: ListGroup
    level: int
    counter: int = 0
    last_item: Optional[ListItem] = None


def _is_metafile(image_bytes: bytes) -> bool:
    """Return True when the bytes are a Windows metafile (WMF or EMF).

    Pillow renders metafiles only on Windows, where it delegates to the GDI
    ``PlayEnhMetaFile`` API; the wheels for every other platform carry a stub
    that raises on load. Telling a metafile apart from a genuinely broken
    image decides whether a picture is worth rasterizing externally.

    Args:
        image_bytes: Raw bytes of an embedded picture.

    Returns:
        True when the bytes carry a placeable WMF or an EMF signature.
    """
    return image_bytes[:4] == _WMF_PLACEABLE_MAGIC or (
        image_bytes[_EMF_SIGNATURE_OFFSET : _EMF_SIGNATURE_OFFSET + 4] == _EMF_SIGNATURE
    )


def _is_date_format(format_code: str) -> bool:
    """Return True when an Excel number format shows a date.

    A ``d`` or ``y`` code marks a date. An ``m`` code marks a month only when
    there is no ``h`` or ``s`` code, because ``m`` also means minutes. Time-only
    formats and ``General`` therefore return False.

    Args:
        format_code: An Excel number format, such as ``mmm d, yy``.

    Returns:
        True when the format shows a date.
    """
    codes = _NUMBER_FORMAT_LITERALS.sub("", format_code).lower()
    if "d" in codes or "y" in codes:
        return True
    return "m" in codes and "h" not in codes and "s" not in codes


def _serial_to_iso(text: str, date1904: bool) -> str:
    """Write an Excel date serial as an ISO 8601 date, or keep the text.

    A chart caches its date categories as serial numbers, such as ``40909``
    for 1 January 2012. A whole serial becomes a date (``2012-01-01``). A
    serial with a time of day becomes a date and time to the second.

    Args:
        text: A category label from the chart's data cache.
        date1904: True when the chart uses the 1904 date system.

    Returns:
        The ISO 8601 text, or ``text`` unchanged when it is not a serial.
    """
    try:
        serial = float(text)
    except ValueError:
        return text
    if not math.isfinite(serial) or serial < 0:
        return text
    if date1904:
        epoch = _EXCEL_1904_EPOCH
    else:
        # Excel counts the nonexistent 1900-02-29 as serial 60.
        epoch = _EXCEL_EPOCH + timedelta(days=1 if serial < 60 else 0)
    try:
        moment = epoch + timedelta(seconds=round(serial * 86400))
    except OverflowError:
        return text
    return datetime_to_iso(moment)


class MsPowerpointDocumentBackend(DeclarativeDocumentBackend, PaginatedDocumentBackend):
    """Backend for parsing PowerPoint presentations (PPTX and PPT files).

    Converts presentations into structured DoclingDocument format, extracting
    text, tables, images, lists, and comments from slides.

    Legacy ``.ppt`` files (binary PowerPoint 97-2003 format) are first converted
    to ``.pptx`` via LibreOffice before parsing.

    Note:
        Comments are extracted and added to the NOTES content layer. Unlike Word documents,
            PPTX comments only contain position coordinates (x, y) and do not directly reference
            specific shapes or text ranges. Therefore, comments are associated with their parent
            slide group rather than specific document elements.
    """

    # XML namespaces for element lookup
    NAMESPACES: Final[dict[str, str]] = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
        "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    }
    _SHAPE_ROW_TOLERANCE_EMU: Final[int] = 45720
    """Row-grouping tolerance used by ``_iter_shapes_by_position``, in EMUs.

    0.05 inch expressed in EMUs (1 inch = 914400 EMUs). This threshold is
    intentionally small — roughly half a typical title line-height — so that
    icon-label pairs and other tightly aligned objects on the same visual row
    are grouped together, while shapes in separate rows (which are normally at
    least one line-height apart) are not.
    """

    # XML relationship types
    COMMENT_REL = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments"
    )
    COMMENT_AUTHORS_REL = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
        "commentAuthors"
    )

    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
        options: Optional[MsPowerpointBackendOptions] = None,
    ) -> None:
        if not _PPTX_AVAILABLE:
            raise ImportError(_INSTALL_HINT) from _PPTX_IMPORT_ERROR
        if in_doc.format == InputFormat.PPT:
            path_or_stream = convert_to_modern_format(path_or_stream, "ppt", "pptx")
        if options is None:
            options = MsPowerpointBackendOptions()
        super().__init__(in_doc, path_or_stream, options)
        self.path_or_stream: Union[BytesIO, Path] = path_or_stream
        self.page_range = in_doc.limits.page_range

        self.pptx_to_pdf_converter: Optional[Callable] = None
        self.pptx_to_pdf_converter_init: bool = False
        self._render_charts: bool = False
        self._metafile_hint_emitted: bool = False

        self.pptx_obj: Optional[presentation.Presentation] = None
        self.valid: bool = False
        try:
            if isinstance(self.path_or_stream, BytesIO):
                self.pptx_obj = Presentation(self.path_or_stream)
            elif isinstance(self.path_or_stream, Path):
                self.pptx_obj = Presentation(str(self.path_or_stream))

            self.valid = True
        except Exception as e:
            raise DocumentLoadError(
                f"MsPowerpointDocumentBackend could not load document with hash {self.document_hash}"
            ) from e

        return

    def page_count(self) -> int:
        if self.is_valid():
            assert self.pptx_obj is not None
            return len(self.pptx_obj.slides)
        else:
            return 0

    @override
    def is_valid(self) -> bool:
        return self.valid

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return True

    @override
    def unload(self):
        if isinstance(self.path_or_stream, BytesIO):
            self.path_or_stream.close()

        self.path_or_stream = None

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.PPTX, InputFormat.PPT}

    @override
    def convert(self) -> DoclingDocument:
        """Parse the PPTX into a structured document model.

        Returns:
            The parsed document.
        """
        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype=FormatToMimeType[self.input_format][0],
            binary_hash=self.document_hash,
        )

        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)
        if self.pptx_obj:
            # Build author map once for all comments
            author_map = self._build_comment_author_map(self.pptx_obj)

            start_page, end_page = self.page_range
            doc = self._walk_linear(
                self.pptx_obj, doc, author_map, start_page=start_page, end_page=end_page
            )

        return doc

    def _generate_prov(
        self, shape, slide_ind, text="", slide_size=Size(width=1, height=1)
    ):
        if None not in (shape.left, shape.top, shape.width, shape.height):
            left = shape.left
            top = shape.top
            width = shape.width
            height = shape.height
        else:
            left = 0
            top = 0
            width = slide_size.width
            height = slide_size.height
        shape_bbox = [left, top, left + width, top + height]
        if None not in (shape.left, shape.top, shape.width, shape.height):
            shape_bbox = self._to_slide_coordinates(shape, shape_bbox)
        # python-pptx reports left and top as EMU from the slide's top-left, with
        # y growing downward, so the tuple above is in TOPLEFT order. Tagging it
        # BOTTOMLEFT does not convert it: BoundingBox.from_tuple unpacks
        # l, b, r, t for that origin, so the top edge lands in b and the bottom
        # edge in t, and a consumer calling to_top_left_origin then computes
        # page_height - t and mirrors the box. html_backend and msexcel_backend
        # tag their own top-left coordinates TOPLEFT for the same reason.
        shape_bbox = BoundingBox.from_tuple(shape_bbox, origin=CoordOrigin.TOPLEFT)
        prov = ProvenanceItem(
            page_no=slide_ind + 1, charspan=[0, len(text)], bbox=shape_bbox
        )

        return prov

    @staticmethod
    def _to_slide_coordinates(shape, bbox: list) -> list:
        """Map a box in a grouped shape's coordinates to slide coordinates.

        python-pptx reports the position of a shape inside a group in the
        group's child coordinate space (``a:chOff``/``a:chExt``), which only
        matches the slide while the group has not been moved, resized or
        flipped since its shapes were grouped. Each enclosing group maps that
        space onto its own box (``a:off``/``a:ext``), so the transforms are
        applied from the innermost group outwards. Group rotation is not
        applied.

        Args:
            shape: The shape the box belongs to.
            bbox: The box as ``[left, top, right, bottom]`` in the shape's own
                coordinate space.

        Returns:
            The box as ``[left, top, right, bottom]`` in slide coordinates.
        """
        left, top, right, bottom = bbox
        parent = shape._element.getparent()
        while parent is not None and parent.tag == qn("p:grpSp"):
            xfrm = parent.find(f"{qn('p:grpSpPr')}/{qn('a:xfrm')}")
            parts = (
                None
                if xfrm is None
                else [
                    xfrm.find(qn(f"a:{name}"))
                    for name in ("off", "ext", "chOff", "chExt")
                ]
            )
            if parts is not None and None not in parts:
                off, ext, ch_off, ch_ext = parts
                try:
                    off_x, off_y = int(off.get("x")), int(off.get("y"))
                    ext_x, ext_y = int(ext.get("cx")), int(ext.get("cy"))
                    ch_x, ch_y = int(ch_off.get("x")), int(ch_off.get("y"))
                    ch_cx, ch_cy = int(ch_ext.get("cx")), int(ch_ext.get("cy"))
                except (TypeError, ValueError):
                    pass
                else:
                    scale_x = ext_x / ch_cx if ch_cx else 1.0
                    scale_y = ext_y / ch_cy if ch_cy else 1.0
                    left = off_x + (left - ch_x) * scale_x
                    right = off_x + (right - ch_x) * scale_x
                    top = off_y + (top - ch_y) * scale_y
                    bottom = off_y + (bottom - ch_y) * scale_y
                    if xfrm.get("flipH") in ("1", "true"):
                        left, right = (
                            2 * off_x + ext_x - right,
                            2 * off_x + ext_x - left,
                        )
                    if xfrm.get("flipV") in ("1", "true"):
                        top, bottom = (
                            2 * off_y + ext_y - bottom,
                            2 * off_y + ext_y - top,
                        )
            parent = parent.getparent()
        return [left, top, right, bottom]

    def _get_paragraph_level(self, paragraph) -> int:
        """Return the indentation level of a paragraph XML element.

        Paragraphs can have different indentation levels (0-8). The level is
        stored in the `lvl` attribute of the `a:pPr` element (paragraph properties).

        Args:
            paragraph: Paragraph XML element whose level should be extracted.

        Returns:
            Paragraph level in the range (0, 8). Returns 0 when no `a:pPr` element is
                found, no `lvl` attribute exists, or the `lvl` attribute value is
                invalid.
        """
        pPr = paragraph.find("a:pPr", namespaces=self.NAMESPACES)
        if pPr is not None and "lvl" in pPr.attrib:
            try:
                return int(pPr.get("lvl"))
            except ValueError:
                pass
        return 0

    def _get_auto_number_start(self, paragraph) -> int:
        """Return the number an auto-numbered paragraph's list starts from.

        PowerPoint records the "Start at" value of a numbered list in the
        `startAt` attribute of the paragraph's `a:buAutoNum` element.

        Args:
            paragraph: Paragraph XML element whose start value should be extracted.

        Returns:
            The `startAt` value, or 1 when the paragraph carries no `a:buAutoNum`
                element, no `startAt` attribute, or an invalid value.
        """
        bu_auto = paragraph.find("a:pPr/a:buAutoNum", namespaces=self.NAMESPACES)
        if bu_auto is not None and "startAt" in bu_auto.attrib:
            try:
                return int(bu_auto.get("startAt"))
            except ValueError:
                pass
        return 1

    def _parse_bullet_from_paragraph_properties(
        self, pPr
    ) -> tuple[Optional[bool], Optional[str], Optional[str]]:
        """Parse bullet or numbering information from a paragraph properties node.

        This inspects the `a:pPr` or `a:lvlXpPr` element and extracts
        information about the bullet character, automatic numbering, picture
        bullets, or explicit `buNone` markers.

        Args:
            pPr: Paragraph properties XML element (`a:pPr` or `a:lvlXpPr`).

        Returns:
            A 3-tuple (`is_list`, `kind`, `detail`) where: `is_list` is True/False/None
                indicating whether this is a list item; `kind` is one of `buChar`,
                `buAutoNum`, `buBlip`, `buNone` or None, describing the marker type;
                `detail` is the bullet character, numbering type string, or None if not
                applicable.
        """
        if pPr is None:
            return (None, None, None)

        # Explicitly no bullet
        if pPr.find("a:buNone", namespaces=self.NAMESPACES) is not None:
            return (False, "buNone", None)

        # Bullet character
        buChar = pPr.find("a:buChar", namespaces=self.NAMESPACES)
        if buChar is not None:
            return (True, "buChar", buChar.get("char"))

        # Auto numbering
        buAuto = pPr.find("a:buAutoNum", namespaces=self.NAMESPACES)
        if buAuto is not None:
            return (True, "buAutoNum", buAuto.get("type"))

        # Picture bullet
        buBlip = pPr.find("a:buBlip", namespaces=self.NAMESPACES)
        if buBlip is not None:
            return (True, "buBlip", "image")

        return (None, None, None)

    def _find_level_properties_in_list_style(self, lstStyle, lvl: int):
        """Find the level-specific paragraph properties node from a list style.

        This looks for an `a:lvl{lvl+1}pPr` node inside an `a:lstStyle` element, where
        `a:lvl1pPr` corresponds to level 0, `a:lvl2pPr` to level 1, and so on.

        Args:
            lstStyle: List style XML element `a:lstStyle`.
            lvl: Paragraph level in the range (0, 8).

        Returns:
            Matching `a:lvl{lvl+1}pPr` XML element, or None if no matching element is
                found.
        """
        if lstStyle is None:
            return None
        tag = f"a:lvl{lvl + 1}pPr"
        return lstStyle.find(tag, namespaces=self.NAMESPACES)

    def _parse_bullet_from_text_body_list_style(
        self, txBody, lvl: int
    ) -> tuple[Optional[bool], Optional[str], Optional[str]]:
        """Parse bullet or numbering information from a text body's list style.

        This searches for `a:lstStyle/a:lvl{lvl+1}pPr` under a `txBody` and uses the
        level-specific paragraph properties to deduce bullet or numbering information.

        Args:
            txBody: Text body XML element `p:txBody`.
            lvl: Paragraph level in the range (0, 8).

        Returns:
            A 3-tuple (`is_list`, `kind`, `detail`) where: `is_list` is True/False/None
                indicating whether this is a list item; `kind` is one of `buChar`,
                `buAutoNum`, `buBlip`, `buNone` or None, describing the marker type;
                `detail` is the bullet character, numbering type string, or None if not
                applicable.
        """
        if txBody is None:
            return (None, None, None)
        lstStyle = txBody.find("a:lstStyle", namespaces=self.NAMESPACES)
        lvl_pPr = self._find_level_properties_in_list_style(lstStyle, lvl)
        is_list, kind, detail = self._parse_bullet_from_paragraph_properties(lvl_pPr)
        return (is_list, kind, detail)

    def _get_master_text_style_node(
        self, slide_master, placeholder_type
    ) -> Optional[etree._Element]:
        """Get the appropriate master text style node for a placeholder.

        Most content placeholders (BODY/OBJECT) use `p:bodyStyle`, while titles use
        `p:titleStyle`. All other placeholders default to `p:otherStyle`.

        Args:
            slide_master: Slide master object associated with the current slide.
            placeholder_type: Placeholder type enum from `PP_PLACEHOLDER`.

        Returns:
            Matching style node from master `p:txStyles` (`p:bodyStyle`, `p:titleStyle`
                or `p:otherStyle`) or None when no styles are defined.
        """
        txStyles = slide_master._element.find(
            ".//p:txStyles", namespaces=self.NAMESPACES
        )
        if txStyles is None:
            return None

        if placeholder_type in (PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT):
            return txStyles.find("p:bodyStyle", namespaces=self.NAMESPACES)

        if placeholder_type == PP_PLACEHOLDER.TITLE:
            return txStyles.find("p:titleStyle", namespaces=self.NAMESPACES)

        return txStyles.find("p:otherStyle", namespaces=self.NAMESPACES)

    def _parse_bullet_from_master_text_styles(
        self, slide_master, placeholder_type, lvl: int
    ) -> tuple[Optional[bool], Optional[str], Optional[str]]:
        """Parse bullet or numbering information from the slide master text styles.

        This looks up the appropriate style bucket in the slide master's `p:txStyles`
        (`titleStyle`, `bodyStyle` or `otherStyle`) and extracts bullet or numbering
        information for the given level.

        Args:
            slide_master: Slide master object associated with the current slide.
            placeholder_type: Placeholder type enum from `PP_PLACEHOLDER`.
            lvl: Paragraph level in the range (0, 8).

        Returns:
            A 3-tuple (`is_list`, `kind`, `detail`) where: `is_list` is True/False/None
                indicating whether this is a list item; `kind` is one of `buChar`,
                `buAutoNum`, `buBlip`, `buNone` or None, describing the marker type;
                `detail` is the bullet character, numbering type string, or None if not
                applicable.
        """
        style = self._get_master_text_style_node(slide_master, placeholder_type)
        if style is None:
            return (None, None, None)

        lvl_pPr = style.find(f".//a:lvl{lvl + 1}pPr", namespaces=self.NAMESPACES)
        is_list, kind, detail = self._parse_bullet_from_paragraph_properties(lvl_pPr)
        return (is_list, kind, detail)

    def _is_list_item(self, paragraph) -> tuple[bool, str]:
        """Determine whether a paragraph should be treated as a list item.

        The method first tries to resolve list style information via the shape that
        owns the paragraph. If that is not possible, it falls back to simpler checks
        based on paragraph properties and level.

        Args:
            paragraph: `python-pptx` paragraph object to inspect.

        Returns:
            A 2-tuple (`is_list`, `bullet_type`) where: `is_list` is True if the
                paragraph is considered a list item, otherwise False; `bullet_type` is
                one of `Bullet`, `Numbered` or `None`, describing the list marker type.
        """
        p = paragraph._element

        # Try to get shape from paragraph if possible
        shape = None
        try:
            # This path works for python-pptx paragraphs
            # First get the text_frame (paragraph's parent)
            text_frame = paragraph._parent
            # Then get the shape (text_frame's parent)
            shape = text_frame._parent
        except AttributeError:
            pass

        if shape is not None:
            marker_info = self._get_effective_list_marker(shape, paragraph)

            # Check if it's definitely a list item
            if marker_info["is_list"] is True or marker_info["kind"] in (
                "buChar",
                "buAutoNum",
                "buBlip",
            ):
                if marker_info["kind"] == "buChar":
                    return (True, "Bullet")
                elif marker_info["kind"] == "buAutoNum":
                    return (True, "Numbered")
                else:
                    return (True, "None")

            # Check if it's definitely not a list item
            if marker_info["is_list"] is False:
                return (False, "None")

            # Fallback to paragraph level check
            if paragraph.level > 0:
                return (True, "None")

            return (False, "None")

        # Fallback to simpler check if shape is not available
        if p.find(".//a:buChar", namespaces={"a": self.NAMESPACES["a"]}) is not None:
            return (True, "Bullet")
        elif (
            p.find(".//a:buAutoNum", namespaces={"a": self.NAMESPACES["a"]}) is not None
        ):
            return (True, "Numbered")
        elif paragraph.level > 0:
            # Most likely a sub-list
            return (True, "None")
        else:
            return (False, "None")

    def _get_effective_list_marker(self, shape, paragraph) -> dict:
        """Return a dictionary describing the effective list marker for a paragraph.

        List marker information can come from several sources: direct paragraph
        properties, shape-level list styles, layout placeholders, or slide master text
        styles. This helper resolves all of these layers and returns a unified view of
        the effective marker.

        Args:
            shape: Shape object that contains the paragraph.
            paragraph: `python-pptx` paragraph object to inspect.

        Returns:
            Information about the list marker in a dictionary, where: `is_list` is
                True/False/None indicating if this is a list item; `kind` is one of
                `buChar`, `buAutoNum`, `buBlip`, `buNone` or None, describing the
                marker type; `detail` is the bullet character or numbering type string,
                or None if not applicable; `level` is the paragraph level in the range
                (0, 8).
        """
        p = paragraph._element
        lvl = self._get_paragraph_level(p)

        # 1) Direct paragraph properties
        pPr = p.find("a:pPr", namespaces=self.NAMESPACES)
        is_list, kind, detail = self._parse_bullet_from_paragraph_properties(pPr)
        if is_list is not None:
            return {
                "is_list": is_list,
                "kind": kind,
                "detail": detail,
                "level": lvl,
            }

        # 2) Shape-level lstStyle (txBody/a:lstStyle)
        txBody = shape._element.find(".//p:txBody", namespaces=self.NAMESPACES)
        is_list, kind, detail = self._parse_bullet_from_text_body_list_style(
            txBody, lvl
        )
        if is_list is not None:
            return {
                "is_list": is_list,
                "kind": kind,
                "detail": detail,
                "level": lvl,
            }

        # 3) Layout placeholder lstStyle (if this is a placeholder)
        if shape.is_placeholder:
            idx = shape.placeholder_format.idx
            layout = shape.part.slide.slide_layout
            layout_ph = None
            try:
                layout_ph = layout.placeholders.get(idx)
            except Exception:
                layout_ph = None

            if layout_ph is not None:
                layout_tx = layout_ph._element.find(
                    ".//p:txBody", namespaces=self.NAMESPACES
                )
                is_list, kind, detail = self._parse_bullet_from_text_body_list_style(
                    layout_tx, lvl
                )

                # Only use layout result if is_list is explicitly True/False
                if is_list is not None:
                    return {
                        "is_list": is_list,
                        "kind": kind,
                        "detail": detail,
                        "level": lvl,
                    }

                # 4) Parse master txStyles
                ph_type = shape.placeholder_format.type
                master = shape.part.slide.slide_layout.slide_master
                is_list, kind, detail = self._parse_bullet_from_master_text_styles(
                    master, ph_type, lvl
                )

                # Check if master has marker information
                if kind in ("buChar", "buAutoNum", "buBlip"):
                    return {
                        "is_list": True,
                        "kind": kind,
                        "detail": detail,
                        "level": lvl,
                    }
                elif is_list is not None:
                    return {
                        "is_list": is_list,
                        "kind": kind,
                        "detail": detail,
                        "level": lvl,
                    }

        return {
            "is_list": None,
            "kind": None,
            "detail": None,
            "level": lvl,
        }

    def _get_shape_position(
        self, shape: BaseShape, attr: Literal["top", "left"]
    ) -> Optional[int]:
        """Return a shape position attribute as an integer for ordering.

        Args:
            shape: A python-pptx shape object. Property getters that raise are
                silently treated as absent.
            attr: The position attribute to read, ``"top"`` or ``"left"``.

        Returns:
            The attribute value cast to ``int``, or ``None`` if the attribute
            is ``None``-valued or its getter raises ``AttributeError``,
            ``ValueError``, or ``TypeError``.
        """
        # python-pptx wraps an element without a registered class, such as a
        # p:contentPart inside a group, in a plain BaseShape whose getters
        # raise AttributeError.
        try:
            value = shape.top if attr == "top" else shape.left
        except (AttributeError, ValueError, TypeError):
            return None

        if value is None:
            return None

        return int(value)

    def _iter_shapes_by_position(
        self, shapes: Iterable[BaseShape]
    ) -> Iterator[BaseShape]:
        """Iterate shapes in visual top-to-bottom, left-to-right order.

        PowerPoint stores shapes in creation/z-order, which can differ from the
        way content is visually read on a slide. Sorting by position keeps split
        text boxes, such as a subheading followed by its bullet text box, adjacent
        in the extracted document.

        Shapes that share a top-edge within ``row_tolerance`` EMUs of their
        immediate predecessor (after sorting by ``top``) are placed in the same
        row and then sorted left-to-right within that row. Adjacency is evaluated
        against the previous shape's ``top``, not the row's first anchor, so that
        a row can span more than one tolerance step when shapes form a contiguous
        band. Shapes whose position attributes are unavailable or raise an
        exception are placed at the end in their original relative order.

        Args:
            shapes: Iterable of python-pptx shape objects to sort. May be a
                slide's top-level shape collection or the children of a group
                shape.

        Yields:
            Shapes from ``shapes`` in visual reading order (top-to-bottom,
            left-to-right).
        """
        from typing import NamedTuple

        class _ShapeInfo(NamedTuple):
            index: int
            shape: BaseShape
            top: int
            left: int

        row_tolerance = self._SHAPE_ROW_TOLERANCE_EMU
        fallback_position = 2**63 - 1

        entries: list[_ShapeInfo] = []
        for index, shape in enumerate(shapes):
            top = self._get_shape_position(shape, "top")
            left = self._get_shape_position(shape, "left")
            entries.append(
                _ShapeInfo(
                    index=index,
                    shape=shape,
                    top=top if top is not None else fallback_position,
                    left=left if left is not None else fallback_position,
                )
            )

        entries.sort(key=lambda si: (si.top, si.index))

        rows: list[list[_ShapeInfo]] = []
        current_row: list[_ShapeInfo] = []
        prev_top = None

        for entry in entries:
            if prev_top is None or entry.top - prev_top <= row_tolerance:
                current_row.append(entry)
                prev_top = entry.top
            else:
                rows.append(current_row)
                current_row = [entry]
                prev_top = entry.top

        if current_row:
            rows.append(current_row)

        for row in rows:
            for si in sorted(row, key=lambda si: (si.left, si.index)):
                yield si.shape

    def _handle_text_elements(
        self, shape, parent_slide, slide_ind, doc: DoclingDocument, slide_size
    ):
        # Lists that are open, outermost first; a deeper paragraph level opens a
        # list nested under the last item of the enclosing one.
        open_lists: list[_OpenList] = []
        doc_label = DocItemLabel.LIST_ITEM

        # Footer, date and slide-number placeholders are slide chrome, like page
        # headers and footers in a PDF.
        content_layer = None
        if shape.is_placeholder and shape.placeholder_format.type in (
            PP_PLACEHOLDER.FOOTER,
            PP_PLACEHOLDER.DATE,
            PP_PLACEHOLDER.SLIDE_NUMBER,
        ):
            content_layer = ContentLayer.FURNITURE

        # Iterate through paragraphs to build up text
        for paragraph in shape.text_frame.paragraphs:
            is_a_list, bullet_type = self._is_list_item(paragraph)
            p = paragraph._element

            fragments = _paragraph_fragments(p)
            p_text = "".join(text for _, text in fragments)
            has_equations = any(label == DocItemLabel.FORMULA for label, _ in fragments)

            prov = self._generate_prov(shape, slide_ind, p_text, slide_size)

            if is_a_list:
                enum_marker = ""
                enumerated = bullet_type == "Numbered"
                level = self._get_paragraph_level(p)

                while len(open_lists) > 1 and open_lists[-1].level > level:
                    open_lists.pop()
                if not open_lists:
                    open_lists.append(
                        _OpenList(
                            group=doc.add_list_group(
                                name="list",
                                parent=parent_slide,
                                content_layer=content_layer,
                            ),
                            level=level,
                        )
                    )
                elif level > open_lists[-1].level:
                    open_lists.append(
                        _OpenList(
                            group=doc.add_list_group(
                                name="list",
                                parent=open_lists[-1].last_item,
                                content_layer=content_layer,
                            ),
                            level=level,
                        )
                    )
                current = open_lists[-1]

                if enumerated:
                    if current.counter == 0:
                        current.counter = self._get_auto_number_start(p) - 1
                    current.counter += 1
                    enum_marker = str(current.counter) + "."

                current.last_item = doc.add_list_item(
                    marker=enum_marker,
                    enumerated=enumerated,
                    parent=current.group,
                    text="" if has_equations else p_text,
                    prov=prov,
                    content_layer=content_layer,
                )
                if has_equations:
                    inline_group = doc.add_inline_group(
                        parent=current.last_item, content_layer=content_layer
                    )
                    for label, text in fragments:
                        doc.add_text(
                            label=label,
                            parent=inline_group,
                            text=text,
                            prov=self._generate_prov(
                                shape, slide_ind, text, slide_size
                            ),
                            content_layer=content_layer,
                        )
            else:  # is paragraph not a list item
                open_lists.clear()
                # Assign proper label to the text, depending if it's a Title or Section Header
                # For other types of text, assign - PARAGRAPH
                doc_label = DocItemLabel.PARAGRAPH
                if shape.is_placeholder:
                    placeholder_type = shape.placeholder_format.type
                    if placeholder_type in [
                        PP_PLACEHOLDER.CENTER_TITLE,
                        PP_PLACEHOLDER.TITLE,
                    ]:
                        # It's a title
                        doc_label = DocItemLabel.TITLE

                if has_equations:
                    has_text = any(
                        label != DocItemLabel.FORMULA and text.strip()
                        for label, text in fragments
                    )
                    fragment_parent = parent_slide
                    if has_text and doc_label == DocItemLabel.TITLE:
                        fragment_parent = doc.add_text(
                            label=doc_label, parent=parent_slide, text="", prov=prov
                        )
                    parent = (
                        doc.add_inline_group(
                            parent=fragment_parent, content_layer=content_layer
                        )
                        if has_text
                        else fragment_parent
                    )
                    for label, text in fragments:
                        if text.strip():
                            doc.add_text(
                                label=label,
                                parent=parent,
                                text=text,
                                prov=self._generate_prov(
                                    shape, slide_ind, text, slide_size
                                ),
                                content_layer=content_layer,
                            )
                else:
                    doc.add_text(
                        label=doc_label,
                        parent=parent_slide,
                        text=p_text,
                        prov=prov,
                        content_layer=content_layer,
                    )
        return

    def _rasterize_metafile(self, image_bytes: bytes) -> Optional[Image.Image]:
        """Rasterize a raw EMF or WMF picture via LibreOffice.

        LibreOffice converts a standalone ``.emf``/``.wmf`` to PDF directly, so
        no wrapper document is needed; the first page is then rendered with
        pypdfium2. This is the same route ``_render_chart_image`` already takes
        for native charts.

        Args:
            image_bytes: Raw metafile data.

        Returns:
            A PIL Image, or None when LibreOffice is unavailable or the
            conversion fails.
        """
        converter = self._get_libreoffice_converter()
        if converter is None:
            if not self._metafile_hint_emitted:
                self._metafile_hint_emitted = True
                _log.warning(_IMAGE_RENDER_HINT)
            return None

        suffix = ".wmf" if image_bytes[:4] == _WMF_PLACEABLE_MAGIC else ".emf"
        temp_dir = Path(mkdtemp())
        try:
            input_path = temp_dir / f"image{suffix}"
            output_path = temp_dir / "image.pdf"
            input_path.write_bytes(image_bytes)
            converter(input_path, output_path)
            if not output_path.exists():
                _log.debug("LibreOffice produced no PDF output for %s", input_path.name)
                return None
            pdf = pypdfium2.PdfDocument(str(output_path))
            page = pdf[0]
            pil_image = crop_whitespace(page.render(scale=2).to_pil())
            page.close()
            pdf.close()
            return pil_image
        except Exception as exc:
            _log.debug("EMF/WMF rasterization via LibreOffice failed: %s", exc)
            return None
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _handle_pictures(self, shape, parent_slide, slide_ind, doc, slide_size):
        # Read the picture out of the package. A shape that slips past other
        # tools' parsers can still fail here, and there is nothing to record.
        try:
            image = shape.image
            image_bytes = image.blob
            im_dpi, _ = image.dpi
            prov = self._generate_prov(shape, slide_ind, "", slide_size)
        except (
            UnidentifiedImageError,
            InvalidXmlError,
            KeyError,
            AttributeError,
            ValueError,
            OSError,
        ) as e:
            warnings.warn(
                f"Skipping malformed picture shape: {e}",
                UserWarning,
                stacklevel=2,
            )
            return

        # Open it with PIL
        image_ref: Optional[ImageRef] = None
        try:
            pil_image = normalize_image_for_png(Image.open(BytesIO(image_bytes)))
            image_ref = ImageRef.from_pil(image=pil_image, dpi=im_dpi)
        except (UnidentifiedImageError, OSError, ValueError) as e:
            if not _is_metafile(image_bytes):
                warnings.warn(
                    f"Skipping malformed picture shape: {e}",
                    UserWarning,
                    stacklevel=2,
                )
                return
            # A metafile Pillow cannot draw on this platform. Rasterize it
            # externally when possible, and otherwise still record the picture
            # so its place on the slide survives -- only the pixels are lost.
            _log.debug("Pillow cannot render this metafile: %s", e)
            rasterized = self._rasterize_metafile(image_bytes)
            if rasterized is not None:
                # Rendered from a PDF rather than decoded from the original
                # picture, so the metafile's own dpi no longer describes it;
                # 72 matches the other LibreOffice-rendered images.
                image_ref = ImageRef.from_pil(image=rasterized, dpi=72)

        doc.add_picture(
            parent=parent_slide,
            image=image_ref,
            caption=None,
            prov=prov,
        )
        return

    def _handle_tables(self, shape, parent_slide, slide_ind, doc, slide_size):
        # Handling tables, images, charts
        if shape.has_table:
            table = shape.table
            table_xml = shape._element

            prov = self._generate_prov(shape, slide_ind, "", slide_size)

            num_cols = 0
            num_rows = len(table.rows)
            tcells = []
            # Access the XML element for the shape that contains the table
            table_xml = shape._element

            for row_idx, row in enumerate(table.rows):
                if len(row.cells) > num_cols:
                    num_cols = len(row.cells)
                for col_idx, cell in enumerate(row.cells):
                    # Access the XML of the cell (this is the 'tc' element in table XML)
                    cell_xml = table_xml.xpath(
                        f".//a:tbl/a:tr[{row_idx + 1}]/a:tc[{col_idx + 1}]"
                    )

                    if not cell_xml:
                        continue  # If no cell XML is found, skip

                    cell_xml = cell_xml[0]  # Get the first matching XML node
                    row_span = cell_xml.get("rowSpan")  # Vertical span
                    col_span = cell_xml.get("gridSpan")  # Horizontal span

                    if row_span is None:
                        row_span = 1
                    else:
                        row_span = int(row_span)

                    if col_span is None:
                        col_span = 1
                    else:
                        col_span = int(col_span)

                    icell = TableCell(
                        text=cell.text.strip(),
                        row_span=row_span,
                        col_span=col_span,
                        start_row_offset_idx=row_idx,
                        end_row_offset_idx=row_idx + row_span,
                        start_col_offset_idx=col_idx,
                        end_col_offset_idx=col_idx + col_span,
                        column_header=row_idx == 0,
                        row_header=False,
                    )
                    if len(cell.text.strip()) > 0:
                        tcells.append(icell)
            # Initialize Docling TableData
            data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=[])
            # Populate
            for tcell in tcells:
                data.table_cells.append(tcell)
            if len(tcells) > 0:
                # If table is not fully empty...
                # Create Docling table
                doc.add_table(parent=parent_slide, data=data, prov=prov)
        return

    @staticmethod
    def _chart_classifications(plots: list[Any]) -> list[PictureClassificationLabel]:
        """Return the classification labels of a chart, one per plot type.

        Each plot is classified by its tag, so a ``c:bar3DChart`` plot is a bar
        chart. The labels follow the drawing order, so the back-most plot gives
        the main label. A label is listed once, even when several plots share
        it.

        Args:
            plots: The chart's plot elements, back-most first.

        Returns:
            The labels, or ``[OTHER_CHART]`` when the chart has no plot.
        """
        labels: list[PictureClassificationLabel] = []
        for x_chart in plots:
            label = _CHART_TAGNAME_TO_CLASSIFICATION.get(
                etree.QName(x_chart).localname, PictureClassificationLabel.OTHER_CHART
            )
            if label not in labels:
                labels.append(label)
        return labels or [PictureClassificationLabel.OTHER_CHART]

    @staticmethod
    def _plot_series(x_chart: Any) -> list[Any]:
        """Return the series of a plot, in the order of their ``c:order``.

        python-pptx has no plot class for some plot types (3-D bar, line and
        pie, stock, surface, bar-of-pie) and no series class for 3-D area
        plots. Their ``c:ser`` elements still have the same structure, so a
        python-pptx series class reads the series of every plot type.

        Args:
            x_chart: The plot element, such as ``c:bar3DChart``.

        Returns:
            One python-pptx series per ``c:ser`` element.
        """
        series_class = (
            XySeries
            if etree.QName(x_chart).localname in _XY_PLOT_TAGNAMES
            else BarSeries
        )
        sers = sorted(x_chart.iterchildren(qn("c:ser")), key=lambda ser: ser.order.val)
        return [series_class(ser) for ser in sers]

    @staticmethod
    def _chart_title_text(chart: Any) -> Optional[str]:
        """Return the chart's title text, or None when it has no title.

        Args:
            chart: A python-pptx ``Chart`` object.

        Returns:
            The stripped title text, or None.
        """
        if not chart.has_title:
            return None
        text = chart.chart_title.text_frame.text.strip()
        return text or None

    @staticmethod
    def _cell_text(value: Any) -> str:
        """Format a chart category label or data value as cell text.

        python-pptx returns numeric series values as floats and empty points as
        None. Integer-valued floats are rendered without a trailing ``.0`` so the
        reconstructed table reads like the source data (``120`` not ``120.0``).

        Args:
            value: A category label or data point (str, float, int, or None).

        Returns:
            The value as a string ("" for None).
        """
        if value is None:
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    @staticmethod
    def _category_axis(chart: Any, x_chart: Any) -> Any:
        """Return the category or date axis that a plot is drawn against.

        The axis is matched by ``c:axId``. ``chart.category_axis`` would return
        the first ``c:catAx`` even when the plot uses a ``c:dateAx``, so it can
        pick the hidden axis of a secondary plot.

        Args:
            chart: A python-pptx ``Chart`` object.
            x_chart: The plot element, such as ``c:barChart``.

        Returns:
            The ``c:catAx`` or ``c:dateAx`` element, or None for a plot without
            one, such as a pie or a scatter plot.
        """
        axis_ids = {
            axis_id.get("val") for axis_id in x_chart.iterchildren(qn("c:axId"))
        }
        for axis in chart.element.chart.plotArea.xpath("./c:catAx | ./c:dateAx"):
            if axis_ids.intersection(axis.xpath("./c:axId/@val")):
                return axis
        return None

    @staticmethod
    def _categories_are_dates(axis: Any, ser: Any) -> bool:
        """Return True when an axis shows its categories as dates.

        A date axis always does. A text axis does when its number format is a
        date format. That format is the axis's own, unless the axis links it to
        the source data, in which case it is the format of the cached
        categories.

        Args:
            axis: The plot's ``c:catAx`` or ``c:dateAx`` element.
            ser: The first ``c:ser`` element of the plot.

        Returns:
            True when the categories are dates.
        """
        if axis.tag == qn("c:dateAx"):
            return True
        number_format = axis.find(qn("c:numFmt"))
        if number_format is not None and number_format.get("sourceLinked") not in (
            "1",
            "true",
        ):
            format_code = number_format.get("formatCode", "")
        else:
            format_code = "".join(ser.xpath("./c:cat//c:numCache/c:formatCode/text()"))
        return _is_date_format(format_code)

    @staticmethod
    def _axis_title_text(axis: Any) -> str:
        """Return the title text of a chart axis, or "" when it has none.

        The XML is read directly, because python-pptx's ``axis_title`` adds a
        title element to an axis without one. A title linked to a cell keeps its
        text in the cell cache instead of in rich text.

        Args:
            axis: A ``c:catAx``, ``c:dateAx`` or ``c:valAx`` element.

        Returns:
            The title text, with one space between paragraphs.
        """
        paragraphs = [
            "".join(paragraph.xpath(".//a:t/text()")).strip()
            for paragraph in axis.xpath("./c:title/c:tx/c:rich/a:p")
        ]
        text = " ".join(paragraph for paragraph in paragraphs if paragraph)
        if not text:
            text = "".join(axis.xpath("./c:title/c:tx/c:strRef//c:v/text()"))
        return text.strip()

    def _category_labels(self, chart: Any, axis: Any, x_chart: Any) -> list[str]:
        """Return the category labels of a plot, with dates in ISO 8601.

        The labels are read from the first series, as python-pptx's
        ``plot.categories`` reads them, because python-pptx has no plot class
        for some plot types. A multi-level axis gives the labels of its leaf
        level. A category without a cached label gives "".

        Args:
            chart: A python-pptx ``Chart`` object.
            axis: The plot's category axis, or None.
            x_chart: The plot element, such as ``c:barChart``.

        Returns:
            One label per category.
        """
        ser = x_chart.find(qn("c:ser"))
        if ser is None:
            return []
        points = ser.xpath("./c:cat//c:lvl[1]/c:pt") or ser.xpath("./c:cat//c:pt")
        texts = {point.idx: point.v.text or "" for point in points}
        labels = [texts.get(idx, "") for idx in range(ser.cat_ptCount_val)]
        if axis is None or not self._categories_are_dates(axis, ser):
            return labels
        date1904 = bool(chart.element.date_1904)
        return [_serial_to_iso(label, date1904) for label in labels]

    def _chart_to_table_data(self, chart: Any, plots: list[Any]) -> Optional[TableData]:
        """Reconstruct a chart's underlying data grid as a TableData.

        Unlike the Excel backend, python-pptx exposes the plotted numbers
        directly on each series (``series.values``), and the chart caches the
        category labels with the first series — no workbook reference
        resolution is needed. ``build_chart_table`` lays the table out.

        The series of every plot become columns in drawing order, so a
        combination chart keeps its line series next to its column series.
        The categories come from the first plot that has them, and the title of
        that plot's category axis goes in the top-left cell.

        Args:
            chart: A python-pptx ``Chart`` object.
            plots: The chart's plot elements, back-most first.

        Returns:
            A TableData, or None if the chart exposes no usable series or no
            populated data row.
        """
        categories: list[str] = []
        corner = ""
        columns: list[tuple[str, list[str]]] = []
        for x_chart in plots:
            if not categories:
                axis = self._category_axis(chart, x_chart)
                categories = self._category_labels(chart, axis, x_chart)
                corner = self._axis_title_text(axis) if axis is not None else ""
            for series in self._plot_series(x_chart):
                name = self._cell_text(series.name) if series.name is not None else ""
                values = [self._cell_text(v) for v in series.values]
                columns.append((name, values))

        if not columns:
            return None
        return build_chart_table(categories, columns, corner)

    def _handle_chart(self, shape, parent_slide, slide_ind, doc, slide_size):
        """Add a native chart shape as a classified PictureItem with its data.

        Each chart becomes a PictureItem whose meta carries (a) the chart-type
        classification and (b) the chart's plotted numbers reconstructed as a
        TableData. A combination chart gets one prediction per plot type, and
        the back-most plot gives the main one. The chart title, if any, becomes
        the picture caption. When ``render_chart_images`` is enabled and
        LibreOffice is available, a rendered image is attached; on any
        rendering failure the picture keeps its classification and data without
        an image.

        Args:
            shape: The python-pptx graphic-frame shape holding the chart.
            parent_slide: The parent slide group.
            slide_ind: Zero-based slide index.
            doc: The DoclingDocument to update.
            slide_size: The slide size, used for provenance.
        """
        try:
            chart = shape.chart
        except (ValueError, KeyError, InvalidXmlError) as exc:
            warnings.warn(
                f"Skipping malformed chart shape: {exc}",
                UserWarning,
                stacklevel=2,
            )
            return

        try:
            plots = list(chart.element.chart.plotArea.xCharts)
        except InvalidXmlError as exc:
            _log.warning("Could not read the plots of a chart: %s", exc)
            plots = []
        labels = self._chart_classifications(plots)
        caption_text = self._chart_title_text(chart)
        try:
            table_data = self._chart_to_table_data(chart, plots)
        except (AttributeError, ValueError, KeyError, InvalidXmlError) as exc:
            _log.warning("Could not extract chart data: %s", exc)
            table_data = None

        prov = self._generate_prov(shape, slide_ind, "", slide_size)

        image_ref = None
        if self._render_charts:
            try:
                chart_image = self._render_chart_image(slide_ind, shape.shape_id)
                if chart_image is not None:
                    image_ref = ImageRef.from_pil(image=chart_image, dpi=72)
            except Exception:
                _log.warning(
                    "could not render a chart image; keeping chart data without image",
                    exc_info=True,
                )

        caption_item = (
            doc.add_text(
                label=DocItemLabel.CAPTION,
                text=caption_text,
                parent=parent_slide,
                prov=self._generate_prov(shape, slide_ind, caption_text, slide_size),
            )
            if caption_text
            else None
        )

        picture = doc.add_picture(
            parent=parent_slide,
            image=image_ref,
            caption=caption_item,
            prov=prov,
        )
        picture.meta = PictureMeta(
            classification=PictureClassificationMetaField(
                predictions=[
                    PictureClassificationPrediction(class_name=label)
                    for label in labels
                ]
            ),
            tabular_chart=(
                TabularChartMetaField(chart_data=table_data)
                if table_data is not None
                else None
            ),
        )
        return

    def _get_libreoffice_converter(self) -> Optional[Callable]:
        """Lazily initialize and return a LibreOffice converter callable.

        The converter accepts ``(input_path, output_path)`` and converts the
        input file to PDF.

        Returns:
            A converter callable, or None when LibreOffice or pypdfium2 is not
            available; `get_docx_to_pdf_converter` checks for both.
        """
        if self.pptx_to_pdf_converter_init:
            return self.pptx_to_pdf_converter

        self.pptx_to_pdf_converter_init = True
        self.pptx_to_pdf_converter = get_docx_to_pdf_converter()
        if self.pptx_to_pdf_converter is None:
            _log.debug("LibreOffice not found — PPTX charts will not be rendered.")
        return self.pptx_to_pdf_converter

    @staticmethod
    def _find_chart_frame(sp_tree, chart_shape_id: int):
        """Return the graphic frame of the chart carrying ``chart_shape_id``.

        The lookup is scoped to ``p:graphicFrame`` elements whose own
        ``p:cNvPr`` carries the id, because shape ids are not reliably unique
        within a slide: an ``mc:AlternateContent`` block repeats the same shape
        with the same id in its ``mc:Choice`` and ``mc:Fallback``, and some
        generators emit duplicates outright. Frames inside an
        ``mc:AlternateContent`` are skipped — python-pptx does not treat them as
        slide shapes, so a chart is never reached through one — and a frame
        actually holding a chart wins over any other candidate.

        Args:
            sp_tree: The slide's ``p:spTree`` element.
            chart_shape_id: The ``shape_id`` of the chart's graphic frame.

        Returns:
            The matching ``p:graphicFrame`` element, or None when the slide
            carries no such frame.
        """
        frames = [
            frame
            for frame in sp_tree.xpath(
                f'.//p:graphicFrame[p:nvGraphicFramePr/p:cNvPr/@id="{int(chart_shape_id)}"]'
            )
            if not any(
                ancestor.tag == _MC_ALTERNATE_CONTENT
                for ancestor in frame.iterancestors()
            )
        ]
        if not frames:
            return None

        chart_path = f"./{qn('a:graphic')}/{qn('a:graphicData')}/{qn('c:chart')}"
        for frame in frames:
            if frame.find(chart_path) is not None:
                return frame
        return frames[0]

    @staticmethod
    def _prune_to_shape(sp_tree, shape_element) -> None:
        """Strip a slide down to one shape, keeping the groups that hold it.

        Walks from ``shape_element`` up to ``sp_tree`` and, at every level,
        drops the siblings that are not on that path. A chart nested in a group
        therefore keeps its enclosing ``p:grpSp`` elements, whose ``chOff`` and
        ``chExt`` define the coordinate space its own ``xfrm`` is expressed in;
        re-parenting the chart to the slide instead would move it. The group
        bookkeeping children are never removed, or the file stops being valid.

        Args:
            sp_tree: The slide's ``p:spTree`` element, where pruning stops.
            shape_element: The element of the shape to keep.
        """
        keep_tags = (qn("p:nvGrpSpPr"), qn("p:grpSpPr"))

        node = shape_element
        while node is not sp_tree:
            parent = node.getparent()
            if parent is None:
                return
            for sibling in list(parent):
                if sibling is not node and sibling.tag not in keep_tags:
                    parent.remove(sibling)
            node = parent

    def _isolate_chart_presentation(
        self, slide_ind: int, chart_shape_id: int, out_path: Path
    ) -> bool:
        """Save a copy of the presentation holding only the target chart.

        A fresh copy of the loaded presentation is reopened, every slide except
        the chart's is removed, and on that slide every shape except the chart
        is removed. LibreOffice then renders a single-chart page. The chart's
        graphic frame is looked up anywhere in the shape tree, so one nested in
        a group is found too and only its sibling shapes are dropped.

        Nothing is written when the frame cannot be found: rendering the whole
        untouched slide would attach a slide screenshot as the chart's image,
        which misleads downstream consumers more than having no image at all.

        Args:
            slide_ind: Zero-based index of the slide holding the chart.
            chart_shape_id: The ``shape_id`` of the chart's graphic frame.
            out_path: Destination path for the trimmed ``.pptx``.

        Returns:
            True when the trimmed presentation was written.
        """
        if self.pptx_obj is None:
            return False

        buf = BytesIO()
        self.pptx_obj.save(buf)
        buf.seek(0)
        prs = Presentation(buf)

        slide_id_list = prs.slides._sldIdLst
        slide_ids = list(slide_id_list)
        if slide_ind >= len(slide_ids):
            return False
        target_slide = prs.slides[slide_ind]

        for idx, slide_id in enumerate(slide_ids):
            if idx != slide_ind:
                slide_id_list.remove(slide_id)

        sp_tree = target_slide.shapes._spTree
        chart_frame = self._find_chart_frame(sp_tree, chart_shape_id)
        if chart_frame is None:
            _log.warning(
                "No chart graphic frame with shape id %s on slide %s; "
                "keeping the chart data without an image.",
                chart_shape_id,
                slide_ind + 1,
            )
            return False
        self._prune_to_shape(sp_tree, chart_frame)

        prs.save(str(out_path))
        return True

    def _render_chart_image(
        self, slide_ind: int, chart_shape_id: int
    ) -> Optional[Image.Image]:
        """Render a native chart to an image via LibreOffice.

        PPTX stores charts as vector definitions with no embedded raster. To
        obtain a picture we isolate the chart onto a throwaway single-slide
        presentation, convert that to PDF with LibreOffice — the same external
        tool already used for EMF/WMF images — and rasterize the first page with
        pypdfium2, trimming the surrounding whitespace.

        Args:
            slide_ind: Zero-based index of the slide holding the chart.
            chart_shape_id: The ``shape_id`` of the chart's graphic frame.

        Returns:
            A PIL Image, or None when LibreOffice is unavailable or the
            conversion fails.
        """
        converter = self._get_libreoffice_converter()
        if converter is None:
            return None

        temp_dir = Path(mkdtemp())
        try:
            input_path = temp_dir / "chart.pptx"
            output_path = temp_dir / "chart.pdf"
            if not self._isolate_chart_presentation(
                slide_ind, chart_shape_id, input_path
            ):
                return None
            converter(input_path, output_path)
            if not output_path.exists():
                _log.debug("LibreOffice produced no PDF output for a chart")
                return None
            pdf = pypdfium2.PdfDocument(str(output_path))
            page = pdf[0]
            pil_image = crop_whitespace(page.render(scale=2).to_pil())
            page.close()
            pdf.close()
            return pil_image
        except Exception as exc:
            _log.debug("Chart rendering via LibreOffice failed: %s", exc)
            return None
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _walk_linear(
        self,
        pptx_obj: presentation.Presentation,
        doc: DoclingDocument,
        author_map: dict[str, tuple[str, str]],
        start_page: int = 1,
        end_page: Optional[int] = None,
    ) -> DoclingDocument:
        # Units of size in PPTX by default are EMU units (English Metric Units)
        slide_width = pptx_obj.slide_width
        slide_height = pptx_obj.slide_height

        self._render_charts = (
            isinstance(self.options, MsPowerpointBackendOptions)
            and self.options.render_chart_images
        )
        if self._render_charts and self._get_libreoffice_converter() is None:
            _log.warning(_CHART_RENDER_HINT)
            self._render_charts = False

        max_levels = 10
        parents = {}  # type: ignore
        for i in range(max_levels):
            parents[i] = None

        # Loop through each slide
        selected_slides = list(enumerate(pptx_obj.slides))[start_page - 1 : end_page]
        for slide_ind, slide in selected_slides:
            parent_slide = doc.add_group(
                name=f"slide-{slide_ind}", label=GroupLabel.CHAPTER, parent=parents[0]
            )

            slide_size = Size(width=slide_width, height=slide_height)
            doc.add_page(page_no=slide_ind + 1, size=slide_size)

            def _safe_shape_type(shape):
                """Return shape.shape_type, or None if unrecognized.

                python-pptx raises NotImplementedError for <p:sp> elements
                that don't match any known shape category (placeholder,
                freeform, autoshape, textbox).
                """
                try:
                    return shape.shape_type
                except NotImplementedError:
                    _log.debug("Skipping shape with unrecognized type: %s", shape.name)
                    return None

            def handle_shapes(shape, parent_slide, slide_ind, doc, slide_size):
                if not self._is_hidden_shape(shape):
                    handle_shape_content(
                        shape, parent_slide, slide_ind, doc, slide_size
                    )
                    return
                # A hidden shape still goes through the normal handlers, so nothing
                # is lost; whatever they add for it (and, for a group, for every
                # shape inside it) is then moved to the invisible layer.
                known = self._body_item_refs(doc, parent_slide)
                handle_shape_content(shape, parent_slide, slide_ind, doc, slide_size)
                self._hide_new_items(doc, parent_slide, known)

            def handle_shape_content(shape, parent_slide, slide_ind, doc, slide_size):
                handle_groups(shape, parent_slide, slide_ind, doc, slide_size)
                if shape.has_table:
                    # Handle Tables
                    self._handle_tables(shape, parent_slide, slide_ind, doc, slide_size)
                if shape.has_chart:
                    self._handle_chart(shape, parent_slide, slide_ind, doc, slide_size)
                if isinstance(shape, Picture):
                    # Handle Pictures, including those inserted into a picture placeholder
                    self._handle_pictures(
                        shape, parent_slide, slide_ind, doc, slide_size
                    )
                # If shape doesn't have any text, move on to the next shape
                if not isinstance(shape, Shape):
                    return
                if shape.text is None:
                    return
                if not shape.has_text_frame:
                    _log.warning("Warning: shape has text but not text_frame")
                    return
                if not shape.text.strip() and not any(
                    _paragraph_fragments(paragraph._element)
                    for paragraph in shape.text_frame.paragraphs
                ):
                    return
                # Handle other text elements, including lists (bullet lists, numbered
                # lists)
                first_new_text = len(doc.texts)
                self._handle_text_elements(
                    shape, parent_slide, slide_ind, doc, slide_size
                )
                self._apply_shape_hyperlink(shape, doc.texts[first_new_text:])
                return

            def handle_groups(shape, parent_slide, slide_ind, doc, slide_size):
                if _safe_shape_type(shape) == MSO_SHAPE_TYPE.GROUP:
                    for groupedshape in self._iter_shapes_by_position(shape.shapes):
                        handle_shapes(
                            groupedshape, parent_slide, slide_ind, doc, slide_size
                        )

            # Loop through each shape in the slide
            for shape in self._iter_shapes_by_position(slide.shapes):
                handle_shapes(shape, parent_slide, slide_ind, doc, slide_size)

            # Handle notes slide
            if slide.has_notes_slide:
                notes_slide = slide.notes_slide
                if notes_slide.notes_text_frame is not None:
                    notes_text = notes_slide.notes_text_frame.text.strip()
                    if notes_text:
                        bbox = BoundingBox(l=0, t=0, r=0, b=0)
                        prov = ProvenanceItem(
                            page_no=slide_ind + 1,
                            charspan=[0, len(notes_text)],
                            bbox=bbox,
                        )
                        doc.add_text(
                            label=DocItemLabel.TEXT,
                            parent=parent_slide,
                            text=notes_text,
                            prov=prov,
                            content_layer=ContentLayer.NOTES,
                        )

            # Extract comments for this slide
            self._extract_slide_comments(
                slide, slide_ind, author_map, doc, parent_slide
            )

            if slide._element.get("show") in ("0", "false"):
                self._hide_slide(doc, parent_slide)

        return doc

    @staticmethod
    def _is_hidden_shape(shape) -> bool:
        """Return True if the shape is hidden in PowerPoint's Selection Pane.

        The flag is ``hidden`` on the shape's ``p:cNvPr`` element, which every
        shape kind carries (text boxes, pictures, groups, graphic frames and
        connectors).
        """
        hidden = shape._element.xpath("./*/p:cNvPr/@hidden")
        return bool(hidden) and hidden[0] in ("1", "true")

    @staticmethod
    def _shape_hyperlink(shape) -> Optional[Union[AnyUrl, Path]]:
        """Return the link attached to the whole shape, if it has one.

        PowerPoint stores such a link as ``a:hlinkClick`` under the shape's
        ``p:cNvPr``. Only a link to a URL or a file is returned: other click
        actions, such as a jump to another slide, also carry a relationship
        target (``slide2.xml``), but it is not something a reader can follow.
        """
        try:
            click_action = shape.click_action
            if click_action.action not in (PP_ACTION.HYPERLINK, PP_ACTION.OPEN_FILE):
                return None
            address = click_action.hyperlink.address
        except (AttributeError, KeyError):
            # No click action on this shape kind, or its relationship is gone.
            return None
        if not address:
            return None
        if not urlparse(address).scheme:
            return Path(address)
        try:
            return AnyUrl(address)
        except ValidationError:
            _log.warning("Skipping malformed hyperlink address: %r", address)
            return None

    @classmethod
    def _apply_shape_hyperlink(cls, shape, items: Sequence[TextItem]) -> None:
        """Give the shape's link to its text items that have no link of their own."""
        if not items:
            return
        hyperlink = cls._shape_hyperlink(shape)
        if hyperlink is None:
            return
        for item in items:
            if item.hyperlink is None:
                item.hyperlink = hyperlink

    @staticmethod
    def _body_item_refs(doc: DoclingDocument, parent_slide: GroupItem) -> set[str]:
        return {
            item.self_ref
            for item, _ in doc.iterate_items(
                root=parent_slide,
                with_groups=True,
                traverse_pictures=True,
                included_content_layers={ContentLayer.BODY},
            )
        }

    @staticmethod
    def _hide_new_items(
        doc: DoclingDocument, parent_slide: GroupItem, known: set[str]
    ) -> None:
        """Move the body items added since ``known`` was taken to the invisible layer.

        PowerPoint does not present a hidden shape, so its content is treated
        like the content of a hidden slide: available, but out of the default
        exports.
        """
        items = [
            item
            for item, _ in doc.iterate_items(
                root=parent_slide,
                with_groups=True,
                traverse_pictures=True,
                included_content_layers={ContentLayer.BODY},
            )
            if item.self_ref not in known
        ]
        for item in items:
            item.content_layer = ContentLayer.INVISIBLE

    @staticmethod
    def _hide_slide(doc: DoclingDocument, parent_slide: GroupItem) -> None:
        """Move the content of a hidden slide to the invisible layer.

        PowerPoint does not present a slide marked ``show="0"`` (Hide Slide), so
        it is kept like a hidden Excel sheet: available, but out of the default
        exports. Speaker notes and comments keep the notes layer.
        """
        items = [
            item
            for item, _ in doc.iterate_items(
                root=parent_slide,
                with_groups=True,
                included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE},
            )
        ]
        for item in items:
            item.content_layer = ContentLayer.INVISIBLE

    def _build_comment_author_map(
        self, pptx_obj: presentation.Presentation
    ) -> dict[str, tuple[str, str]]:
        """Build a map of comment author IDs to (name, initials).

        Args:
            pptx_obj: The PowerPoint presentation object.

        Returns:
            Dictionary mapping author ID to (name, initials) tuple.
        """
        author_map: dict[str, tuple[str, str]] = {}
        try:
            for rel in pptx_obj.part.rels.values():
                if rel.reltype == self.COMMENT_AUTHORS_REL:
                    root = etree.fromstring(
                        rel.target_part.blob, parser=_SAFE_XML_PARSER
                    )
                    author_map = {
                        author_el.get("id", ""): (
                            author_el.get("name", ""),
                            author_el.get("initials", ""),
                        )
                        for author_el in root.findall(
                            "p:cmAuthor", namespaces=self.NAMESPACES
                        )
                    }
        except Exception as e:
            _log.debug(f"Could not parse PPTX comment authors: {e}")
        return author_map

    def _extract_slide_comments(
        self,
        slide,
        slide_idx: int,
        author_map: dict[str, tuple[str, str]],
        doc: DoclingDocument,
        parent_slide,
    ) -> None:
        """Extract and add comments for a specific slide.

        Args:
            slide: The slide object to extract comments from.
            slide_idx: Zero-based slide index.
            author_map: Dictionary mapping author ID to (name, initials).
            doc: The DoclingDocument to add comments to.
            parent_slide: The parent slide group to link comments to.
        """
        try:
            slide_part = slide.part
        except Exception as e:
            _log.debug(
                f"Could not access slide part for slide {slide_idx + 1}, "
                f"skipping comment extraction: {e}"
            )
            return

        for rel in slide_part.rels.values():
            if rel.reltype != self.COMMENT_REL:
                continue
            try:
                root = etree.fromstring(rel.target_part.blob, parser=_SAFE_XML_PARSER)
                for cm in root.findall("p:cm", namespaces=self.NAMESPACES):
                    author_id = cm.get("authorId", "")
                    dt = cm.get("dt", "")
                    text_el = cm.find("p:text", namespaces=self.NAMESPACES)
                    raw_text = (
                        (text_el.text or "").strip() if text_el is not None else ""
                    )
                    if not raw_text:
                        continue

                    name, initials = author_map.get(author_id, ("", ""))
                    metadata_parts = []
                    if name:
                        author_str = f"author: {name}"
                        if initials:
                            author_str += f" ({initials})"
                        metadata_parts.append(author_str)
                    if dt:
                        metadata_parts.append(f"time: {dt}")
                    prefix = ", ".join(metadata_parts)
                    full_text = f"[{prefix}]: {raw_text}" if prefix else raw_text

                    comment_idx = cm.get("idx", str(slide_idx))
                    comment_group = doc.add_group(
                        label=GroupLabel.COMMENT_SECTION,
                        name=f"comment-slide{slide_idx + 1}-{comment_idx}",
                        content_layer=ContentLayer.NOTES,
                    )
                    doc.add_comment(
                        text=full_text,
                        targets=None,
                        parent=comment_group,
                    )
                    _log.debug(
                        f"Added PPTX comment slide {slide_idx + 1} idx={comment_idx}"
                    )
            except Exception as e:
                _log.debug(f"Could not parse comments for slide {slide_idx}: {e}")
