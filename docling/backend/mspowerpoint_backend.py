import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupLabel,
    ImageRef,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from docling_core.types.doc.document import ContentLayer
from lxml import etree
from PIL import Image, UnidentifiedImageError
from pptx import Presentation, presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER
from pptx.oxml.text import CT_TextLineBreak
from typing_extensions import override

from docling.backend.abstract_backend import (
    DeclarativeDocumentBackend,
    PaginatedDocumentBackend,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


class MsPowerpointDocumentBackend(DeclarativeDocumentBackend, PaginatedDocumentBackend):
    def __init__(
        self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]
    ) -> None:
        super().__init__(in_doc, path_or_stream)
        self.namespaces = {
            "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
            "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
            "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
        }
        # Powerpoint file:
        self.path_or_stream: Union[BytesIO, Path] = path_or_stream

        self.pptx_obj: Optional[presentation.Presentation] = None
        self.valid: bool = False
        try:
            if isinstance(self.path_or_stream, BytesIO):
                self.pptx_obj = Presentation(self.path_or_stream)
            elif isinstance(self.path_or_stream, Path):
                self.pptx_obj = Presentation(str(self.path_or_stream))

            self.valid = True
        except Exception as e:
            raise RuntimeError(
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
        return {InputFormat.PPTX}

    @override
    def convert(self) -> DoclingDocument:
        """Parse the PPTX into a structured document model.

        Returns:
            The parsed document.
        """
        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="application/vnd.ms-powerpoint",
            binary_hash=self.document_hash,
        )

        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)
        if self.pptx_obj:
            doc = self._walk_linear(self.pptx_obj, doc)

        return doc

    def _generate_prov(
        self, shape, slide_ind, text="", slide_size=Size(width=1, height=1)
    ):
        if shape.left:
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
        shape_bbox = BoundingBox.from_tuple(shape_bbox, origin=CoordOrigin.BOTTOMLEFT)
        prov = ProvenanceItem(
            page_no=slide_ind + 1, charspan=[0, len(text)], bbox=shape_bbox
        )

        return prov

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
        pPr = paragraph.find("a:pPr", namespaces=self.namespaces)
        if pPr is not None and "lvl" in pPr.attrib:
            try:
                return int(pPr.get("lvl"))
            except ValueError:
                pass
        return 0

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
        if pPr.find("a:buNone", namespaces=self.namespaces) is not None:
            return (False, "buNone", None)

        # Bullet character
        buChar = pPr.find("a:buChar", namespaces=self.namespaces)
        if buChar is not None:
            return (True, "buChar", buChar.get("char"))

        # Auto numbering
        buAuto = pPr.find("a:buAutoNum", namespaces=self.namespaces)
        if buAuto is not None:
            return (True, "buAutoNum", buAuto.get("type"))

        # Picture bullet
        buBlip = pPr.find("a:buBlip", namespaces=self.namespaces)
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
        return lstStyle.find(tag, namespaces=self.namespaces)

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
        lstStyle = txBody.find("a:lstStyle", namespaces=self.namespaces)
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
            ".//p:txStyles", namespaces=self.namespaces
        )
        if txStyles is None:
            return None

        if placeholder_type in (PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT):
            return txStyles.find("p:bodyStyle", namespaces=self.namespaces)

        if placeholder_type == PP_PLACEHOLDER.TITLE:
            return txStyles.find("p:titleStyle", namespaces=self.namespaces)

        return txStyles.find("p:otherStyle", namespaces=self.namespaces)

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

        lvl_pPr = style.find(f".//a:lvl{lvl + 1}pPr", namespaces=self.namespaces)
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
        if p.find(".//a:buChar", namespaces={"a": self.namespaces["a"]}) is not None:
            return (True, "Bullet")
        elif (
            p.find(".//a:buAutoNum", namespaces={"a": self.namespaces["a"]}) is not None
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
        pPr = p.find("a:pPr", namespaces=self.namespaces)
        is_list, kind, detail = self._parse_bullet_from_paragraph_properties(pPr)
        if is_list is not None:
            return {
                "is_list": is_list,
                "kind": kind,
                "detail": detail,
                "level": lvl,
            }

        # 2) Shape-level lstStyle (txBody/a:lstStyle)
        txBody = shape._element.find(".//p:txBody", namespaces=self.namespaces)
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
        layout_result = None
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
                    ".//p:txBody", namespaces=self.namespaces
                )
                is_list, kind, detail = self._parse_bullet_from_text_body_list_style(
                    layout_tx, lvl
                )

                # Only use layout result if is_list is explicitly True/False
                if is_list is not None:
                    layout_result = {
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

            # If layout has explicit is_list value but master didn't override it, use
            # layout
            if layout_result is not None:
                return layout_result

        return {
            "is_list": None,
            "kind": None,
            "detail": None,
            "level": lvl,
        }

    def _handle_text_elements(
        self, shape, parent_slide, slide_ind, doc: DoclingDocument, slide_size
    ):
        is_list_group_created = False
        enum_list_item_value = 0
        new_list = None
        doc_label = DocItemLabel.LIST_ITEM
        prov = self._generate_prov(shape, slide_ind, shape.text.strip(), slide_size)

        # Iterate through paragraphs to build up text
        for paragraph in shape.text_frame.paragraphs:
            is_a_list, bullet_type = self._is_list_item(paragraph)
            p = paragraph._element

            # Convert line breaks to spaces and accumulate text
            p_text = ""
            for e in p.content_children:
                if isinstance(e, CT_TextLineBreak):
                    p_text += " "
                else:
                    p_text += e.text

            if is_a_list:
                enum_marker = ""
                enumerated = bullet_type == "Numbered"

                if not is_list_group_created:
                    new_list = doc.add_list_group(
                        name="list",
                        parent=parent_slide,
                    )
                    is_list_group_created = True
                    enum_list_item_value = 0

                if enumerated:
                    enum_list_item_value += 1
                    enum_marker = str(enum_list_item_value) + "."

                doc.add_list_item(
                    marker=enum_marker,
                    enumerated=enumerated,
                    parent=new_list,
                    text=p_text,
                    prov=prov,
                )
            else:  # is paragraph not a list item
                if is_list_group_created:
                    is_list_group_created = False
                    new_list = None
                    enum_list_item_value = 0
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
                    elif placeholder_type == PP_PLACEHOLDER.SUBTITLE:
                        DocItemLabel.SECTION_HEADER

                # output accumulated inline text:
                doc.add_text(
                    label=doc_label,
                    parent=parent_slide,
                    text=p_text,
                    prov=prov,
                )
        return

    def _handle_title(self, shape, parent_slide, slide_ind, doc):
        placeholder_type = shape.placeholder_format.type
        txt = shape.text.strip()
        prov = self._generate_prov(shape, slide_ind, txt)

        if len(txt.strip()) > 0:
            # title = slide.shapes.title.text if slide.shapes.title else "No title"
            if placeholder_type in [PP_PLACEHOLDER.CENTER_TITLE, PP_PLACEHOLDER.TITLE]:
                _log.info(f"Title found: {shape.text}")
                doc.add_text(
                    label=DocItemLabel.TITLE, parent=parent_slide, text=txt, prov=prov
                )
            elif placeholder_type == PP_PLACEHOLDER.SUBTITLE:
                _log.info(f"Subtitle found: {shape.text}")
                # Using DocItemLabel.FOOTNOTE, while SUBTITLE label is not avail.
                doc.add_text(
                    label=DocItemLabel.SECTION_HEADER,
                    parent=parent_slide,
                    text=txt,
                    prov=prov,
                )
        return

    def _handle_pictures(self, shape, parent_slide, slide_ind, doc, slide_size):
        # Open it with PIL
        try:
            # Get the image bytes
            image = shape.image
            image_bytes = image.blob
            im_dpi, _ = image.dpi
            pil_image = Image.open(BytesIO(image_bytes))

            # shape has picture
            prov = self._generate_prov(shape, slide_ind, "", slide_size)
            doc.add_picture(
                parent=parent_slide,
                image=ImageRef.from_pil(image=pil_image, dpi=im_dpi),
                caption=None,
                prov=prov,
            )
        except (UnidentifiedImageError, OSError) as e:
            _log.warning(f"Warning: image cannot be loaded by Pillow: {e}")
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

    def _walk_linear(
        self, pptx_obj: presentation.Presentation, doc: DoclingDocument
    ) -> DoclingDocument:
        # Units of size in PPTX by default are EMU units (English Metric Units)
        slide_width = pptx_obj.slide_width
        slide_height = pptx_obj.slide_height

        max_levels = 10
        parents = {}  # type: ignore
        for i in range(max_levels):
            parents[i] = None

        # Loop through each slide
        for _, slide in enumerate(pptx_obj.slides):
            slide_ind = pptx_obj.slides.index(slide)
            parent_slide = doc.add_group(
                name=f"slide-{slide_ind}", label=GroupLabel.CHAPTER, parent=parents[0]
            )

            slide_size = Size(width=slide_width, height=slide_height)
            doc.add_page(page_no=slide_ind + 1, size=slide_size)

            def handle_shapes(shape, parent_slide, slide_ind, doc, slide_size):
                handle_groups(shape, parent_slide, slide_ind, doc, slide_size)
                if shape.has_table:
                    # Handle Tables
                    self._handle_tables(shape, parent_slide, slide_ind, doc, slide_size)
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    # Handle Pictures
                    if hasattr(shape, "image"):
                        self._handle_pictures(
                            shape, parent_slide, slide_ind, doc, slide_size
                        )
                # If shape doesn't have any text, move on to the next shape
                if not hasattr(shape, "text"):
                    return
                if shape.text is None:
                    return
                if len(shape.text.strip()) == 0:
                    return
                if not shape.has_text_frame:
                    _log.warning("Warning: shape has text but not text_frame")
                    return
                # Handle other text elements, including lists (bullet lists, numbered
                # lists)
                self._handle_text_elements(
                    shape, parent_slide, slide_ind, doc, slide_size
                )
                return

            def handle_groups(shape, parent_slide, slide_ind, doc, slide_size):
                if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                    for groupedshape in shape.shapes:
                        handle_shapes(
                            groupedshape, parent_slide, slide_ind, doc, slide_size
                        )

            # Loop through each shape in the slide
            for shape in slide.shapes:
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
                            content_layer=ContentLayer.FURNITURE,
                        )

        return doc
