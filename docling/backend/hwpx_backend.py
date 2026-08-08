import logging
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupLabel,
    ImageRef,
    NodeItem,
    TableCell,
    TableData,
)
from lxml import etree
from PIL import Image, UnidentifiedImageError
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError

_log = logging.getLogger(__name__)

# OWPML (Open Word-processor Markup Language, KS X 6101) namespaces. HWPX pins
# these 2011 revisions across every writer version, so binding the prefixes
# statically is safe and avoids re-reading them from each document.
_HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"
_HH = "http://www.hancom.co.kr/hwpml/2011/head"
_HC = "http://www.hancom.co.kr/hwpml/2011/core"

# The value stored (uncompressed, first) in the container's ``mimetype`` entry.
_HWPX_MIMETYPE = "application/hwp+zip"

# DocumentOrigin only accepts MIME types registered with the stdlib ``mimetypes``
# module or docling-core's allow-list, neither of which yet knows the HWPX type;
# fall back to the (valid) container type for the recorded origin.
_ORIGIN_MIMETYPE = "application/zip"

# Outline levels are 0-based in OWPML; DoclingDocument heading levels start at 1.
_MAX_HEADING_LEVEL = 9


def _localname(element: etree._Element) -> str:
    return etree.QName(element).localname


class HwpxDocumentBackend(DeclarativeDocumentBackend):
    """Declarative backend for HWPX (Hangul Word Processor XML / OWPML).

    HWPX is the packaging the Korean Hangul word processor uses for the OWPML
    schema (KS X 6101): a ZIP container, structurally analogous to DOCX, holding
    a ``mimetype`` entry (``application/hwp+zip``), one or more section files
    under ``Contents/``, the shared style table in ``Contents/header.xml``, a
    package manifest in ``Contents/content.hpf``, and embedded binaries under
    ``BinData/``. It is parsed natively here with the standard-library
    ``zipfile`` plus ``lxml`` -- no HWPX-specific third-party dependency.

    The legacy binary ``.hwp`` format is an OLE/CFB compound file, not a ZIP, and
    is not handled by this backend; such inputs are reported as unsupported.
    """

    @override
    def __init__(self, in_doc: InputDocument, path_or_stream: Union[BytesIO, Path]):
        super().__init__(in_doc, path_or_stream)

        # Zip entries are read eagerly into memory so embedded binaries (BinData)
        # stay reachable during conversion without holding the archive open.
        self._entries: dict[str, bytes] = {}
        self._para_roles: dict[str, tuple[str, int]] = {}
        self._bin_items: dict[str, tuple[str, str]] = {}
        self._section_names: list[str] = []
        self._valid = False

        try:
            raw = (
                self.path_or_stream.getvalue()
                if isinstance(self.path_or_stream, BytesIO)
                else self.path_or_stream.read_bytes()
            )
            with zipfile.ZipFile(BytesIO(raw)) as zf:
                for name in zf.namelist():
                    self._entries[name] = zf.read(name)
        except (zipfile.BadZipFile, OSError) as exc:
            raise DocumentLoadError(
                f"Could not read HWPX document with hash {self.document_hash}: "
                "the file is not a valid HWPX (ZIP) container. Binary '.hwp' and "
                "DRM/distribution documents are not supported."
            ) from exc

        mimetype = self._entries.get("mimetype", b"").decode("ascii", "ignore").strip()
        self._section_names = sorted(
            (n for n in self._entries if _is_section_entry(n)),
            key=_section_sort_key,
        )
        if mimetype != _HWPX_MIMETYPE or not self._section_names:
            raise DocumentLoadError(
                f"File with hash {self.document_hash} is not a supported HWPX "
                "document (missing 'application/hwp+zip' marker or section content)."
            )
        self._valid = True

    @override
    def is_valid(self) -> bool:
        return self._valid

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.HWPX}

    @override
    def convert(self) -> DoclingDocument:
        if not self._valid:
            raise RuntimeError(
                f"Cannot convert invalid HWPX document with hash {self.document_hash}."
            )

        self._load_header()

        origin = DocumentOrigin(
            filename=self.file.name or "file.hwpx",
            mimetype=_ORIGIN_MIMETYPE,
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)

        for name in self._section_names:
            try:
                section = etree.fromstring(self._entries[name])
            except etree.XMLSyntaxError as exc:
                raise DocumentLoadError(
                    f"Malformed OWPML in '{name}' of HWPX document "
                    f"{self.document_hash}."
                ) from exc
            paragraphs = section.findall(f"{{{_HP}}}p")
            self._add_paragraphs(paragraphs, doc, None)

        return doc

    # -- header / manifest ----------------------------------------------------

    def _load_header(self) -> None:
        """Index paragraph outline roles and the binary-item manifest.

        A paragraph's role (heading, numbered/bulleted list item, or plain body
        text) is driven by the ``<hh:heading>`` of its referenced ``paraPr``, and
        embedded pictures resolve through the package manifest in
        ``content.hpf``.
        """
        header_bytes = self._entries.get("Contents/header.xml")
        if header_bytes is not None:
            header = etree.fromstring(header_bytes)
            for para_pr in header.iter(f"{{{_HH}}}paraPr"):
                para_id = para_pr.get("id")
                heading = para_pr.find(f"{{{_HH}}}heading")
                if para_id is not None and heading is not None:
                    self._para_roles[para_id] = (
                        heading.get("type") or "NONE",
                        int(heading.get("level") or 0),
                    )

        manifest_bytes = self._entries.get("Contents/content.hpf")
        if manifest_bytes is not None:
            manifest = etree.fromstring(manifest_bytes)
            for item in manifest.iter():
                if _localname(item) != "item":
                    continue
                item_id = item.get("id")
                href = item.get("href")
                if item_id is not None and href is not None:
                    self._bin_items[item_id] = (href, item.get("media-type") or "")

    def _role(self, paragraph: etree._Element) -> tuple[str, int]:
        para_pr = paragraph.get("paraPrIDRef")
        return self._para_roles.get(para_pr or "", ("NONE", 0))

    # -- paragraph flow -------------------------------------------------------

    def _add_paragraphs(
        self,
        paragraphs: list[etree._Element],
        doc: DoclingDocument,
        parent: Optional[NodeItem],
    ) -> None:
        # Consecutive list paragraphs are grouped; the stack tracks one open list
        # group per outline depth so nested items land under their parent item.
        list_stack: list[list] = []

        for paragraph in paragraphs:
            role, level = self._role(paragraph)
            text, tables, pictures, equations, notes = self._collect_run_content(
                paragraph
            )

            is_list_item = role in ("NUMBER", "BULLET") and bool(text)
            if not is_list_item:
                list_stack.clear()

            if role == "OUTLINE" and text:
                doc.add_heading(
                    text=text,
                    level=min(level + 1, _MAX_HEADING_LEVEL),
                    parent=parent,
                )
            elif is_list_item:
                self._append_list_item(
                    text, role == "NUMBER", level, list_stack, doc, parent
                )
            elif text:
                doc.add_text(label=DocItemLabel.TEXT, text=text, parent=parent)

            for equation in equations:
                self._add_formula(equation, doc, parent)
            for table in tables:
                self._add_table(table, doc, parent)
            for picture in pictures:
                self._add_picture(picture, doc, parent)
            for note in notes:
                self._add_note(note, doc, parent)

    def _append_list_item(
        self,
        text: str,
        enumerated: bool,
        level: int,
        list_stack: list[list],
        doc: DoclingDocument,
        parent: Optional[NodeItem],
    ) -> None:
        while list_stack and list_stack[-1][0] > level:
            list_stack.pop()
        if not list_stack or list_stack[-1][0] < level:
            group_parent: Optional[NodeItem] = parent
            if list_stack:
                group_parent = list_stack[-1][2] or list_stack[-1][1]
            group = doc.add_group(
                label=GroupLabel.LIST, name="list", parent=group_parent
            )
            list_stack.append([level, group, None])
        item = doc.add_list_item(
            text=text, enumerated=enumerated, parent=list_stack[-1][1]
        )
        list_stack[-1][2] = item

    def _collect_run_content(
        self, paragraph: etree._Element
    ) -> tuple[
        str,
        list[etree._Element],
        list[etree._Element],
        list[etree._Element],
        list[etree._Element],
    ]:
        """Split a paragraph's runs into merged text plus anchored objects.

        Text of every ``<hp:t>`` is concatenated in run order (runs are style
        boundaries, not word boundaries, so no separator is inserted). Tables,
        pictures, equations and foot/endnotes are returned as opaque elements;
        their descendants are never merged into the paragraph text.
        """
        text_parts: list[str] = []
        tables: list[etree._Element] = []
        pictures: list[etree._Element] = []
        equations: list[etree._Element] = []
        notes: list[etree._Element] = []

        def walk(container: etree._Element) -> None:
            for child in container:
                tag = _localname(child)
                if tag == "t":
                    text_parts.append("".join(child.itertext()))
                elif tag == "tbl":
                    tables.append(child)
                elif tag == "pic":
                    pictures.append(child)
                elif tag == "equation":
                    equations.append(child)
                elif tag in ("footNote", "endNote"):
                    notes.append(child)
                elif tag == "ctrl":
                    # Foot/endnotes and other inline controls nest one level down.
                    walk(child)

        for run in paragraph.findall(f"{{{_HP}}}run"):
            walk(run)

        return "".join(text_parts).strip(), tables, pictures, equations, notes

    # -- tables ---------------------------------------------------------------

    def _add_table(
        self, tbl: etree._Element, doc: DoclingDocument, parent: Optional[NodeItem]
    ) -> None:
        rows = tbl.findall(f"{{{_HP}}}tr")
        if not rows:
            return

        cells: list[TableCell] = []
        num_rows = int(tbl.get("rowCnt") or len(rows))
        num_cols = int(tbl.get("colCnt") or 0)

        for tr in rows:
            for tc in tr.findall(f"{{{_HP}}}tc"):
                addr = tc.find(f"{{{_HP}}}cellAddr")
                span = tc.find(f"{{{_HP}}}cellSpan")
                if addr is None:
                    continue
                row = int(addr.get("rowAddr") or 0)
                col = int(addr.get("colAddr") or 0)
                row_span = int(span.get("rowSpan") or 1) if span is not None else 1
                col_span = int(span.get("colSpan") or 1) if span is not None else 1
                cells.append(
                    TableCell(
                        text=self._cell_text(tc),
                        row_span=row_span,
                        col_span=col_span,
                        start_row_offset_idx=row,
                        end_row_offset_idx=row + row_span,
                        start_col_offset_idx=col,
                        end_col_offset_idx=col + col_span,
                        column_header=tc.get("header") == "1",
                    )
                )
                num_rows = max(num_rows, row + row_span)
                num_cols = max(num_cols, col + col_span)

        data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells)
        doc.add_table(data=data, parent=parent)

    def _cell_text(self, tc: etree._Element) -> str:
        texts: list[str] = []
        for paragraph in tc.findall(f".//{{{_HP}}}p"):
            merged, *_ = self._collect_run_content(paragraph)
            if merged:
                texts.append(merged)
        return " ".join(texts)

    # -- pictures / equations / notes -----------------------------------------

    def _add_picture(
        self, pic: etree._Element, doc: DoclingDocument, parent: Optional[NodeItem]
    ) -> None:
        image_ref: Optional[ImageRef] = None
        img = pic.find(f"{{{_HC}}}img")
        if img is not None:
            bin_ref = img.get("binaryItemIDRef")
            entry = self._bin_items.get(bin_ref or "")
            if entry is not None:
                data = self._entries.get(entry[0])
                if data is not None:
                    image_ref = self._decode_image(data)

        # Pure vector shapes carry no embedded binary; they still emit a picture
        # placeholder so the anchor is not silently dropped.
        doc.add_picture(image=image_ref, parent=parent)

    @staticmethod
    def _decode_image(data: bytes) -> Optional[ImageRef]:
        try:
            with Image.open(BytesIO(data)) as pil_image:
                pil_image.load()
                return ImageRef.from_pil(image=pil_image.convert("RGB"), dpi=72)
        except (UnidentifiedImageError, OSError, ValueError):
            # Hancom also embeds vector formats (EMF/WMF) Pillow cannot decode;
            # fall back to a picture without a raster payload.
            return None

    def _add_formula(
        self, equation: etree._Element, doc: DoclingDocument, parent: Optional[NodeItem]
    ) -> None:
        script = equation.find(f"{{{_HP}}}script")
        text = "".join(script.itertext()).strip() if script is not None else ""
        if text:
            # OWPML stores the Hancom equation script verbatim; v1 emits it as-is
            # without converting to LaTeX.
            doc.add_formula(text=text, parent=parent)

    def _add_note(
        self, note: etree._Element, doc: DoclingDocument, parent: Optional[NodeItem]
    ) -> None:
        texts: list[str] = []
        for paragraph in note.findall(f".//{{{_HP}}}p"):
            merged, *_ = self._collect_run_content(paragraph)
            if merged:
                texts.append(merged)
        text = " ".join(texts)
        if text:
            doc.add_text(label=DocItemLabel.FOOTNOTE, text=text, parent=parent)


def _is_section_entry(name: str) -> bool:
    return (
        name.startswith("Contents/section")
        and name.endswith(".xml")
        and name[len("Contents/section") : -len(".xml")].isdigit()
    )


def _section_sort_key(name: str) -> int:
    return int(name[len("Contents/section") : -len(".xml")])
