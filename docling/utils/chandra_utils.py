# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Parse Chandra's HTML dialect, retaining layout provenance and HTML structure."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DescriptionMetaField,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    Formatting,
    ImageRef,
    ListItem,
    MoleculeMetaField,
    NodeItem,
    PictureItem,
    PictureMeta,
    ProvenanceItem,
    RichTableCell,
    Script,
    SectionHeaderItem,
    Size,
    TableCell,
    TableData,
    TableItem,
)
from PIL import Image as PILImage
from pydantic import AnyUrl, ValidationError

from docling.utils.code_language import detect_code_language

_log = logging.getLogger(__name__)

_LABEL_MAP = {
    "Title": DocItemLabel.TITLE,
    "Section-Header": DocItemLabel.SECTION_HEADER,
    "Caption": DocItemLabel.CAPTION,
    "Footnote": DocItemLabel.FOOTNOTE,
    "Page-Header": DocItemLabel.PAGE_HEADER,
    "Page-Footer": DocItemLabel.PAGE_FOOTER,
    "Equation-Block": DocItemLabel.FORMULA,
    "Code-Block": DocItemLabel.CODE,
    "Bibliography": DocItemLabel.REFERENCE,
}
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
_BLOCK_TAGS = {
    "div",
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "table",
    "ul",
    "ol",
    "li",
    "pre",
    "figure",
    "form",
    "caption",
    "figcaption",
    "hr",
}
_FORMAT_TAGS = {
    "b": "bold",
    "strong": "bold",
    "i": "italic",
    "em": "italic",
    "u": "underline",
    "del": "strikethrough",
    "s": "strikethrough",
}


@dataclass
class _Element:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[_Element | str] = field(default_factory=list)

    def elements(self) -> Iterator[_Element]:
        for child in self.children:
            if isinstance(child, _Element):
                yield child
                yield from child.elements()

    def text(self) -> str:
        if self.tag == "br":
            return "\n"
        if self.tag == "input":
            if self.attrs.get("type", "").lower() in {"checkbox", "radio"}:
                return "☑" if "checked" in self.attrs else "☐"
            return self.attrs.get("value", "")
        return "".join(
            child
            if isinstance(child, str)
            else child.text() + ("\n" if child.tag in _BLOCK_TAGS else "")
            for child in self.children
        )


class _HTMLTreeParser(HTMLParser):
    """A tree for the generated dialect; no browser, resource fetching, or CSS inference."""

    def __init__(self, content: str):
        super().__init__(convert_charrefs=True)
        self.root = _Element("root")
        self.stack = [self.root]
        self.feed(content)
        if self.rawdata.strip():
            _log.warning(
                "Chandra HTML ends with an incomplete tag; preserving preceding content"
            )
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Accept omitted closing tags for adjacent cells, rows, and list items.
        closes = {
            "td": ({"td", "th"}, {"tr", "table"}),
            "th": ({"td", "th"}, {"tr", "table"}),
            "tr": ({"tr"}, {"table"}),
            "li": ({"li"}, {"ul", "ol"}),
            "p": ({"p"}, _BLOCK_TAGS - {"p"}),
        }
        if tag in closes:
            targets, boundaries = closes[tag]
            for index in range(len(self.stack) - 1, 0, -1):
                if self.stack[index].tag in targets:
                    del self.stack[index:]
                    break
                if self.stack[index].tag in boundaries:
                    break
        node = _Element(tag, {key: value or "" for key, value in attrs})
        self.stack[-1].children.append(node)
        if tag not in _VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)


def _span(value: str, maximum: int, default: int = 1) -> int:
    try:
        return min(maximum, max(1, int(value)))
    except ValueError:
        _log.warning("Invalid Chandra table span %r; using %s", value, default)
        return default


def _table_cells(table: _Element) -> list[tuple[TableCell, _Element]]:
    rows: list[tuple[_Element, bool, int | None]] = []
    for child in table.children:
        if isinstance(child, _Element):
            if child.tag == "tr":
                rows.append((child, False, None))
            elif child.tag in {"thead", "tbody", "tfoot"}:
                group_rows = [
                    row
                    for row in child.children
                    if isinstance(row, _Element) and row.tag == "tr"
                ]
                group_end = len(rows) + len(group_rows)
                rows.extend(
                    (row, child.tag == "thead", group_end) for row in group_rows
                )
    occupied: dict[int, int] = {}
    result = []
    for row_index, (row, in_header, group_end) in enumerate(rows):
        col = 0
        for node in row.children:
            if not isinstance(node, _Element) or node.tag not in {"td", "th"}:
                continue
            colspan = _span(node.attrs.get("colspan", "1"), 1000)
            remaining_rows = (group_end or len(rows)) - row_index
            rowspan = (
                remaining_rows
                if node.attrs.get("rowspan") == "0"
                else _span(node.attrs.get("rowspan", "1"), remaining_rows)
            )
            while any(
                occupied.get(c, 0) > row_index for c in range(col, col + colspan)
            ):
                col += 1
            for c in range(col, col + colspan):
                occupied[c] = row_index + rowspan
            result.append(
                (
                    TableCell(
                        text=" ".join(node.text().split()),
                        row_span=rowspan,
                        col_span=colspan,
                        start_row_offset_idx=row_index,
                        end_row_offset_idx=row_index + rowspan,
                        start_col_offset_idx=col,
                        end_col_offset_idx=col + colspan,
                        column_header=node.attrs.get("scope") == "col"
                        or (node.tag == "th" and (in_header or row_index == 0)),
                        row_header=node.attrs.get("scope") == "row"
                        or (
                            node.tag == "th"
                            and col == 0
                            and row_index > 0
                            and not in_header
                        ),
                    ),
                    node,
                )
            )
            col += colspan
    return result


def _table_data(cells: list[TableCell]) -> TableData:
    return TableData(
        num_rows=max((cell.end_row_offset_idx for cell in cells), default=0),
        num_cols=max((cell.end_col_offset_idx for cell in cells), default=0),
        table_cells=cells,
    )


def _parse_table_html(html_content: str) -> TableData:
    """Plain table conversion also used by the DOTS parser."""
    root = _HTMLTreeParser(html_content).root
    table = next((node for node in root.elements() if node.tag == "table"), None)
    return (
        _table_data([cell for cell, _ in _table_cells(table)])
        if table is not None
        else _table_data([])
    )


@dataclass
class _Run:
    text: str
    label: DocItemLabel = DocItemLabel.TEXT
    formatting: Formatting = field(default_factory=Formatting)
    hyperlink: AnyUrl | Path | None = None


def _inline_runs(
    node: _Element | str,
    formatting: Formatting | None = None,
    hyperlink: AnyUrl | Path | None = None,
) -> Iterator[_Run]:
    formatting = formatting or Formatting()
    if isinstance(node, str):
        yield _Run(
            re.sub(r"\s+", " ", node), formatting=formatting, hyperlink=hyperlink
        )
        return
    formatting = formatting.model_copy()
    if node.tag in _FORMAT_TAGS:
        setattr(formatting, _FORMAT_TAGS[node.tag], True)
    elif node.tag in {"sup", "sub"}:
        formatting.script = Script.SUPER if node.tag == "sup" else Script.SUB
    elif node.tag == "a" and node.attrs.get("href"):
        href = node.attrs["href"]
        try:
            hyperlink = AnyUrl(href) if ":" in href else Path(href)
        except ValidationError:
            _log.warning("Ignoring invalid Chandra hyperlink %r", href)
    label = {"math": DocItemLabel.FORMULA, "code": DocItemLabel.CODE}.get(node.tag)
    if node.tag == "input":
        kind = node.attrs.get("type", "text").lower()
        if kind == "hidden":
            return
        label = (
            (
                DocItemLabel.CHECKBOX_SELECTED
                if "checked" in node.attrs
                else DocItemLabel.CHECKBOX_UNSELECTED
            )
            if kind in {"checkbox", "radio"}
            else DocItemLabel.FIELD_VALUE
        )
        yield _Run(node.text(), label, formatting, hyperlink)
    elif label is not None:
        yield _Run(node.text().strip(), label, formatting, hyperlink)
    elif node.tag == "br":
        yield _Run("\n", formatting=formatting, hyperlink=hyperlink)
    elif node.tag not in {"script", "style", "head"}:
        for child in node.children:
            yield from _inline_runs(child, formatting, hyperlink)


class _ChandraDocumentBuilder:
    def __init__(self, doc: DoclingDocument, size: Size, page_no: int):
        self.doc = doc
        self.size = size
        self.page_no = page_no

    def _provenance(self, node: _Element) -> ProvenanceItem | None:
        raw = node.attrs.get("data-bbox", "")
        try:
            x0, y0, x1, y1 = (float(c) for c in raw.split())
        except ValueError:
            _log.warning(
                "Missing or invalid Chandra bbox %r; preserving content without coordinates",
                raw,
            )
            return None
        if not (0 <= x0 <= x1 <= 1000 and 0 <= y0 <= y1 <= 1000):
            _log.warning(
                "Invalid Chandra bbox %r; preserving content without coordinates", raw
            )
            return None
        return ProvenanceItem(
            page_no=self.page_no,
            charspan=(0, 0),
            bbox=BoundingBox(
                l=x0 * self.size.width / 1000,
                t=y0 * self.size.height / 1000,
                r=x1 * self.size.width / 1000,
                b=y1 * self.size.height / 1000,
                coord_origin=CoordOrigin.TOPLEFT,
            ),
        )

    @staticmethod
    def _text_prov(prov: ProvenanceItem | None, text: str) -> ProvenanceItem | None:
        return (
            prov.model_copy(update={"charspan": (0, len(text))})
            if prov is not None
            else None
        )

    def _emit_runs(
        self,
        runs: list[_Run],
        parent: NodeItem | None,
        prov: ProvenanceItem | None,
        label: DocItemLabel,
    ) -> None:
        merged: list[_Run] = []
        for run in runs:
            if (
                merged
                and run.label == merged[-1].label == DocItemLabel.TEXT
                and run.formatting == merged[-1].formatting
                and run.hyperlink == merged[-1].hyperlink
            ):
                merged[-1].text += run.text
            else:
                merged.append(run)
        merged = [
            run
            for run in merged
            if run.text.strip() or run.label == DocItemLabel.FIELD_VALUE
        ]
        if not merged:
            return
        layer = (
            ContentLayer.FURNITURE
            if label in {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}
            else ContentLayer.BODY
        )
        if len(merged) > 1:
            if label != DocItemLabel.TEXT:
                parent = self.doc.add_text(
                    label=label, text="", parent=parent, prov=prov, content_layer=layer
                )
            parent = self.doc.add_inline_group(parent=parent, content_layer=layer)
        for run in merged:
            run_label = (
                label
                if run.label == DocItemLabel.TEXT and len(merged) == 1
                else run.label
            )
            text = run.text.strip()
            if (
                isinstance(parent, ListItem)
                and not parent.children
                and not parent.text
                and len(merged) == 1
                and run_label == DocItemLabel.TEXT
            ):
                parent.text = parent.orig = text
                parent.formatting = (
                    run.formatting if run.formatting != Formatting() else None
                )
                parent.hyperlink = run.hyperlink
                text_prov = self._text_prov(prov, text)
                parent.prov = [text_prov] if text_prov is not None else []
                continue
            item = self.doc.add_text(
                label=run_label,
                text=text,
                parent=parent,
                prov=self._text_prov(prov, text),
                formatting=run.formatting if run.formatting != Formatting() else None,
                hyperlink=run.hyperlink,
                content_layer=layer,
            )
            if run_label == DocItemLabel.FIELD_VALUE:
                item.kind = "fillable"

    def walk(
        self,
        children: list[_Element | str],
        parent: NodeItem | None = None,
        prov: ProvenanceItem | None = None,
        label: DocItemLabel = DocItemLabel.TEXT,
    ) -> None:
        runs: list[_Run] = []
        for node in children:
            if isinstance(node, str) or (
                node.tag not in _BLOCK_TAGS | {"img", "chem"}
                and "data-label" not in node.attrs
                and not (node.tag == "math" and node.attrs.get("display") == "block")
                and not any(child.tag in _BLOCK_TAGS for child in node.elements())
            ):
                runs.extend(_inline_runs(node))
                continue
            self._emit_runs(runs, parent, prov, label)
            runs = []
            self._block(node, parent, prov, label)
        self._emit_runs(runs, parent, prov, label)

    def _block(
        self,
        node: _Element,
        parent: NodeItem | None,
        prov: ProvenanceItem | None,
        label: DocItemLabel,
    ) -> None:
        layout_label = node.attrs.get("data-label")
        if "data-bbox" in node.attrs or layout_label is not None:
            prov = self._provenance(node)
            label = _LABEL_MAP.get(layout_label or "", DocItemLabel.TEXT)
        if layout_label in {
            "Figure",
            "Image",
            "Diagram",
            "Chemical-Block",
        } or node.tag in {"figure", "img", "chem"}:
            self._picture(node, parent, prov)
        elif node.tag == "table":
            self._table(node, parent, prov)
        elif node.tag in {"ul", "ol"} or (
            layout_label == "List-Group"
            and not any(child.tag in {"ul", "ol"} for child in node.elements())
        ):
            self._list(node, parent, prov)
        elif node.tag == "pre" or (
            layout_label == "Code-Block"
            and not any(child.tag == "pre" for child in node.elements())
        ):
            text = node.text().strip("\n\r")
            self.doc.add_code(
                text=text,
                parent=parent,
                prov=self._text_prov(prov, text),
                code_language=detect_code_language(text, hint=node.attrs.get("class")),
            )
        elif node.tag == "math":
            text = node.text().strip()
            self.doc.add_formula(
                text=text, parent=parent, prov=self._text_prov(prov, text)
            )
        elif layout_label == "Form" or node.tag == "form":
            region = self.doc.add_field_region(parent=parent, prov=prov)
            self.walk(node.children, region, prov)
        elif node.tag in {"caption", "figcaption"} or layout_label in {
            "Caption",
            "Footnote",
        }:
            label = (
                DocItemLabel.FOOTNOTE
                if layout_label == "Footnote"
                else DocItemLabel.CAPTION
            )
            before = len(self.doc.texts)
            self.walk(node.children, parent, prov, label)
            if isinstance(parent, (PictureItem, TableItem)):
                refs = (
                    parent.footnotes
                    if label == DocItemLabel.FOOTNOTE
                    else parent.captions
                )
                refs.extend(
                    item.get_ref()
                    for item in self.doc.texts[before:]
                    if item.label == label
                )
        elif node.tag not in {"script", "style", "head", "hr"}:
            if re.fullmatch(r"h[1-6]", node.tag) and label == DocItemLabel.TEXT:
                label = DocItemLabel.SECTION_HEADER
            before = len(self.doc.texts)
            self.walk(node.children, parent, prov, label)
            if re.fullmatch(r"h[1-6]", node.tag):
                for item in self.doc.texts[before:]:
                    if isinstance(item, SectionHeaderItem):
                        item.level = int(node.tag[1])

    def _list(
        self, node: _Element, parent: NodeItem | None, prov: ProvenanceItem | None
    ) -> None:
        group = self.doc.add_list_group(parent=parent)
        ordered = node.tag == "ol"
        number = _span(node.attrs.get("start", "1"), 2**31 - 1)
        children = node.children
        if node.tag not in {"ul", "ol"}:
            children = []
            pending = _Element("li")
            for child in node.children:
                if isinstance(child, _Element) and child.tag in {"li", "p", "br"}:
                    if pending.children:
                        children.append(pending)
                        pending = _Element("li")
                    if child.tag != "br":
                        children.append(child)
                else:
                    pending.children.append(child)
            if pending.text().strip() or any(True for _ in pending.elements()):
                children.append(pending)
        for child in children:
            if isinstance(child, str) and not child.strip():
                continue
            if isinstance(child, _Element) and child.tag in {"ul", "ol"}:
                self._list(child, group, prov)
                continue
            item = self.doc.add_list_item(
                text="",
                enumerated=ordered,
                marker=f"{number}." if ordered else "",
                parent=group,
                prov=prov,
            )
            contents = (
                child.children
                if isinstance(child, _Element) and child.tag in {"li", "p"}
                else [child]
            )
            self.walk(contents, item, prov)
            number += 1

    def _table(
        self, node: _Element, parent: NodeItem | None, prov: ProvenanceItem | None
    ) -> None:
        parsed = _table_cells(node)
        if not parsed:
            _log.warning("Chandra table contains no cells; preserving its content")
            self.walk(node.children, parent, prov)
            return
        table = self.doc.add_table(data=_table_data([]), parent=parent, prov=prov)
        cells: list[TableCell] = []
        for cell, cell_node in parsed:
            if any(child.tag != "span" for child in cell_node.elements()):
                group = self.doc.add_group(parent=table)
                self.walk(cell_node.children, group, prov)
                if group.children:
                    cell = RichTableCell(**cell.model_dump(), ref=group.get_ref())
            cells.append(cell)
        table.data = _table_data(cells)
        for child in node.children:
            if isinstance(child, _Element) and (
                child.tag in {"caption", "figcaption"}
                or child.attrs.get("data-label") in {"Caption", "Footnote"}
            ):
                self._block(child, table, prov, DocItemLabel.CAPTION)

    def _picture(
        self, node: _Element, parent: NodeItem | None, prov: ProvenanceItem | None
    ) -> None:
        picture = self.doc.add_picture(parent=parent, prov=prov)
        images = (
            [node]
            if node.tag == "img"
            else [child for child in node.elements() if child.tag == "img"]
        )
        descriptions = list(
            dict.fromkeys(
                image.attrs["alt"] for image in images if image.attrs.get("alt")
            )
        )
        picture.meta = PictureMeta(
            description=DescriptionMetaField(
                text="\n".join(descriptions), created_by="chandra"
            )
            if descriptions
            else None
        )
        chemicals = (
            [node]
            if node.tag == "chem"
            else [child for child in node.elements() if child.tag == "chem"]
        )
        if len(chemicals) == 1:
            picture.meta.molecule = MoleculeMetaField(
                smi=chemicals[0].text().strip(), created_by="chandra"
            )
        if node.tag == "chem":
            return

        # The layout bbox describes the whole picture. Its img tags provide
        # descriptions, not separately located images. Keep structured children.
        for element in (node, *node.elements()):
            element.children[:] = [
                child
                for child in element.children
                if isinstance(child, str)
                or (
                    child.tag != "img"
                    and not (child.tag == "chem" and len(chemicals) == 1)
                )
            ]
        self.walk(node.children, picture, prov)


def parse_chandra_html(
    content: str,
    original_page_size: Size,
    page_no: int,
    filename: str = "file",
    page_image: PILImage.Image | None = None,
) -> DoclingDocument:
    """Map Chandra layout blocks and their HTML contents to document primitives.

    Inner items inherit the block bbox, not invented element coordinates. Bare
    HTML is recovered without provenance; non-HTML responses raise ValueError.
    Caption/footnote links require explicit nesting. Separate layout blocks are
    neither associated nor merged by proximity.
    """
    doc = DoclingDocument(
        name=Path(filename).stem,
        origin=DocumentOrigin(filename=filename, mimetype="text/html", binary_hash=0),
    )
    dpi = (
        max(1, round(72 * page_image.width / original_page_size.width))
        if page_image is not None
        else 72
    )
    doc.add_page(
        page_no=page_no,
        size=original_page_size,
        image=ImageRef.from_pil(image=page_image, dpi=dpi)
        if page_image is not None
        else None,
    )
    if not content.strip():
        return doc
    root = _HTMLTreeParser(content).root
    builder = _ChandraDocumentBuilder(doc, original_page_size, page_no)
    blocks = [
        node
        for node in root.elements()
        if "data-label" in node.attrs or "data-bbox" in node.attrs
    ]
    if blocks:
        # Walk each outer layout block once, ignoring deployment reasoning prose.
        nested = {id(child) for block in blocks for child in block.elements()}
        for block in blocks:
            if id(block) not in nested:
                builder._block(block, None, None, DocItemLabel.TEXT)
    else:
        html_nodes = [
            child
            for child in root.children
            if isinstance(child, _Element)
            and child.tag not in {"script", "style", "head"}
        ]
        if not html_nodes:
            raise ValueError(
                "Chandra response contains no HTML transcription or layout blocks"
            )
        _log.warning(
            "Chandra response has no layout blocks; recovering HTML without coordinates"
        )
        builder.walk(list(html_nodes))
    if not (doc.texts or doc.tables or doc.pictures or doc.field_regions) and not any(
        block.attrs.get("data-label") == "Blank-Page" for block in blocks
    ):
        raise ValueError("Chandra HTML contains no document content")
    return doc
