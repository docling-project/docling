# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Final, Optional, Union

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    Formatting,
    GroupItem,
    GroupLabel,
    ImageRef,
    ListItem,
    TableCell,
    TableData,
    TextItem,
)

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.utils.image_resource_loader import ImageResourceLoader
from docling.datamodel.backend_options import AsciiDocBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError

_log = logging.getLogger(__name__)

# Cell format specifier that may precede a "|" delimiter, e.g. "^.^h" in
# "^.^h|Header": span (3*, 2+), alignment (<, ^, >, .^), style (a/d/e/h/l/m/s).
_CELL_SPEC: Final = r"(?:\d+(?:\.\d+)?[*+])*[<^>]?(?:\.[<^>])?[adehlms]?"
_LIST_ITEM_PATTERN: Final = r"^(\s*)(\*|-|\.+|\d+\.|\w+\.)\s+(.*)"


@dataclass(frozen=True)
class _LiteralBlock:
    text: str


class AsciiDocBackend(DeclarativeDocumentBackend):
    def __init__(
        self,
        in_doc: "InputDocument",
        path_or_stream: Union[BytesIO, Path],
        options: Optional[AsciiDocBackendOptions] = None,
    ):
        if options is None:
            options = AsciiDocBackendOptions()
        super().__init__(in_doc, path_or_stream, options)

        self.path_or_stream = path_or_stream
        self.options: AsciiDocBackendOptions
        self._image_loader = ImageResourceLoader(
            enable_local_fetch=options.enable_local_fetch,
            enable_remote_fetch=options.enable_remote_fetch,
            max_image_data_base64_bytes=options.max_image_data_base64_bytes,
        )

        # utf-8-sig drops a leading BOM. Kept, it prefixes the first line, so a
        # document title ("= Title") is no longer recognized as one and the BOM
        # reaches the output. Equivalent to utf-8 when no BOM is present.
        try:
            if isinstance(self.path_or_stream, BytesIO):
                text_stream = self.path_or_stream.getvalue().decode("utf-8-sig")
                self.lines = text_stream.split("\n")
            if isinstance(self.path_or_stream, Path):
                with open(self.path_or_stream, encoding="utf-8-sig") as f:
                    self.lines = f.readlines()
            self.valid = True

        except Exception as e:
            raise DocumentLoadError(
                f"Could not initialize AsciiDoc backend for file with hash {self.document_hash}."
            ) from e
        return

    def is_valid(self) -> bool:
        return self.valid

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    def unload(self):
        return

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.ASCIIDOC}

    def convert(self) -> DoclingDocument:
        """
        Parses the ASCII into a structured document model.
        """

        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="text/asciidoc",
            binary_hash=self.document_hash,
        )

        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)

        doc = self._parse(doc)

        return doc

    def _parse(self, doc: DoclingDocument):
        """Orchestrate parsing and populate `doc` from the source lines.

        Handles titles, section headers, text paragraphs, lists, tables,
        pictures, literal (code) blocks, and block titles.

        Block titles (AsciiDoc lines that start with `.` immediately
        followed by text, e.g. `.Procedure`) are treated as follows:

        * FloatingItem targets (picture, table, code block): the block
          title is created as a `CAPTION`-labelled `TextItem` and
          attached to the floating item via its `caption` parameter, so it
          participates in the standard caption relationship.
        * All other targets (lists, paragraphs, ...): the block title is
          emitted as a bold `PARAGRAPH` `TextItem` inserted
          immediately before the element it precedes.  This preserves
          reading order while acknowledging that non-floating items (e.g.
          `GroupItem`) have no caption slot.
        """

        in_list = False
        in_table = False

        text_data: list[str] = []
        table_data: list[str] = []
        caption_data: list[str] = []
        last_list_item: ListItem | None = None
        list_continuation = False

        # parents: dict[int, Union[DocItem, GroupItem, None]] = {}
        parents: dict[int, Union[GroupItem, None]] = {}
        # indents: dict[int, Union[DocItem, GroupItem, None]] = {}
        indents: dict[int, Union[GroupItem, None]] = {}

        for i in range(10):
            parents[i] = None
            indents[i] = None

        for block in self._iter_blocks(self.lines):
            # line = line.strip()
            if isinstance(block, _LiteralBlock):
                in_list, last_list_item, list_continuation = self._close_list_if_needed(
                    line="<literal-block>",
                    in_list=in_list,
                    parents=parents,
                    last_list_item=last_list_item,
                    list_continuation=list_continuation,
                    is_continuation_block=True,
                )
                text_data = self._flush_text_data(
                    doc=doc,
                    text_data=text_data,
                    parent=self._get_current_parent(parents),
                )
                caption: Optional[TextItem] = None
                if caption_data:
                    caption = doc.add_text(
                        text=" ".join(caption_data),
                        label=DocItemLabel.CAPTION,
                    )
                    caption_data = []
                doc.add_code(
                    text=block.text,
                    caption=caption,
                    parent=(
                        last_list_item if in_list else self._get_current_parent(parents)
                    ),
                )
                list_continuation = False
                continue

            line = block
            stripped_line = line.strip()
            in_list, last_list_item, list_continuation = self._close_list_if_needed(
                line=line,
                in_list=in_list,
                parents=parents,
                last_list_item=last_list_item,
                list_continuation=list_continuation,
                is_continuation_block=False,
            )

            # Title
            if self._is_title(line):
                item = self._parse_title(line)
                level = item["level"]

                parents[level] = doc.add_text(
                    text=item["text"], label=DocItemLabel.TITLE
                )

            # Section headers
            elif self._is_section_header(line):
                item = self._parse_section_header(line)
                level = item["level"]

                ancestor = next(
                    (parents[k] for k in range(level - 1, -1, -1) if parents[k]),
                    None,
                )
                parents[level] = doc.add_heading(
                    text=item["text"], level=item["level"], parent=ancestor
                )
                for k, v in parents.items():
                    if k > level:
                        parents[k] = None

            # Lists
            elif self._is_list_item(line):
                _log.debug(f"line: {line}")
                item = self._parse_list_item(line)
                _log.debug(f"parsed list-item: {item}")

                level = self._get_current_level(parents)

                if not in_list:
                    in_list = True
                    # GroupItem has no caption slot; emit the pending block
                    # title as a bold paragraph immediately before the list.
                    if caption_data:
                        doc.add_text(
                            text=" ".join(caption_data),
                            label=DocItemLabel.PARAGRAPH,
                            parent=parents[level],
                            formatting=Formatting(bold=True),
                        )
                        caption_data = []

                    parents[level + 1] = doc.add_group(
                        parent=parents[level], name="list", label=GroupLabel.LIST
                    )
                    indents[level + 1] = item["indent"]

                elif in_list and item["indent"] > indents[level]:
                    parents[level + 1] = doc.add_group(
                        parent=parents[level], name="list", label=GroupLabel.LIST
                    )
                    indents[level + 1] = item["indent"]

                elif in_list and item["indent"] < indents[level]:
                    # print(item["indent"], " => ", indents[level])
                    while level > 0 and item["indent"] < indents[level]:
                        # print(item["indent"], " => ", indents[level])
                        parents[level] = None
                        indents[level] = None
                        level -= 1

                last_list_item = doc.add_list_item(
                    item["text"],
                    enumerated=item["numbered"],
                    marker=(item["marker"] if item["marker"][:-1].isdigit() else None),
                    parent=self._get_current_parent(parents),
                )
                list_continuation = False

            elif in_list and stripped_line == "+":
                list_continuation = True
                continue

            # Tables
            elif line.strip() == "|===" and not in_table:  # start of table
                in_table = True

            elif self._is_table_line(line):  # within a table
                in_table = True
                table_data.append(self._parse_table_line(line))

            elif in_table and (
                (not self._is_table_line(line)) or line.strip() == "|==="
            ):  # end of table
                caption = None
                if len(caption_data) > 0:
                    caption = doc.add_text(
                        text=" ".join(caption_data), label=DocItemLabel.CAPTION
                    )

                caption_data = []

                data = self._populate_table_as_grid(table_data)
                doc.add_table(
                    data=data, parent=self._get_current_parent(parents), caption=caption
                )

                in_table = False
                table_data = []

            # Picture
            elif self._is_picture(line):
                caption = None
                if len(caption_data) > 0:
                    caption = doc.add_text(
                        text=" ".join(caption_data), label=DocItemLabel.CAPTION
                    )

                caption_data = []

                item = self._parse_picture(line)

                image: Optional[ImageRef] = None
                if "uri" in item and self.options.fetch_images:
                    base_path = (
                        str(self.options.source_uri)
                        if self.options.source_uri is not None
                        else None
                    )
                    image = self._image_loader.load_image_ref(item["uri"], base_path)
                doc.add_picture(
                    image=image,
                    caption=caption,
                    parent=last_list_item
                    if in_list
                    else self._get_current_parent(parents),
                )
                list_continuation = False

            # Caption
            elif self._is_caption(line) and len(caption_data) == 0:
                item = self._parse_caption(line)
                caption_data.append(item["text"])

            elif (
                len(line.strip()) > 0 and len(caption_data) > 0
            ):  # allow multiline captions
                item = self._parse_text(line)
                caption_data.append(item["text"])

            # Plain text
            elif len(line.strip()) == 0 and len(text_data) > 0:
                doc.add_text(
                    text=" ".join(text_data),
                    label=DocItemLabel.PARAGRAPH,
                    parent=self._get_current_parent(parents),
                )
                text_data = []

            elif len(line.strip()) > 0:  # allow multiline texts
                item = self._parse_text(line)
                text_data.append(item["text"])

        if len(text_data) > 0:
            doc.add_text(
                text=" ".join(text_data),
                label=DocItemLabel.PARAGRAPH,
                parent=self._get_current_parent(parents),
            )
            text_data = []

        if in_table and len(table_data) > 0:
            data = self._populate_table_as_grid(table_data)
            doc.add_table(data=data, parent=self._get_current_parent(parents))

            in_table = False
            table_data = []

        return doc

    @staticmethod
    def _get_current_level(parents):
        for k, v in parents.items():
            if v is None and k > 0:
                return k - 1

        return 0

    @staticmethod
    def _get_current_parent(parents):
        for k, v in parents.items():
            if v is None and k > 0:
                return parents[k - 1]

        return None

    @staticmethod
    def _iter_blocks(lines: list[str]) -> Iterator[str | _LiteralBlock]:
        literal_data: list[str] | None = None

        for line in lines:
            if line.strip() == "....":
                if literal_data is None:
                    literal_data = []
                else:
                    yield _LiteralBlock(text="\n".join(literal_data))
                    literal_data = None
                continue

            if literal_data is None:
                yield line
            else:
                literal_data.append(line.rstrip("\r\n"))

        if literal_data is not None:
            yield _LiteralBlock(text="\n".join(literal_data))

    @classmethod
    def _close_list_if_needed(
        cls,
        *,
        line: str,
        in_list: bool,
        parents: dict[int, GroupItem | None],
        last_list_item: ListItem | None,
        list_continuation: bool,
        is_continuation_block: bool,
    ) -> tuple[bool, ListItem | None, bool]:
        if (
            not in_list
            or cls._is_list_item(line)
            or line.strip() in {"", "+"}
            or (list_continuation and (is_continuation_block or cls._is_picture(line)))
        ):
            return in_list, last_list_item, list_continuation

        level = cls._get_current_level(parents)
        parents[level] = None
        return False, None, False

    @staticmethod
    def _flush_text_data(
        *,
        doc: DoclingDocument,
        text_data: list[str],
        parent: GroupItem | None,
    ) -> list[str]:
        if len(text_data) > 0:
            doc.add_text(
                text=" ".join(text_data),
                label=DocItemLabel.PARAGRAPH,
                parent=parent,
            )
        return []

    #   =========   Title
    @staticmethod
    def _is_title(line):
        return re.match(r"^= ", line)

    @staticmethod
    def _parse_title(line):
        return {"type": "title", "text": line[2:].strip(), "level": 0}

    #   =========   Section headers
    @staticmethod
    def _is_section_header(line):
        return re.match(r"^==+\s+", line)

    @staticmethod
    def _parse_section_header(line):
        match = re.match(r"^(=+)\s+(.*)", line)

        marker = match.group(1)  # The list marker (e.g., "*", "-", "1.")
        text = match.group(2)  # The actual text of the list item

        header_level = marker.count("=")  # number of '=' represents level
        return {
            "type": "header",
            "level": header_level - 1,
            "text": text.strip(),
        }

    #   =========   Lists
    @staticmethod
    def _is_list_item(line):
        return re.match(_LIST_ITEM_PATTERN, line)

    @staticmethod
    def _parse_list_item(line):
        """Extract the item marker (number or bullet symbol) and the text of the item."""

        match = re.match(_LIST_ITEM_PATTERN, line)
        if match:
            indent = match.group(1)
            marker = match.group(2)  # The list marker (e.g., "*", "-", "1.")
            text = match.group(3)  # The actual text of the list item
            indent_width = len(indent)
            if marker.startswith("."):
                indent_width += len(marker) - 1

            if marker == "*" or marker == "-":
                return {
                    "type": "list_item",
                    "marker": marker,
                    "text": text.strip(),
                    "numbered": False,
                    "indent": indent_width,
                }
            else:
                return {
                    "type": "list_item",
                    "marker": marker,
                    "text": text.strip(),
                    "numbered": True,
                    "indent": indent_width,
                }
        else:
            # Fallback if no match
            return {
                "type": "list_item",
                "marker": "-",
                "text": line,
                "numbered": False,
                "indent": 0,
            }

    #   =========   Tables
    @staticmethod
    def _is_table_line(line):
        return re.match(rf"^{_CELL_SPEC}\|.*\|", line)

    @staticmethod
    def _parse_table_line(line):
        # Drop cell specifiers glued to a "|" (e.g. "^.^h"); anchored to
        # whitespace so content ending in a style letter (e.g. "Eth") survives.
        line = re.sub(rf"(^|\s){_CELL_SPEC}(?=\|)", r"\1", line)
        # Split by "|" and remove the leading empty string from the first "|"
        cells = line.split("|")[1:]
        # Strip whitespace from each cell (empty cells become empty strings)
        return [cell.strip() for cell in cells]

    @staticmethod
    def _populate_table_as_grid(table_data):
        num_rows = len(table_data)

        # Adjust the table data into a grid format
        num_cols = max((len(row) for row in table_data), default=0)

        data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=[])
        for row_idx, row in enumerate(table_data):
            # Pad rows with empty strings to match column count
            # grid.append(row + [''] * (max_cols - len(row)))

            for col_idx, text in enumerate(row):
                row_span = 1
                col_span = 1

                cell = TableCell(
                    text=text,
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=row_idx,
                    end_row_offset_idx=row_idx + row_span,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + col_span,
                    column_header=row_idx == 0,
                    row_header=False,
                )
                data.table_cells.append(cell)

        return data

    #   =========   Pictures
    @staticmethod
    def _is_picture(line):
        return re.match(r"^image::", line)

    @staticmethod
    def _parse_picture(line):
        """
        Parse an image macro, extracting its path and attributes.
        Syntax: image::path/to/image.png[Alt Text, width=200, height=150, align=center]
        """
        mtch = re.match(r"^image::(.+)\[(.*)\]$", line)
        if mtch:
            picture_path = mtch.group(1).strip()
            attributes = mtch.group(2).split(",")
            picture_info = {"type": "picture", "uri": picture_path}

            # Extract optional attributes (alt text, width, height, alignment)
            if attributes:
                alt_parts = [attributes[0].strip()] if attributes[0] else [""]
                for attr in attributes[1:]:
                    if "=" in attr:
                        key, value = attr.split("=", 1)
                        picture_info[key.strip()] = value.strip()
                    else:
                        alt_parts.append(attr.strip())
                picture_info["alt"] = ", ".join(alt_parts)

            return picture_info

        return {"type": "picture", "uri": line}

    #   =========   Captions
    @staticmethod
    def _is_caption(line):
        return re.match(r"^\.(\S.*)", line)

    @staticmethod
    def _parse_caption(line):
        mtch = re.match(r"^\.(.+)", line)
        if mtch:
            text = mtch.group(1)
            return {"type": "caption", "text": text}

        return {"type": "caption", "text": ""}

    #   =========   Plain text
    @staticmethod
    def _parse_text(line):
        return {"type": "text", "text": line.strip()}
