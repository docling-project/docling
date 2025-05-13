import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Set, Union

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupItem,
    GroupLabel,
    ImageRef,
    Size,
    TableCell,
    TableData,
)

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


class AsciiDocBackend(DeclarativeDocumentBackend):
    def __init__(self, in_doc: InputDocument, path_or_stream: Union[BytesIO, Path]):
        super().__init__(in_doc, path_or_stream)

        self.path_or_stream = path_or_stream

        try:
            if isinstance(self.path_or_stream, BytesIO):
                text_stream = self.path_or_stream.getvalue().decode("utf-8")
                self.lines = text_stream.split("\n")
            if isinstance(self.path_or_stream, Path):
                with open(self.path_or_stream, encoding="utf-8") as f:
                    self.lines = f.readlines()
            self.valid = True

        except Exception as e:
            raise RuntimeError(
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
    def supported_formats(cls) -> Set[InputFormat]:
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

    def _parse(self, doc: DoclingDocument):  # noqa: C901
        """
        Main function that orchestrates the parsing by yielding components:
        title, section headers, text, lists, and tables.
        """

        in_list = False
        in_table = False

        text_data: list[str] = []
        table_data: list[str] = []
        caption_data: list[str] = []

        # parents: dict[int, Union[DocItem, GroupItem, None]] = {}
        parents: dict[int, Union[GroupItem, None]] = {}
        # indents: dict[int, Union[DocItem, GroupItem, None]] = {}
        indents: dict[int, Union[GroupItem, None]] = {}

        for i in range(10):
            parents[i] = None
            indents[i] = None

        for line in self.lines:
            # line = line.strip()

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

                parents[level] = doc.add_heading(
                    text=item["text"], level=item["level"], parent=parents[level - 1]
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
                    while item["indent"] < indents[level]:
                        # print(item["indent"], " => ", indents[level])
                        parents[level] = None
                        indents[level] = None
                        level -= 1

                doc.add_list_item(
                    item["text"], parent=self._get_current_parent(parents)
                )

            elif in_list and not self._is_list_item(line):
                in_list = False

                level = self._get_current_level(parents)
                parents[level] = None

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

                size = None
                if "width" in item and "height" in item:
                    size = Size(width=int(item["width"]), height=int(item["height"]))

                uri = None
                if (
                    "uri" in item
                    and not item["uri"].startswith("http")
                    and item["uri"].startswith("//")
                ):
                    uri = "file:" + item["uri"]
                elif (
                    "uri" in item
                    and not item["uri"].startswith("http")
                    and item["uri"].startswith("/")
                ):
                    uri = "file:/" + item["uri"]
                elif "uri" in item and not item["uri"].startswith("http"):
                    uri = "file://" + item["uri"]

                image = ImageRef(mimetype="image/png", size=size, dpi=70, uri=uri)
                doc.add_picture(image=image, caption=caption)

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

    def _get_current_level(self, parents):
        for k, v in parents.items():
            if v is None and k > 0:
                return k - 1

        return 0

    def _get_current_parent(self, parents):
        for k, v in parents.items():
            if v is None and k > 0:
                return parents[k - 1]

        return None

    #   =========   Title
    def _is_title(self, line):
        return re.match(r"^= ", line)

    def _parse_title(self, line):
        return {"type": "title", "text": line[2:].strip(), "level": 0}

    #   =========   Section headers
    def _is_section_header(self, line):
        return re.match(r"^==+\s+", line)

    def _parse_section_header(self, line):
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
    def _is_list_item(self, line):
        return re.match(r"^(\s)*(\*|-|\d+\.|\w+\.) ", line)

    def _parse_list_item(self, line):
        """Extract the item marker (number or bullet symbol) and the text of the item."""

        match = re.match(r"^(\s*)(\*|-|\d+\.)\s+(.*)", line)
        if match:
            indent = match.group(1)
            marker = match.group(2)  # The list marker (e.g., "*", "-", "1.")
            text = match.group(3)  # The actual text of the list item

            if marker == "*" or marker == "-":
                return {
                    "type": "list_item",
                    "marker": marker,
                    "text": text.strip(),
                    "numbered": False,
                    "indent": 0 if indent is None else len(indent),
                }
            else:
                return {
                    "type": "list_item",
                    "marker": marker,
                    "text": text.strip(),
                    "numbered": True,
                    "indent": 0 if indent is None else len(indent),
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
    def _is_table_line(self, line):
        return re.match(r"^\|.*\|", line)

    def _parse_table_line(self, line):
        # Split table cells and trim extra spaces
        return [cell.strip() for cell in line.split("|") if cell.strip()]

    def _populate_table_as_grid(self, table_data):
        num_rows = len(table_data)

        # Adjust the table data into a grid format
        num_cols = max(len(row) for row in table_data)

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
    def _is_picture(self, line):
        return re.match(r"^image::", line)

    def _parse_picture(self, line):
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
                picture_info["alt"] = attributes[0].strip() if attributes[0] else ""
                for attr in attributes[1:]:
                    key, value = attr.split("=")
                    picture_info[key.strip()] = value.strip()

            return picture_info

        return {"type": "picture", "uri": line}

    #   =========   Captions
    def _is_caption(self, line):
        return re.match(r"^\.(.+)", line)

    def _parse_caption(self, line):
        mtch = re.match(r"^\.(.+)", line)
        if mtch:
            text = mtch.group(1)
            return {"type": "caption", "text": text}

        return {"type": "caption", "text": ""}

    #   =========   Plain text
    def _parse_text(self, line):
        return {"type": "text", "text": line.strip()}
