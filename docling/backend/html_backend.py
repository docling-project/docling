import logging
import traceback
from io import BytesIO
from pathlib import Path
from typing import Final, Optional, Union, cast

from bs4 import BeautifulSoup, NavigableString, PageElement, Tag
from bs4.element import PreformattedString
from docling_core.types.doc import (
    DocItem,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupItem,
    GroupLabel,
    TableCell,
    TableData,
)
from docling_core.types.doc.document import ContentLayer
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)

# tags that generate NodeItem elements
TAGS_FOR_NODE_ITEMS: Final = [
    "address",
    "details",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "pre",
    "code",
    "ul",
    "ol",
    "li",
    "summary",
    "table",
    "figure",
    "img",
]


class HTMLDocumentBackend(DeclarativeDocumentBackend):
    @override
    def __init__(self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]):
        super().__init__(in_doc, path_or_stream)
        self.soup: Optional[Tag] = None
        # HTML file:
        self.path_or_stream = path_or_stream
        # Initialise the parents for the hierarchy
        self.max_levels = 10
        self.level = 0
        self.parents: dict[int, Optional[Union[DocItem, GroupItem]]] = {}
        for i in range(self.max_levels):
            self.parents[i] = None

        try:
            if isinstance(self.path_or_stream, BytesIO):
                text_stream = self.path_or_stream.getvalue()
                self.soup = BeautifulSoup(text_stream, "html.parser")
            if isinstance(self.path_or_stream, Path):
                with open(self.path_or_stream, "rb") as f:
                    html_content = f.read()
                    self.soup = BeautifulSoup(html_content, "html.parser")
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
        # access self.path_or_stream to load stuff
        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="text/html",
            binary_hash=self.document_hash,
        )

        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)
        _log.debug("Trying to convert HTML...")

        if self.is_valid():
            assert self.soup is not None
            content = self.soup.body or self.soup
            # Replace <br> tags with newline characters
            # TODO: remove style to avoid losing text from tags like i, b, span, ...
            for br in content("br"):
                br.replace_with(NavigableString("\n"))

            headers = content.find(["h1", "h2", "h3", "h4", "h5", "h6"])
            self.content_layer = (
                ContentLayer.BODY if headers is None else ContentLayer.FURNITURE
            )
            self.walk(content, doc)
        else:
            raise RuntimeError(
                f"Cannot convert doc with {self.document_hash} because the backend "
                "failed to init."
            )
        return doc

    def walk(self, tag: Tag, doc: DoclingDocument) -> None:
        # Iterate over elements in the body of the document
        text: str = ""
        for element in tag.children:
            if isinstance(element, Tag):
                try:
                    self.analyze_tag(cast(Tag, element), doc)
                except Exception as exc_child:
                    _log.error(
                        f"Error processing child from tag {tag.name}:\n{traceback.format_exc()}"
                    )
                    raise exc_child
            elif isinstance(element, NavigableString) and not isinstance(
                element, PreformattedString
            ):
                # Floating text outside paragraphs or analyzed tags
                text += element
                siblings: list[Tag] = [
                    item for item in element.next_siblings if isinstance(item, Tag)
                ]
                if element.next_sibling is None or any(
                    item.name in TAGS_FOR_NODE_ITEMS for item in siblings
                ):
                    text = text.strip()
                    if text and tag.name in ["div"]:
                        doc.add_text(
                            parent=self.parents[self.level],
                            label=DocItemLabel.TEXT,
                            text=text,
                            content_layer=self.content_layer,
                        )
                    text = ""

        return

    def analyze_tag(self, tag: Tag, doc: DoclingDocument) -> None:
        if tag.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            self.handle_header(tag, doc)
        elif tag.name in ["p", "address", "summary"]:
            self.handle_paragraph(tag, doc)
        elif tag.name in ["pre", "code"]:
            self.handle_code(tag, doc)
        elif tag.name in ["ul", "ol"]:
            self.handle_list(tag, doc)
        elif tag.name in ["li"]:
            self.handle_list_item(tag, doc)
        elif tag.name == "table":
            self.handle_table(tag, doc)
        elif tag.name == "figure":
            self.handle_figure(tag, doc)
        elif tag.name == "img":
            self.handle_image(tag, doc)
        elif tag.name == "details":
            self.handle_details(tag, doc)
        else:
            self.walk(tag, doc)

    def get_text(self, item: PageElement) -> str:
        """Get the text content of a tag."""
        parts: list[str] = self.extract_text_recursively(item)

        return "".join(parts) + " "

    # Function to recursively extract text from all child nodes
    def extract_text_recursively(self, item: PageElement) -> list[str]:
        result: list[str] = []

        if isinstance(item, NavigableString):
            return [item]

        tag = cast(Tag, item)
        if tag.name not in ["ul", "ol"]:
            for child in tag:
                # Recursively get the child's text content
                result.extend(self.extract_text_recursively(child))

        return ["".join(result) + " "]

    def handle_details(self, element: Tag, doc: DoclingDocument) -> None:
        """Handle details tag (details) and its content."""

        self.parents[self.level + 1] = doc.add_group(
            name="details",
            label=GroupLabel.SECTION,
            parent=self.parents[self.level],
            content_layer=self.content_layer,
        )

        self.level += 1
        self.walk(element, doc)
        self.parents[self.level + 1] = None
        self.level -= 1

    def handle_header(self, element: Tag, doc: DoclingDocument) -> None:
        """Handles header tags (h1, h2, etc.)."""
        hlevel = int(element.name.replace("h", ""))
        text = element.text.strip()

        self.content_layer = ContentLayer.BODY

        if hlevel == 1:
            for key in self.parents.keys():
                self.parents[key] = None

            self.level = 1
            self.parents[self.level] = doc.add_text(
                parent=self.parents[0],
                label=DocItemLabel.TITLE,
                text=text,
                content_layer=self.content_layer,
            )
        else:
            if hlevel > self.level:
                # add invisible group
                for i in range(self.level + 1, hlevel):
                    self.parents[i] = doc.add_group(
                        name=f"header-{i}",
                        label=GroupLabel.SECTION,
                        parent=self.parents[i - 1],
                        content_layer=self.content_layer,
                    )
                self.level = hlevel

            elif hlevel < self.level:
                # remove the tail
                for key in self.parents.keys():
                    if key > hlevel:
                        self.parents[key] = None
                self.level = hlevel

            self.parents[hlevel] = doc.add_heading(
                parent=self.parents[hlevel - 1],
                text=text,
                level=hlevel - 1,
                content_layer=self.content_layer,
            )

    def handle_code(self, element: Tag, doc: DoclingDocument) -> None:
        """Handles monospace code snippets (pre)."""
        if element.text is None:
            return
        text = element.text.strip()
        if text:
            doc.add_code(
                parent=self.parents[self.level],
                text=text,
                content_layer=self.content_layer,
            )

    def handle_paragraph(self, element: Tag, doc: DoclingDocument) -> None:
        """Handles paragraph tags (p) or equivalent ones."""
        if element.text is None:
            return
        text = element.text.strip()
        if text:
            doc.add_text(
                parent=self.parents[self.level],
                label=DocItemLabel.TEXT,
                text=text,
                content_layer=self.content_layer,
            )

    def handle_list(self, element: Tag, doc: DoclingDocument) -> None:
        """Handles list tags (ul, ol) and their list items."""

        if element.name == "ul":
            # create a list group
            self.parents[self.level + 1] = doc.add_group(
                parent=self.parents[self.level],
                name="list",
                label=GroupLabel.LIST,
                content_layer=self.content_layer,
            )
        elif element.name == "ol":
            start_attr = element.get("start")
            start: int = (
                int(start_attr)
                if isinstance(start_attr, str) and start_attr.isnumeric()
                else 1
            )
            # create a list group
            self.parents[self.level + 1] = doc.add_group(
                parent=self.parents[self.level],
                name="ordered list" + (f" start {start}" if start != 1 else ""),
                label=GroupLabel.ORDERED_LIST,
                content_layer=self.content_layer,
            )
        self.level += 1

        self.walk(element, doc)

        self.parents[self.level + 1] = None
        self.level -= 1

    def handle_list_item(self, element: Tag, doc: DoclingDocument) -> None:
        """Handles list item tags (li)."""
        nested_list = element.find(["ul", "ol"])

        parent = self.parents[self.level]
        if parent is None:
            _log.debug(f"list-item has no parent in DoclingDocument: {element}")
            return
        parent_label: str = parent.label
        index_in_list = len(parent.children) + 1
        if (
            parent_label == GroupLabel.ORDERED_LIST
            and isinstance(parent, GroupItem)
            and parent.name
        ):
            start_in_list: str = parent.name.split(" ")[-1]
            start: int = int(start_in_list) if start_in_list.isnumeric() else 1
            index_in_list += start - 1

        if nested_list:
            # Text in list item can be hidden within hierarchy, hence
            # we need to extract it recursively
            text: str = self.get_text(element)
            # Flatten text, remove break lines:
            text = text.replace("\n", "").replace("\r", "")
            text = " ".join(text.split()).strip()

            marker = ""
            enumerated = False
            if parent_label == GroupLabel.ORDERED_LIST:
                marker = str(index_in_list)
                enumerated = True

            if len(text) > 0:
                # create a list-item
                self.parents[self.level + 1] = doc.add_list_item(
                    text=text,
                    enumerated=enumerated,
                    marker=marker,
                    parent=parent,
                    content_layer=self.content_layer,
                )
                self.level += 1
                self.walk(element, doc)
                self.parents[self.level + 1] = None
                self.level -= 1
            else:
                self.walk(element, doc)

        elif element.text.strip():
            text = element.text.strip()

            marker = ""
            enumerated = False
            if parent_label == GroupLabel.ORDERED_LIST:
                marker = f"{index_in_list!s}."
                enumerated = True
            doc.add_list_item(
                text=text,
                enumerated=enumerated,
                marker=marker,
                parent=parent,
                content_layer=self.content_layer,
            )
        else:
            _log.debug(f"list-item has no text: {element}")

    @staticmethod
    def parse_table_data(element: Tag) -> Optional[TableData]:  # noqa: C901
        nested_tables = element.find("table")
        if nested_tables is not None:
            _log.debug("Skipping nested table.")
            return None

        # Find the number of rows and columns (taking into account spans)
        num_rows = 0
        num_cols = 0
        for row in element("tr"):
            col_count = 0
            is_row_header = True
            if not isinstance(row, Tag):
                continue
            for cell in row(["td", "th"]):
                if not isinstance(row, Tag):
                    continue
                cell_tag = cast(Tag, cell)
                val = cell_tag.get("colspan", "1")
                colspan = int(val) if (isinstance(val, str) and val.isnumeric()) else 1
                col_count += colspan
                if cell_tag.name == "td" or cell_tag.get("rowspan") is None:
                    is_row_header = False
            num_cols = max(num_cols, col_count)
            if not is_row_header:
                num_rows += 1

        _log.debug(f"The table has {num_rows} rows and {num_cols} cols.")

        grid: list = [[None for _ in range(num_cols)] for _ in range(num_rows)]

        data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=[])

        # Iterate over the rows in the table
        start_row_span = 0
        row_idx = -1
        for row in element("tr"):
            if not isinstance(row, Tag):
                continue

            # For each row, find all the column cells (both <td> and <th>)
            cells = row(["td", "th"])

            # Check if cell is in a column header or row header
            col_header = True
            row_header = True
            for html_cell in cells:
                if isinstance(html_cell, Tag):
                    if html_cell.name == "td":
                        col_header = False
                        row_header = False
                    elif html_cell.get("rowspan") is None:
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

                # TODO: extract content correctly from table-cells with lists
                text = html_cell.text

                # label = html_cell.name
                col_val = html_cell.get("colspan", "1")
                col_span = (
                    int(col_val)
                    if isinstance(col_val, str) and col_val.isnumeric()
                    else 1
                )
                row_val = html_cell.get("rowspan", "1")
                row_span = (
                    int(row_val)
                    if isinstance(row_val, str) and row_val.isnumeric()
                    else 1
                )
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

                table_cell = TableCell(
                    text=text,
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=start_row_span + row_idx,
                    end_row_offset_idx=start_row_span + row_idx + row_span,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + col_span,
                    column_header=col_header,
                    row_header=((not col_header) and html_cell.name == "th"),
                )
                data.table_cells.append(table_cell)

        return data

    def handle_table(self, element: Tag, doc: DoclingDocument) -> None:
        """Handles table tags."""

        table_data = HTMLDocumentBackend.parse_table_data(element)

        if table_data is not None:
            doc.add_table(
                data=table_data,
                parent=self.parents[self.level],
                content_layer=self.content_layer,
            )

    def get_list_text(self, list_element: Tag, level: int = 0) -> list[str]:
        """Recursively extract text from <ul> or <ol> with proper indentation."""
        result = []
        bullet_char = "*"  # Default bullet character for unordered lists

        if list_element.name == "ol":  # For ordered lists, use numbers
            for i, li in enumerate(list_element("li", recursive=False), 1):
                if not isinstance(li, Tag):
                    continue
                # Add numbering for ordered lists
                result.append(f"{'    ' * level}{i}. {li.get_text(strip=True)}")
                # Handle nested lists
                nested_list = li.find(["ul", "ol"])
                if isinstance(nested_list, Tag):
                    result.extend(self.get_list_text(nested_list, level + 1))
        elif list_element.name == "ul":  # For unordered lists, use bullet points
            for li in list_element("li", recursive=False):
                if not isinstance(li, Tag):
                    continue
                # Add bullet points for unordered lists
                result.append(
                    f"{'    ' * level}{bullet_char} {li.get_text(strip=True)}"
                )
                # Handle nested lists
                nested_list = li.find(["ul", "ol"])
                if isinstance(nested_list, Tag):
                    result.extend(self.get_list_text(nested_list, level + 1))

        return result

    def handle_figure(self, element: Tag, doc: DoclingDocument) -> None:
        """Handles image tags (img)."""

        # Extract the image URI from the <img> tag
        # image_uri = root.xpath('//figure//img/@src')[0]

        contains_captions = element.find(["figcaption"])
        if not isinstance(contains_captions, Tag):
            doc.add_picture(
                parent=self.parents[self.level],
                caption=None,
                content_layer=self.content_layer,
            )
        else:
            texts = []
            for item in contains_captions:
                texts.append(item.text)

            fig_caption = doc.add_text(
                label=DocItemLabel.CAPTION,
                text=("".join(texts)).strip(),
                content_layer=self.content_layer,
            )
            doc.add_picture(
                parent=self.parents[self.level],
                caption=fig_caption,
                content_layer=self.content_layer,
            )

    def handle_image(self, element: Tag, doc: DoclingDocument) -> None:
        """Handles image tags (img)."""
        _log.debug(f"ignoring <img> tags at the moment: {element}")

        doc.add_picture(
            parent=self.parents[self.level],
            caption=None,
            content_layer=self.content_layer,
        )
