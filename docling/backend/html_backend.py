import logging
from io import BytesIO
from pathlib import Path
from typing import Set, Union

from bs4 import BeautifulSoup
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupLabel,
    TableCell,
    TableData,
)

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


class HTMLDocumentBackend(DeclarativeDocumentBackend):
    def __init__(self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]):
        super().__init__(in_doc, path_or_stream)
        _log.debug("About to init HTML backend...")
        self.soup = None
        # HTML file:
        self.path_or_stream = path_or_stream
        # Initialise the parents for the hierarchy
        self.max_levels = 10
        self.level = 0
        self.parents = {}  # type: ignore
        for i in range(0, self.max_levels):
            self.parents[i] = None
        self.labels = {}  # type: ignore

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
                f"Could not initialize HTML backend for file with hash {self.document_hash}."
            ) from e

    def is_valid(self) -> bool:
        return self.soup is not None

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    def unload(self):
        if isinstance(self.path_or_stream, BytesIO):
            self.path_or_stream.close()

        self.path_or_stream = None

    @classmethod
    def supported_formats(cls) -> Set[InputFormat]:
        return {InputFormat.HTML}

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
            # Replace <br> tags with newline characters
            for br in self.soup.body.find_all("br"):
                br.replace_with("\n")
            doc = self.walk(self.soup.body, doc)
        else:
            raise RuntimeError(
                f"Cannot convert doc with {self.document_hash} because the backend failed to init."
            )
        return doc

    def walk(self, element, doc):
        try:
            # Iterate over elements in the body of the document
            for idx, element in enumerate(element.children):
                try:
                    self.analyse_element(element, idx, doc)
                except Exception as exc_child:

                    _log.error(" -> error treating child: ", exc_child)
                    _log.error(" => element: ", element, "\n")
                    raise exc_child

        except Exception as exc:
            pass

        return doc

    def analyse_element(self, element, idx, doc):
        """
        if element.name!=None:
            _log.debug("\t"*self.level, idx, "\t", f"{element.name} ({self.level})")
        """

        if element.name in self.labels:
            self.labels[element.name] += 1
        else:
            self.labels[element.name] = 1

        if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            self.handle_header(element, idx, doc)
        elif element.name in ["p"]:
            self.handle_paragraph(element, idx, doc)
        elif element.name in ["pre"]:
            self.handle_code(element, idx, doc)
        elif element.name in ["ul", "ol"]:
            self.handle_list(element, idx, doc)
        elif element.name in ["li"]:
            self.handle_listitem(element, idx, doc)
        elif element.name == "table":
            self.handle_table(element, idx, doc)
        elif element.name == "figure":
            self.handle_figure(element, idx, doc)
        elif element.name == "img":
            self.handle_image(element, idx, doc)
        else:
            self.walk(element, doc)

    def get_direct_text(self, item):
        """Get the direct text of the <li> element (ignoring nested lists)."""
        text = item.find(string=True, recursive=False)
        if isinstance(text, str):
            return text.strip()

        return ""

    # Function to recursively extract text from all child nodes
    def extract_text_recursively(self, item):
        result = []

        if isinstance(item, str):
            return [item]

        if item.name not in ["ul", "ol"]:
            try:
                # Iterate over the children (and their text and tails)
                for child in item:
                    try:
                        # Recursively get the child's text content
                        result.extend(self.extract_text_recursively(child))
                    except:
                        pass
            except:
                _log.warn("item has no children")
                pass

        return "".join(result) + " "

    def handle_header(self, element, idx, doc):
        """Handles header tags (h1, h2, etc.)."""
        hlevel = int(element.name.replace("h", ""))
        slevel = hlevel - 1

        label = DocItemLabel.SECTION_HEADER
        text = element.text.strip()

        if hlevel == 1:
            for key, val in self.parents.items():
                self.parents[key] = None

            self.level = 1
            self.parents[self.level] = doc.add_text(
                parent=self.parents[0], label=DocItemLabel.TITLE, text=text
            )
        else:
            if hlevel > self.level:

                # add invisible group
                for i in range(self.level + 1, hlevel):
                    self.parents[i] = doc.add_group(
                        name=f"header-{i}",
                        label=GroupLabel.SECTION,
                        parent=self.parents[i - 1],
                    )
                self.level = hlevel

            elif hlevel < self.level:

                # remove the tail
                for key, val in self.parents.items():
                    if key > hlevel:
                        self.parents[key] = None
                self.level = hlevel

            self.parents[hlevel] = doc.add_heading(
                parent=self.parents[hlevel - 1],
                text=text,
                level=hlevel,
            )

    def handle_code(self, element, idx, doc):
        """Handles monospace code snippets (pre)."""
        if element.text is None:
            return
        text = element.text.strip()
        label = DocItemLabel.CODE
        if len(text) == 0:
            return
        doc.add_text(parent=self.parents[self.level], label=label, text=text)

    def handle_paragraph(self, element, idx, doc):
        """Handles paragraph tags (p)."""
        if element.text is None:
            return
        text = element.text.strip()
        label = DocItemLabel.PARAGRAPH
        if len(text) == 0:
            return
        doc.add_text(parent=self.parents[self.level], label=label, text=text)

    def handle_list(self, element, idx, doc):
        """Handles list tags (ul, ol) and their list items."""

        if element.name == "ul":
            # create a list group
            self.parents[self.level + 1] = doc.add_group(
                parent=self.parents[self.level], name="list", label=GroupLabel.LIST
            )
        elif element.name == "ol":
            # create a list group
            self.parents[self.level + 1] = doc.add_group(
                parent=self.parents[self.level],
                name="ordered list",
                label=GroupLabel.ORDERED_LIST,
            )
        self.level += 1

        self.walk(element, doc)

        self.parents[self.level + 1] = None
        self.level -= 1

    def handle_listitem(self, element, idx, doc):
        """Handles listitem tags (li)."""
        nested_lists = element.find(["ul", "ol"])

        parent_list_label = self.parents[self.level].label
        index_in_list = len(self.parents[self.level].children) + 1

        if nested_lists:
            name = element.name
            # Text in list item can be hidden within hierarchy, hence
            # we need to extract it recursively
            text = self.extract_text_recursively(element)
            # Flatten text, remove break lines:
            text = text.replace("\n", "").replace("\r", "")
            text = " ".join(text.split()).strip()

            marker = ""
            enumerated = False
            if parent_list_label == GroupLabel.ORDERED_LIST:
                marker = str(index_in_list)
                enumerated = True

            if len(text) > 0:
                # create a list-item
                self.parents[self.level + 1] = doc.add_list_item(
                    text=text,
                    enumerated=enumerated,
                    marker=marker,
                    parent=self.parents[self.level],
                )
                self.level += 1

            self.walk(element, doc)

            self.parents[self.level + 1] = None
            self.level -= 1

        elif isinstance(element.text, str):
            text = element.text.strip()

            marker = ""
            enumerated = False
            if parent_list_label == GroupLabel.ORDERED_LIST:
                marker = f"{str(index_in_list)}."
                enumerated = True
            doc.add_list_item(
                text=text,
                enumerated=enumerated,
                marker=marker,
                parent=self.parents[self.level],
            )
        else:
            _log.warn("list-item has no text: ", element)

    def handle_table(self, element, idx, doc):
        """Handles table tags."""

        nested_tables = element.find("table")
        if nested_tables is not None:
            _log.warn("detected nested tables: skipping for now")
            return

        # Count the number of rows (number of <tr> elements)
        num_rows = len(element.find_all("tr"))

        # Find the number of columns (taking into account colspan)
        num_cols = 0
        for row in element.find_all("tr"):
            col_count = 0
            for cell in row.find_all(["td", "th"]):
                colspan = int(cell.get("colspan", 1))
                col_count += colspan
            num_cols = max(num_cols, col_count)

        grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]

        data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=[])

        # Iterate over the rows in the table
        for row_idx, row in enumerate(element.find_all("tr")):

            # For each row, find all the column cells (both <td> and <th>)
            cells = row.find_all(["td", "th"])

            # Check if each cell in the row is a header -> means it is a column header
            col_header = True
            for j, html_cell in enumerate(cells):
                if html_cell.name == "td":
                    col_header = False

            col_idx = 0
            # Extract and print the text content of each cell
            for _, html_cell in enumerate(cells):

                text = html_cell.text
                try:
                    text = self.extract_table_cell_text(html_cell)
                except Exception as exc:
                    _log.warn("exception: ", exc)
                    exit(-1)

                # label = html_cell.name

                col_span = int(html_cell.get("colspan", 1))
                row_span = int(html_cell.get("rowspan", 1))

                while grid[row_idx][col_idx] is not None:
                    col_idx += 1
                for r in range(row_span):
                    for c in range(col_span):
                        grid[row_idx + r][col_idx + c] = text

                cell = TableCell(
                    text=text,
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=row_idx,
                    end_row_offset_idx=row_idx + row_span,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + col_span,
                    col_header=col_header,
                    row_header=((not col_header) and html_cell.name == "th"),
                )
                data.table_cells.append(cell)

        doc.add_table(data=data, parent=self.parents[self.level])

    def get_list_text(self, list_element, level=0):
        """Recursively extract text from <ul> or <ol> with proper indentation."""
        result = []
        bullet_char = "*"  # Default bullet character for unordered lists

        if list_element.name == "ol":  # For ordered lists, use numbers
            for i, li in enumerate(list_element.find_all("li", recursive=False), 1):
                # Add numbering for ordered lists
                result.append(f"{'    ' * level}{i}. {li.get_text(strip=True)}")
                # Handle nested lists
                nested_list = li.find(["ul", "ol"])
                if nested_list:
                    result.extend(self.get_list_text(nested_list, level + 1))
        elif list_element.name == "ul":  # For unordered lists, use bullet points
            for li in list_element.find_all("li", recursive=False):
                # Add bullet points for unordered lists
                result.append(
                    f"{'    ' * level}{bullet_char} {li.get_text(strip=True)}"
                )
                # Handle nested lists
                nested_list = li.find(["ul", "ol"])
                if nested_list:
                    result.extend(self.get_list_text(nested_list, level + 1))

        return result

    def extract_table_cell_text(self, cell):
        """Extract text from a table cell, including lists with indents."""
        contains_lists = cell.find(["ul", "ol"])
        if contains_lists is None:
            return cell.text
        else:
            _log.debug(
                "should extract the content correctly for table-cells with lists ..."
            )
            return cell.text

    def handle_figure(self, element, idx, doc):
        """Handles image tags (img)."""

        # Extract the image URI from the <img> tag
        # image_uri = root.xpath('//figure//img/@src')[0]

        contains_captions = element.find(["figcaption"])
        if contains_captions is None:
            doc.add_picture(parent=self.parents[self.level], caption=None)

        else:
            texts = []
            for item in contains_captions:
                texts.append(item.text)

            fig_caption = doc.add_text(
                label=DocItemLabel.CAPTION, text=("".join(texts)).strip()
            )
            doc.add_picture(
                parent=self.parents[self.level],
                caption=fig_caption,
            )

    def handle_image(self, element, idx, doc):
        """Handles image tags (img)."""
        doc.add_picture(parent=self.parents[self.level], caption=None)
