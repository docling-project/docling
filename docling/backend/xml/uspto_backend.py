"""Backend to parse patents from the United States Patent Office (USPTO).

The parsers included in this module can handle patent grants published since 1976 and
patent applications since 2001.
The original files can be found in https://bulkdata.uspto.gov.
"""

import html
import logging
import re
import xml.sax
import xml.sax.xmlreader
from abc import ABC, abstractmethod
from enum import Enum, unique
from io import BytesIO
from pathlib import Path
from typing import Final, Optional, Union

from bs4 import BeautifulSoup, Tag
from docling_core.types.doc import (
    DocItem,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    TableCell,
    TableData,
    TextItem,
)
from docling_core.types.doc.document import LevelNumber
from pydantic import NonNegativeInt
from typing_extensions import Self, TypedDict, override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)

XML_DECLARATION: Final = '<?xml version="1.0" encoding="UTF-8"?>'


@unique
class PatentHeading(Enum):
    """Text of docling headings for tagged sections in USPTO patent documents."""

    ABSTRACT = "ABSTRACT", 2
    CLAIMS = "CLAIMS", 2

    @override
    def __new__(cls, value: str, _) -> Self:
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    @override
    def __init__(self, _, level: LevelNumber) -> None:
        self.level: LevelNumber = level


class PatentUsptoDocumentBackend(DeclarativeDocumentBackend):
    @override
    def __init__(
        self, in_doc: InputDocument, path_or_stream: Union[BytesIO, Path]
    ) -> None:
        super().__init__(in_doc, path_or_stream)

        self.patent_content: str = ""
        self.parser: Optional[PatentUspto] = None

        try:
            if isinstance(self.path_or_stream, BytesIO):
                while line := self.path_or_stream.readline().decode("utf-8"):
                    if line.startswith("<!DOCTYPE") or line == "PATN\n":
                        self._set_parser(line)
                    self.patent_content += line
            elif isinstance(self.path_or_stream, Path):
                with open(self.path_or_stream, encoding="utf-8") as file_obj:
                    while line := file_obj.readline():
                        if line.startswith("<!DOCTYPE") or line == "PATN\n":
                            self._set_parser(line)
                        self.patent_content += line
        except Exception as exc:
            raise RuntimeError(
                f"Could not initialize USPTO backend for file with hash {self.document_hash}."
            ) from exc

    def _set_parser(self, doctype: str) -> None:
        doctype_line = doctype.lower()
        if doctype == "PATN\n":
            self.parser = PatentUsptoGrantAps()
        elif "us-patent-application-v4" in doctype_line:
            self.parser = PatentUsptoIce()
        elif "us-patent-grant-v4" in doctype_line:
            self.parser = PatentUsptoIce()
        elif "us-grant-025" in doctype_line:
            self.parser = PatentUsptoGrantV2()
        elif all(
            item in doctype_line
            for item in ("patent-application-publication", "pap-v1")
        ):
            self.parser = PatentUsptoAppV1()
        else:
            self.parser = None

    @override
    def is_valid(self) -> bool:
        return bool(self.patent_content) and bool(self.parser)

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return False

    @override
    def unload(self) -> None:
        return

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.XML_USPTO}

    @override
    def convert(self) -> DoclingDocument:
        if self.parser is not None:
            doc = self.parser.parse(self.patent_content)
            if doc is None:
                raise RuntimeError(
                    f"Failed to convert doc (hash={self.document_hash}, "
                    f"name={self.file.name})."
                )
            doc.name = self.file.name or "file"
            mime_type = (
                "text/plain"
                if isinstance(self.parser, PatentUsptoGrantAps)
                else "application/xml"
            )
            doc.origin = DocumentOrigin(
                mimetype=mime_type,
                binary_hash=self.document_hash,
                filename=self.file.name or "file",
            )

            return doc
        else:
            raise RuntimeError(
                f"Cannot convert doc (hash={self.document_hash}, "
                f"name={self.file.name}) because the backend failed to init."
            )


class PatentUspto(ABC):
    """Parser of patent documents from the US Patent Office."""

    @abstractmethod
    def parse(self, patent_content: str) -> Optional[DoclingDocument]:
        """Parse a USPTO patent.

        Parameters:
            patent_content: The content of a single patent in a USPTO file.

        Returns:
            The patent parsed as a docling document.
        """


class PatentUsptoIce(PatentUspto):
    """Parser of patent documents from the US Patent Office (ICE).

    The compatible formats are:
    - Patent Grant Full Text Data/XML Version 4.x ICE (from January 2005)
    - Patent Application Full Text Data/XML Version 4.x ICE (from January 2005)
    """

    def __init__(self) -> None:
        """Build an instance of PatentUsptoIce class."""
        self.handler = PatentUsptoIce.PatentHandler()
        self.pattern = re.compile(r"^(<table .*?</table>)", re.MULTILINE | re.DOTALL)

    def parse(self, patent_content: str) -> Optional[DoclingDocument]:
        try:
            xml.sax.parseString(patent_content, self.handler)
        except xml.sax._exceptions.SAXParseException as exc_sax:
            _log.error(f"Error in parsing USPTO document: {exc_sax}")

            return None

        doc = self.handler.doc
        if doc:
            raw_tables = re.findall(self.pattern, patent_content)
            parsed_tables: list[TableData] = []
            _log.debug(f"Found {len(raw_tables)} tables to be parsed with XmlTable.")
            for table in raw_tables:
                table_parser = XmlTable(XML_DECLARATION + "\n" + table)
                try:
                    table_data = table_parser.parse()
                    if table_data:
                        parsed_tables.append(table_data)
                except Exception as exc_table:
                    _log.error(f"Error in parsing USPTO tables: {exc_table}")
            if len(parsed_tables) != len(doc.tables):
                _log.error(
                    f"Number of referenced ({len(doc.tables)}) and parsed "
                    f"({len(parsed_tables)}) tables differ."
                )
            else:
                for idx, item in enumerate(parsed_tables):
                    doc.tables[idx].data = item

        return doc

    class PatentHandler(xml.sax.handler.ContentHandler):
        """SAX ContentHandler for patent documents."""

        APP_DOC_ELEMENT: Final = "us-patent-application"
        GRANT_DOC_ELEMENT: Final = "us-patent-grant"

        @unique
        class Element(Enum):
            """Represents an element of interest in the patent application document."""

            ABSTRACT = "abstract", True
            TITLE = "invention-title", True
            CLAIMS = "claims", False
            CLAIM = "claim", False
            CLAIM_TEXT = "claim-text", True
            PARAGRAPH = "p", True
            HEADING = "heading", True
            DESCRIPTION = "description", False
            TABLE = "table", False  # to track its position, without text
            DRAWINGS = "description-of-drawings", True
            STYLE_SUPERSCRIPT = "sup", True
            STYLE_SUBSCRIPT = "sub", True
            MATHS = "maths", False  # to avoid keeping formulas

            @override
            def __new__(cls, value: str, _) -> Self:
                obj = object.__new__(cls)
                obj._value_ = value
                return obj

            @override
            def __init__(self, _, is_text: bool) -> None:
                self.is_text: bool = is_text

        @override
        def __init__(self) -> None:
            """Build an instance of the patent handler."""
            # Current patent being parsed
            self.doc: Optional[DoclingDocument] = None
            # Keep track of docling hierarchy level
            self.level: LevelNumber = 1
            # Keep track of docling parents by level
            self.parents: dict[LevelNumber, Optional[DocItem]] = {1: None}
            # Content to retain for the current patent
            self.property: list[str]
            self.claim: str
            self.claims: list[str]
            self.abstract: str
            self.text: str
            self._clean_data()
            # To handle mathematical styling
            self.style_html = HtmlEntity()

        @override
        def startElement(self, tag, attributes):
            """Signal the start of an element.

            Args:
                tag: The element tag.
                attributes: The element attributes.
            """
            if tag in (
                self.APP_DOC_ELEMENT,
                self.GRANT_DOC_ELEMENT,
            ):
                self.doc = DoclingDocument(name="file")
                self.text = ""
            self._start_registered_elements(tag, attributes)

        @override
        def skippedEntity(self, name):
            """Receive notification of a skipped entity.

            HTML entities will be skipped by the parser. This method will unescape them
            and add them to the text.

            Args:
                name: Entity name.
            """
            if self.property:
                elm_val = self.property[-1]
                element = self.Element(elm_val)
                if element.is_text:
                    escaped = self.style_html.get_greek_from_iso8879(f"&{name};")
                    unescaped = html.unescape(escaped)
                    if unescaped == escaped:
                        _log.debug(f"Unrecognized HTML entity: {name}")
                        return

                    if element in (
                        self.Element.STYLE_SUPERSCRIPT,
                        self.Element.STYLE_SUBSCRIPT,
                    ):
                        # superscripts and subscripts need to be under text elements
                        if len(self.property) < 2:
                            return
                        parent_val = self.property[-2]
                        parent = self.Element(parent_val)
                        if parent.is_text:
                            self.text += self._apply_style(unescaped, elm_val)
                    else:
                        self.text += unescaped

        @override
        def endElement(self, tag):
            """Signal the end of an element.

            Args:
                tag: The element tag.
            """
            if tag in (
                self.APP_DOC_ELEMENT,
                self.GRANT_DOC_ELEMENT,
            ):
                self._clean_data()
            self._end_registered_element(tag)

        @override
        def characters(self, content):
            """Receive notification of character data.

            Args:
                content: Data reported by the handler.
            """
            if self.property:
                elm_val = self.property[-1]
                element = self.Element(elm_val)
                if element.is_text:
                    if element in (
                        self.Element.STYLE_SUPERSCRIPT,
                        self.Element.STYLE_SUBSCRIPT,
                    ):
                        # superscripts and subscripts need to be under text elements
                        if len(self.property) < 2:
                            return
                        parent_val = self.property[-2]
                        parent = self.Element(parent_val)
                        if parent.is_text:
                            self.text += self._apply_style(content, elm_val)
                    else:
                        self.text += content

        def _start_registered_elements(
            self, tag: str, attributes: xml.sax.xmlreader.AttributesImpl
        ) -> None:
            if tag in [member.value for member in self.Element]:
                # special case for claims: claim lines may start before the
                # previous one is closed
                if (
                    tag == self.Element.CLAIM_TEXT.value
                    and self.property
                    and self.property[-1] == tag
                    and self.text.strip()
                ):
                    self.claim += " " + self.text.strip()
                    self.text = ""
                elif tag == self.Element.HEADING.value:
                    level_attr: str = attributes.get("level", "")
                    new_level: int = int(level_attr) if level_attr.isnumeric() else 1
                    max_level = min(self.parents.keys())
                    # increase heading level with 1 for title, if any
                    self.level = (
                        new_level + 1 if (new_level + 1) in self.parents else max_level
                    )
                self.property.append(tag)

        def _end_registered_element(self, tag: str) -> None:
            if tag in [item.value for item in self.Element] and self.property:
                current_tag = self.property.pop()
                self._add_property(current_tag, self.text.strip())

        def _add_property(self, name: str, text: str) -> None:
            if not name or not self.doc:
                return

            if name == self.Element.TITLE.value:
                if text:
                    self.parents[self.level + 1] = self.doc.add_title(
                        parent=self.parents[self.level],
                        text=text,
                    )
                    self.level += 1
                self.text = ""

            elif name == self.Element.ABSTRACT.value:
                if self.abstract:
                    heading_text = PatentHeading.ABSTRACT.value
                    heading_level = (
                        PatentHeading.ABSTRACT.level
                        if PatentHeading.ABSTRACT.level in self.parents
                        else 1
                    )
                    abstract_item = self.doc.add_heading(
                        heading_text,
                        level=heading_level,
                        parent=self.parents[heading_level],
                    )
                    self.doc.add_text(
                        label=DocItemLabel.PARAGRAPH,
                        text=self.abstract,
                        parent=abstract_item,
                    )

            elif name == self.Element.CLAIM_TEXT.value:
                text = re.sub("\\s+", " ", text).strip()
                if text:
                    self.claim += " " + text
                self.text = ""

            elif name == self.Element.CLAIM.value and self.claim:
                self.claims.append(self.claim.strip())
                self.claim = ""

            elif name == self.Element.CLAIMS.value and self.claims:
                heading_text = PatentHeading.CLAIMS.value
                heading_level = (
                    PatentHeading.CLAIMS.level
                    if PatentHeading.CLAIMS.level in self.parents
                    else 1
                )
                claims_item = self.doc.add_heading(
                    heading_text,
                    level=heading_level,
                    parent=self.parents[heading_level],
                )
                for text in self.claims:
                    self.doc.add_text(
                        label=DocItemLabel.PARAGRAPH, text=text, parent=claims_item
                    )

            elif name == self.Element.PARAGRAPH.value and text:
                # remove blank spaces added in paragraphs
                text = re.sub("\\s+", " ", text)
                if self.Element.ABSTRACT.value in self.property:
                    self.abstract = (
                        (self.abstract + " " + text) if self.abstract else text
                    )
                else:
                    self.doc.add_text(
                        label=DocItemLabel.PARAGRAPH,
                        text=text,
                        parent=self.parents[self.level],
                    )
                self.text = ""

            elif name == self.Element.HEADING.value and text:
                self.parents[self.level + 1] = self.doc.add_heading(
                    text=text,
                    level=self.level,
                    parent=self.parents[self.level],
                )
                self.level += 1
                self.text = ""

            elif name == self.Element.TABLE.value:
                # set an empty table as placeholder
                empty_table = TableData(num_rows=0, num_cols=0, table_cells=[])
                self.doc.add_table(
                    data=empty_table,
                    parent=self.parents[self.level],
                )

        def _apply_style(self, text: str, style_tag: str) -> str:
            """Apply an HTML style to text.

            Args:
                text: A string containing plain text.
                style_tag: An HTML tag name for styling text. If the tag name is not
                  recognized as one of the supported styles, the method will return
                  the original `text`.

            Returns:
                A string after applying the style.
            """
            formatted = text

            if style_tag == self.Element.STYLE_SUPERSCRIPT.value:
                formatted = html.unescape(self.style_html.get_superscript(text))
            elif style_tag == self.Element.STYLE_SUBSCRIPT.value:
                formatted = html.unescape(self.style_html.get_subscript(text))

            return formatted

        def _clean_data(self) -> None:
            """Reset the variables from stream data."""
            self.property = []
            self.claim = ""
            self.claims = []
            self.abstract = ""


class PatentUsptoGrantV2(PatentUspto):
    """Parser of patent documents from the US Patent Office (grants v2.5).

    The compatible format is:
    - Patent Grant Full Text Data/XML Version 2.5 (from January 2002 till December 2004)
    """

    @override
    def __init__(self) -> None:
        """Build an instance of PatentUsptoGrantV2 class."""
        self.handler = PatentUsptoGrantV2.PatentHandler()
        self.pattern = re.compile(r"^(<table .*?</table>)", re.MULTILINE | re.DOTALL)

    @override
    def parse(self, patent_content: str) -> Optional[DoclingDocument]:
        try:
            xml.sax.parseString(patent_content, self.handler)
        except xml.sax._exceptions.SAXParseException as exc_sax:
            _log.error(f"Error in parsing USPTO document: {exc_sax}")

            return None

        doc = self.handler.doc
        if doc:
            raw_tables = re.findall(self.pattern, patent_content)
            parsed_tables: list[TableData] = []
            _log.debug(f"Found {len(raw_tables)} tables to be parsed with XmlTable.")
            for table in raw_tables:
                table_parser = XmlTable(XML_DECLARATION + "\n" + table)
                try:
                    table_data = table_parser.parse()
                    if table_data:
                        parsed_tables.append(table_data)
                except Exception as exc_table:
                    _log.error(f"Error in parsing USPTO tables: {exc_table}")
            if len(parsed_tables) != len(doc.tables):
                _log.error(
                    f"Number of referenced ({len(doc.tables)}) and parsed "
                    f"({len(parsed_tables)}) tables differ."
                )
            else:
                for idx, item in enumerate(parsed_tables):
                    doc.tables[idx].data = item

        return doc

    class PatentHandler(xml.sax.handler.ContentHandler):
        """SAX ContentHandler for patent documents."""

        GRANT_DOC_ELEMENT: Final = "PATDOC"
        CLAIM_STATEMENT: Final = "What is claimed is:"

        @unique
        class Element(Enum):
            """Represents an element of interest in the patent application document."""

            PDAT = "PDAT", True  # any type of data
            ABSTRACT = ("SDOAB", False)
            SDOCL = ("SDOCL", False)
            TITLE = ("B540", False)
            CLAIMS = ("CL", False)
            CLAIM = ("CLM", False)
            PARAGRAPH = ("PARA", True)
            HEADING = ("H", True)
            DRAWINGS = ("DRWDESC", False)
            STYLE_SUPERSCRIPT = ("SP", False)
            STYLE_SUBSCRIPT = ("SB", False)
            STYLE_ITALIC = ("ITALIC", False)
            CWU = ("CWU", False)  # avoid tables, chemicals, formulas
            TABLE = ("table", False)  # to keep track of table positions

            @override
            def __new__(cls, value: str, _) -> Self:
                obj = object.__new__(cls)
                obj._value_ = value
                return obj

            @override
            def __init__(self, _, is_text: bool) -> None:
                self.is_text: bool = is_text

        @override
        def __init__(self) -> None:
            """Build an instance of the patent handler."""
            # Current patent being parsed
            self.doc: Optional[DoclingDocument] = None
            # Keep track of docling hierarchy level
            self.level: LevelNumber = 1
            # Keep track of docling parents by level
            self.parents: dict[LevelNumber, Optional[DocItem]] = {1: None}
            # Content to retain for the current patent
            self.property: list[str]
            self.claim: str
            self.claims: list[str]
            self.paragraph: str
            self.abstract: str
            self._clean_data()
            # To handle mathematical styling
            self.style_html = HtmlEntity()

        @override
        def startElement(self, tag, attributes):
            """Signal the start of an element.

            Args:
                tag: The element tag.
                attributes: The element attributes.
            """
            if tag == self.GRANT_DOC_ELEMENT:
                self.doc = DoclingDocument(name="file")
                self.text = ""
            self._start_registered_elements(tag, attributes)

        @override
        def skippedEntity(self, name):
            """Receive notification of a skipped entity.

            HTML entities will be skipped by the parser. This method will unescape them
            and add them to the text.

            Args:
                name: Entity name.
            """
            if self.property:
                elm_val = self.property[-1]
                element = self.Element(elm_val)
                if element.is_text:
                    escaped = self.style_html.get_greek_from_iso8879(f"&{name};")
                    unescaped = html.unescape(escaped)
                    if unescaped == escaped:
                        logging.debug("Unrecognized HTML entity: " + name)
                        return

                    if element in (
                        self.Element.STYLE_SUPERSCRIPT,
                        self.Element.STYLE_SUBSCRIPT,
                    ):
                        # superscripts and subscripts need to be under text elements
                        if len(self.property) < 2:
                            return
                        parent_val = self.property[-2]
                        parent = self.Element(parent_val)
                        if parent.is_text:
                            self.text += self._apply_style(unescaped, elm_val)
                    else:
                        self.text += unescaped

        @override
        def endElement(self, tag):
            """Signal the end of an element.

            Args:
                tag: The element tag.
            """
            if tag == self.GRANT_DOC_ELEMENT:
                self._clean_data()
            self._end_registered_element(tag)

        @override
        def characters(self, content):
            """Receive notification of character data.

            Args:
                content: Data reported by the handler.
            """
            if self.property:
                elm_val = self.property[-1]
                element = self.Element(elm_val)
                if element.is_text:
                    if element in (
                        self.Element.STYLE_SUPERSCRIPT,
                        self.Element.STYLE_SUBSCRIPT,
                    ):
                        # superscripts and subscripts need to be under text elements
                        if len(self.property) < 2:
                            return
                        parent_val = self.property[-2]
                        parent = self.Element(parent_val)
                        if parent.is_text:
                            self.text += self._apply_style(content, elm_val)
                    else:
                        self.text += content

        def _start_registered_elements(
            self, tag: str, attributes: xml.sax.xmlreader.AttributesImpl
        ) -> None:
            if tag in [member.value for member in self.Element]:
                if (
                    tag == self.Element.HEADING.value
                    and self.Element.SDOCL.value not in self.property
                ):
                    level_attr: str = attributes.get("LVL", "")
                    new_level: int = int(level_attr) if level_attr.isnumeric() else 1
                    max_level = min(self.parents.keys())
                    # increase heading level with 1 for title, if any
                    self.level = (
                        new_level + 1 if (new_level + 1) in self.parents else max_level
                    )
                self.property.append(tag)

        def _end_registered_element(self, tag: str) -> None:
            if tag in [elm.value for elm in self.Element] and self.property:
                current_tag = self.property.pop()
                self._add_property(current_tag, self.text)

        def _add_property(self, name: str, text: str) -> None:
            if not name or not self.doc:
                return
            if name == self.Element.PDAT.value and text:
                if not self.property:
                    self.text = ""
                    return

                wrapper = self.property[-1]
                text = self._apply_style(text, wrapper)

                if self.Element.TITLE.value in self.property and text.strip():
                    title = text.strip()
                    self.parents[self.level + 1] = self.doc.add_title(
                        parent=self.parents[self.level],
                        text=title,
                    )
                    self.level += 1

                elif self.Element.ABSTRACT.value in self.property:
                    self.abstract += text

                elif self.Element.CLAIM.value in self.property:
                    self.claim += text

                # Paragraph text not in claims or abstract
                elif (
                    self.Element.PARAGRAPH.value in self.property
                    and self.Element.CLAIM.value not in self.property
                    and self.Element.ABSTRACT.value not in self.property
                ):
                    self.paragraph += text

                # headers except claims statement
                elif (
                    self.Element.HEADING.value in self.property
                    and self.Element.SDOCL.value not in self.property
                    and text.strip()
                ):
                    self.parents[self.level + 1] = self.doc.add_heading(
                        text=text.strip(),
                        level=self.level,
                        parent=self.parents[self.level],
                    )
                    self.level += 1

                self.text = ""

            elif name == self.Element.CLAIM.value and self.claim.strip():
                self.claims.append(self.claim.strip())
                self.claim = ""

            elif name == self.Element.CLAIMS.value and self.claims:
                heading_text = PatentHeading.CLAIMS.value
                heading_level = (
                    PatentHeading.CLAIMS.level
                    if PatentHeading.CLAIMS.level in self.parents
                    else 1
                )
                claims_item = self.doc.add_heading(
                    heading_text,
                    level=heading_level,
                    parent=self.parents[heading_level],
                )
                for text in self.claims:
                    self.doc.add_text(
                        label=DocItemLabel.PARAGRAPH, text=text, parent=claims_item
                    )

            elif name == self.Element.ABSTRACT.value and self.abstract.strip():
                abstract = self.abstract.strip()
                heading_text = PatentHeading.ABSTRACT.value
                heading_level = (
                    PatentHeading.ABSTRACT.level
                    if PatentHeading.ABSTRACT.level in self.parents
                    else 1
                )
                abstract_item = self.doc.add_heading(
                    heading_text,
                    level=heading_level,
                    parent=self.parents[heading_level],
                )
                self.doc.add_text(
                    label=DocItemLabel.PARAGRAPH, text=abstract, parent=abstract_item
                )

            elif name == self.Element.PARAGRAPH.value:
                paragraph = self.paragraph.strip()
                if paragraph and self.Element.CLAIM.value not in self.property:
                    self.doc.add_text(
                        label=DocItemLabel.PARAGRAPH,
                        text=paragraph,
                        parent=self.parents[self.level],
                    )
                elif self.Element.CLAIM.value in self.property:
                    # we may need a space after a paragraph in claim text
                    self.claim += " "
                self.paragraph = ""

            elif name == self.Element.TABLE.value:
                # set an empty table as placeholder
                empty_table = TableData(num_rows=0, num_cols=0, table_cells=[])
                self.doc.add_table(
                    data=empty_table,
                    parent=self.parents[self.level],
                )

        def _apply_style(self, text: str, style_tag: str) -> str:
            """Apply an HTML style to text.

            Args:
                text: A string containing plain text.
                style_tag: An HTML tag name for styling text. If the tag name is not
                  recognized as one of the supported styles, the method will return
                  the original `text`.

            Returns:
                A string after applying the style.
            """
            formatted = text

            if style_tag == self.Element.STYLE_SUPERSCRIPT.value:
                formatted = html.unescape(self.style_html.get_superscript(text))
            elif style_tag == self.Element.STYLE_SUBSCRIPT.value:
                formatted = html.unescape(self.style_html.get_subscript(text))
            elif style_tag == self.Element.STYLE_ITALIC.value:
                formatted = html.unescape(self.style_html.get_math_italic(text))

            return formatted

        def _clean_data(self) -> None:
            """Reset the variables from stream data."""
            self.text = ""
            self.property = []
            self.claim = ""
            self.claims = []
            self.paragraph = ""
            self.abstract = ""


class PatentUsptoGrantAps(PatentUspto):
    """Parser of patents documents from the US Patent Office (grants APS).

    The compatible format is:
    - Patent Grant Full Text Data/APS (from January 1976 till December 2001)
    """

    @unique
    class Section(Enum):
        """Represent a section in a patent APS document."""

        ABSTRACT = "ABST"
        SUMMARY = "BSUM"
        DETAILS = "DETD"
        CLAIMS = "CLMS"
        DRAWINGS = "DRWD"

    @unique
    class Field(Enum):
        """Represent a field in a patent APS document."""

        DOC_NUMBER = "WKU"
        TITLE = "TTL"
        PARAGRAPH = "PAR"
        PARAGRAPH_1 = "PA1"
        PARAGRAPH_2 = "PA2"
        PARAGRAPH_3 = "PA3"
        TEXT = "PAL"
        CAPTION = "PAC"
        NUMBER = "NUM"
        NAME = "NAM"
        IPC = "ICL"
        ISSUED = "ISD"
        FILED = "APD"
        PATENT_NUMBER = "PNO"
        APPLICATION_NUMBER = "APN"
        APPLICATION_TYPE = "APT"
        COUNTRY = "CNT"

    @override
    def __init__(self) -> None:
        """Build an instance of PatentUsptoGrantAps class."""
        self.doc: Optional[DoclingDocument] = None
        # Keep track of docling hierarchy level
        self.level: LevelNumber = 1
        # Keep track of docling parents by level
        self.parents: dict[LevelNumber, Optional[DocItem]] = {1: None}

    def get_last_text_item(self) -> Optional[TextItem]:
        """Get the last text item at the current document level.

        Returns:
            The text item or None, if the current level parent has no children."""
        if self.doc:
            parent = self.parents[self.level]
            children = parent.children if parent is not None else []
        else:
            return None
        text_list: list[TextItem] = [
            item
            for item in self.doc.texts
            if isinstance(item, TextItem) and item.get_ref() in children
        ]

        if text_list:
            return text_list[-1]
        else:
            return None

    def store_section(self, section: str) -> None:
        """Store the section heading in the docling document.

        Only the predefined sections from PatentHeading will be handled.
        The other sections are created by the Field.CAPTION field.

        Args:
            section: A patent section name."""
        heading: PatentHeading
        if self.doc is None:
            return
        elif section == self.Section.ABSTRACT.value:
            heading = PatentHeading.ABSTRACT
        elif section == self.Section.CLAIMS.value:
            heading = PatentHeading.CLAIMS
        else:
            return None

        self.level = heading.level if heading.level in self.parents else 1
        self.parents[self.level + 1] = self.doc.add_heading(
            heading.value,
            level=self.level,
            parent=self.parents[self.level],
        )
        self.level += 1

    def store_content(self, section: str, field: str, value: str) -> None:
        """Store the key value within a document section in the docling document.

        Args:
            section: A patent section name.
            field: A field name.
            value: A field value name.
        """
        if (
            not self.doc
            or not field
            or field not in [item.value for item in PatentUsptoGrantAps.Field]
        ):
            return

        if field == self.Field.TITLE.value:
            self.parents[self.level + 1] = self.doc.add_title(
                parent=self.parents[self.level], text=value
            )
            self.level += 1

        elif field == self.Field.TEXT.value and section == self.Section.ABSTRACT.value:
            abst_item = self.get_last_text_item()
            if abst_item:
                abst_item.text += " " + value
            else:
                self.doc.add_text(
                    label=DocItemLabel.PARAGRAPH,
                    text=value,
                    parent=self.parents[self.level],
                )

        elif field == self.Field.NUMBER.value and section == self.Section.CLAIMS.value:
            self.doc.add_text(
                label=DocItemLabel.PARAGRAPH,
                text="",
                parent=self.parents[self.level],
            )

        elif (
            field
            in (
                self.Field.PARAGRAPH.value,
                self.Field.PARAGRAPH_1.value,
                self.Field.PARAGRAPH_2.value,
                self.Field.PARAGRAPH_3.value,
            )
            and section == self.Section.CLAIMS.value
        ):
            last_claim = self.get_last_text_item()
            if last_claim is None:
                last_claim = self.doc.add_text(
                    label=DocItemLabel.PARAGRAPH,
                    text="",
                    parent=self.parents[self.level],
                )

            last_claim.text += f" {value.strip()}" if last_claim.text else value.strip()

        elif field == self.Field.CAPTION.value and section in (
            self.Section.SUMMARY.value,
            self.Section.DETAILS.value,
            self.Section.DRAWINGS.value,
        ):
            # captions are siblings of abstract since no level info is provided
            head_item = PatentHeading.ABSTRACT
            self.level = head_item.level if head_item.level in self.parents else 1
            self.parents[self.level + 1] = self.doc.add_heading(
                value,
                level=self.level,
                parent=self.parents[self.level],
            )
            self.level += 1

        elif field in (
            self.Field.PARAGRAPH.value,
            self.Field.PARAGRAPH_1.value,
            self.Field.PARAGRAPH_2.value,
            self.Field.PARAGRAPH_3.value,
        ) and section in (
            self.Section.SUMMARY.value,
            self.Section.DETAILS.value,
            self.Section.DRAWINGS.value,
        ):
            self.doc.add_text(
                label=DocItemLabel.PARAGRAPH,
                text=value,
                parent=self.parents[self.level],
            )

    def parse(self, patent_content: str) -> Optional[DoclingDocument]:
        self.doc = self.doc = DoclingDocument(name="file")
        section: str = ""
        key: str = ""
        value: str = ""
        line_num = 0
        for line in patent_content.splitlines():
            cols = re.split("\\s{2,}", line, maxsplit=1)
            if key and value and (len(cols) == 1 or (len(cols) == 2 and cols[0])):
                self.store_content(section, key, value)
                key = ""
                value = ""
            if len(cols) == 1:  # section title
                section = cols[0]
                self.store_section(section)
                _log.debug(f"Parsing section {section}")
            elif len(cols) == 2:  # key value
                if cols[0]:  # key present
                    key = cols[0]
                    value = cols[1]
                elif not re.match(r"^##STR\d+##$", cols[1]):  # line continues
                    value += " " + cols[1]
            line_num += 1
        if key and value:
            self.store_content(section, key, value)

        # TODO: parse tables
        return self.doc


class PatentUsptoAppV1(PatentUspto):
    """Parser of patent documents from the US Patent Office (applications v1.x)

    The compatible format is:
    - Patent Application Full Text Data/XML Version 1.x (from March 2001 till December
      2004)
    """

    @override
    def __init__(self) -> None:
        """Build an instance of PatentUsptoAppV1 class."""
        self.handler = PatentUsptoAppV1.PatentHandler()
        self.pattern = re.compile(r"^(<table .*?</table>)", re.MULTILINE | re.DOTALL)

    @override
    def parse(self, patent_content: str) -> Optional[DoclingDocument]:
        try:
            xml.sax.parseString(patent_content, self.handler)
        except xml.sax._exceptions.SAXParseException as exc_sax:
            _log.error(f"Error in parsing USPTO document: {exc_sax}")

            return None

        doc = self.handler.doc
        if doc:
            raw_tables = re.findall(self.pattern, patent_content)
            parsed_tables: list[TableData] = []
            _log.debug(f"Found {len(raw_tables)} tables to be parsed with XmlTable.")
            for table in raw_tables:
                table_parser = XmlTable(XML_DECLARATION + "\n" + table)
                try:
                    table_data = table_parser.parse()
                    if table_data:
                        parsed_tables.append(table_data)
                except Exception as exc_table:
                    _log.error(f"Error in parsing USPTO tables: {exc_table}")
            if len(parsed_tables) != len(doc.tables):
                _log.error(
                    f"Number of referenced ({len(doc.tables)}) and parsed "
                    f"({len(parsed_tables)}) tables differ."
                )
            else:
                for idx, item in enumerate(parsed_tables):
                    doc.tables[idx].data = item

        return doc

    class PatentHandler(xml.sax.handler.ContentHandler):
        """SAX ContentHandler for patent documents."""

        APP_DOC_ELEMENT: Final = "patent-application-publication"

        @unique
        class Element(Enum):
            """Represents an element of interest in the patent application document."""

            DRAWINGS = "brief-description-of-drawings", False
            ABSTRACT = "subdoc-abstract", False
            TITLE = "title-of-invention", True
            CLAIMS = "subdoc-claims", False
            CLAIM = "claim", False
            CLAIM_TEXT = "claim-text", True
            NUMBER = ("number", False)
            PARAGRAPH = "paragraph", True
            HEADING = "heading", True
            STYLE_SUPERSCRIPT = "superscript", True
            STYLE_SUBSCRIPT = "subscript", True
            # do not store text of a table, since it can be within paragraph
            TABLE = "table", False
            # do not store text of a formula, since it can be within paragraph
            MATH = "math-cwu", False

            @override
            def __new__(cls, value: str, _) -> Self:
                obj = object.__new__(cls)
                obj._value_ = value
                return obj

            @override
            def __init__(self, _, is_text: bool) -> None:
                self.is_text: bool = is_text

        @override
        def __init__(self) -> None:
            """Build an instance of the patent handler."""
            # Current patent being parsed
            self.doc: Optional[DoclingDocument] = None
            # Keep track of docling hierarchy level
            self.level: LevelNumber = 1
            # Keep track of docling parents by level
            self.parents: dict[LevelNumber, Optional[DocItem]] = {1: None}
            # Content to retain for the current patent
            self.property: list[str]
            self.claim: str
            self.claims: list[str]
            self.abstract: str
            self.text: str
            self._clean_data()
            # To handle mathematical styling
            self.style_html = HtmlEntity()

        @override
        def startElement(self, tag, attributes):
            """Signal the start of an element.

            Args:
                tag: The element tag.
                attributes: The element attributes.
            """
            if tag == self.APP_DOC_ELEMENT:
                self.doc = DoclingDocument(name="file")
                self.text = ""
            self._start_registered_elements(tag, attributes)

        @override
        def skippedEntity(self, name):
            """Receive notification of a skipped entity.

            HTML entities will be skipped by the parser. This method will unescape them
            and add them to the text.

            Args:
                name: Entity name.
            """
            if self.property:
                elm_val = self.property[-1]
                element = self.Element(elm_val)
                if element.is_text:
                    escaped = self.style_html.get_greek_from_iso8879(f"&{name};")
                    unescaped = html.unescape(escaped)
                    if unescaped == escaped:
                        logging.debug("Unrecognized HTML entity: " + name)
                        return

                    if element in (
                        self.Element.STYLE_SUPERSCRIPT,
                        self.Element.STYLE_SUBSCRIPT,
                    ):
                        # superscripts and subscripts need to be under text elements
                        if len(self.property) < 2:
                            return
                        parent_val = self.property[-2]
                        parent = self.Element(parent_val)
                        if parent.is_text:
                            self.text += self._apply_style(unescaped, elm_val)
                    else:
                        self.text += unescaped

        @override
        def endElement(self, tag):
            """Signal the end of an element.

            Args:
                tag: The element tag.
            """
            if tag == self.APP_DOC_ELEMENT:
                self._clean_data()
            self._end_registered_element(tag)

        @override
        def characters(self, content):
            """Receive notification of character data.

            Args:
                content: Data reported by the handler.
            """
            if self.property:
                elm_val = self.property[-1]
                element = self.Element(elm_val)
                if element.is_text:
                    if element in (
                        self.Element.STYLE_SUPERSCRIPT,
                        self.Element.STYLE_SUBSCRIPT,
                    ):
                        # superscripts and subscripts need to be under text elements
                        if len(self.property) < 2:
                            return
                        parent_val = self.property[-2]
                        parent = self.Element(parent_val)
                        if parent.is_text:
                            self.text += self._apply_style(content, elm_val)
                    else:
                        self.text += content

        def _start_registered_elements(
            self, tag: str, attributes: xml.sax.xmlreader.AttributesImpl
        ) -> None:
            if tag in [member.value for member in self.Element]:
                # special case for claims: claim lines may start before the
                # previous one is closed
                if (
                    tag == self.Element.CLAIM_TEXT.value
                    and self.property
                    and self.property[-1] == tag
                    and self.text.strip()
                ):
                    self.claim += " " + self.text.strip("\n")
                    self.text = ""
                elif tag == self.Element.HEADING.value:
                    level_attr: str = attributes.get("lvl", "")
                    new_level: int = int(level_attr) if level_attr.isnumeric() else 1
                    max_level = min(self.parents.keys())
                    # increase heading level with 1 for title, if any
                    self.level = (
                        new_level + 1 if (new_level + 1) in self.parents else max_level
                    )
                self.property.append(tag)

        def _end_registered_element(self, tag: str) -> None:
            if tag in [elm.value for elm in self.Element] and self.property:
                current_tag = self.property.pop()
                self._add_property(current_tag, self.text)

        def _add_property(self, name: str, text: str) -> None:
            if not name or not self.doc:
                return

            if name == self.Element.TITLE.value:
                title = text.strip()
                if title:
                    self.parents[self.level + 1] = self.doc.add_text(
                        parent=self.parents[self.level],
                        label=DocItemLabel.TITLE,
                        text=title,
                    )
                    self.level += 1
                self.text = ""
            elif name == self.Element.ABSTRACT.value:
                abstract = self.abstract.strip()
                if abstract:
                    heading_text = PatentHeading.ABSTRACT.value
                    heading_level = (
                        PatentHeading.ABSTRACT.level
                        if PatentHeading.ABSTRACT.level in self.parents
                        else 1
                    )
                    abstract_item = self.doc.add_heading(
                        heading_text,
                        level=heading_level,
                        parent=self.parents[heading_level],
                    )
                    self.doc.add_text(
                        label=DocItemLabel.PARAGRAPH,
                        text=self.abstract,
                        parent=abstract_item,
                    )
                    self.abstract = ""
                self.text = ""
            elif name == self.Element.CLAIM_TEXT.value:
                if text:
                    self.claim += self.text.strip("\n")
                self.text = ""

            elif name == self.Element.CLAIM.value:
                claim = self.claim.strip()
                if claim:
                    self.claims.append(claim)
                self.claim = ""

            elif name == self.Element.CLAIMS.value and self.claims:
                heading_text = PatentHeading.CLAIMS.value
                heading_level = (
                    PatentHeading.CLAIMS.level
                    if PatentHeading.CLAIMS.level in self.parents
                    else 1
                )
                claims_item = self.doc.add_heading(
                    heading_text,
                    level=heading_level,
                    parent=self.parents[heading_level],
                )
                for text in self.claims:
                    self.doc.add_text(
                        label=DocItemLabel.PARAGRAPH, text=text, parent=claims_item
                    )

            elif name in (
                self.Element.PARAGRAPH.value,
                self.Element.HEADING.value,
            ):
                if text and self.Element.ABSTRACT.value in self.property:
                    self.abstract = (self.abstract + text) if self.abstract else text
                elif text.strip():
                    text = re.sub("\\s+", " ", text).strip()
                    if name == self.Element.HEADING.value:
                        self.parents[self.level + 1] = self.doc.add_heading(
                            text=text,
                            level=self.level,
                            parent=self.parents[self.level],
                        )
                        self.level += 1
                    else:
                        self.doc.add_text(
                            label=DocItemLabel.PARAGRAPH,
                            text=text,
                            parent=self.parents[self.level],
                        )
                self.text = ""

            elif name == self.Element.TABLE.value:
                # set an empty table as placeholder
                empty_table = TableData(num_rows=0, num_cols=0, table_cells=[])
                self.doc.add_table(
                    data=empty_table,
                    parent=self.parents[self.level],
                )

        def _apply_style(self, text: str, style_tag: str) -> str:
            """Apply an HTML style to text.

            Args:
                text: A string containing plain text.
                style_tag: An HTML tag name for styling text. If the tag name is not
                  recognized as one of the supported styles, the method will return
                  the original `text`.

            Returns:
                A string after applying the style.
            """
            formatted = html.unescape(text)

            if style_tag == self.Element.STYLE_SUPERSCRIPT.value:
                formatted = html.unescape(self.style_html.get_superscript(formatted))
            elif style_tag == self.Element.STYLE_SUBSCRIPT.value:
                formatted = html.unescape(self.style_html.get_subscript(formatted))

            return formatted

        def _clean_data(self):
            """Reset the variables from stream data."""
            self.property = []
            self.abstract = ""
            self.claim = ""
            self.claims = []
            self.text = ""


class XmlTable:
    """Provide a table parser for xml tables in USPTO patent documents.

    The OASIS Open XML Exchange Table Model can be downloaded from:
    http://oasis-open.org/specs/soextblx.dtd
    """

    class ColInfo(TypedDict):
        ncols: int
        colinfo: list[dict]

    class MinColInfoType(TypedDict):
        offset: list[int]
        colwidth: list[int]

    class ColInfoType(MinColInfoType):
        cell_range: list[int]
        cell_offst: list[int]

    def __init__(self, input: str) -> None:
        """Initialize the table parser with the xml content.

        Args:
            input: The xml content.
        """
        self.max_nbr_messages = 2
        self.nbr_messages = 0
        self.empty_text = ""
        self._soup = BeautifulSoup(input, features="xml")

    def _create_tg_range(self, tgs: list[ColInfo]) -> dict[int, ColInfoType]:
        """Create a unified range along the table groups.

        Args:
            tgs: Table group column specifications.

        Returns:
            Unified group column specifications.
        """
        colinfo: dict[int, XmlTable.ColInfoType] = {}

        if len(tgs) == 0:
            return colinfo

        for itg, tg in enumerate(tgs):
            colinfo[itg] = {
                "offset": [],
                "colwidth": [],
                "cell_range": [],
                "cell_offst": [0],
            }
            offst = 0
            for info in tg["colinfo"]:
                cw = info["colwidth"]
                cw = re.sub("pt", "", cw, flags=re.I)
                cw = re.sub("mm", "", cw, flags=re.I)
                try:
                    cw = int(cw)
                except BaseException:
                    cw = float(cw)
                colinfo[itg]["colwidth"].append(cw)
                colinfo[itg]["offset"].append(offst)
                offst += cw
            colinfo[itg]["offset"].append(offst)

        min_colinfo: XmlTable.MinColInfoType = {"offset": [], "colwidth": []}

        min_colinfo["offset"] = colinfo[0]["offset"]
        offset_w0 = []
        for itg, col in colinfo.items():
            # keep track of col with 0 width
            for ic, cw in enumerate(col["colwidth"]):
                if cw == 0:
                    offset_w0.append(col["offset"][ic])

            min_colinfo["offset"] = sorted(set(col["offset"] + min_colinfo["offset"]))

        # add back the 0 width cols to offset list
        offset_w0 = list(set(offset_w0))
        min_colinfo["offset"] = sorted(min_colinfo["offset"] + offset_w0)

        for i in range(len(min_colinfo["offset"]) - 1):
            min_colinfo["colwidth"].append(
                min_colinfo["offset"][i + 1] - min_colinfo["offset"][i]
            )

        for itg, col in colinfo.items():
            i = 1
            range_ = 1
            for min_i in range(1, len(min_colinfo["offset"])):
                min_offst = min_colinfo["offset"][min_i]
                offst = col["offset"][i]
                if min_offst == offst:
                    if (
                        len(col["offset"]) == i + 1
                        and len(min_colinfo["offset"]) > min_i + 1
                    ):
                        range_ += 1
                    else:
                        col["cell_range"].append(range_)
                        col["cell_offst"].append(col["cell_offst"][-1] + range_)
                        range_ = 1
                        i += 1
                elif min_offst < offst:
                    range_ += 1
                else:
                    _log.debug("A USPTO XML table has wrong offsets.")
                    return {}

        return colinfo

    def _get_max_ncols(self, tgs_info: dict[int, ColInfoType]) -> NonNegativeInt:
        """Get the maximum number of columns across table groups.

        Args:
            tgs_info: Unified group column specifications.

        Return:
            The maximum number of columns.
        """
        ncols_max = 0
        for rowinfo in tgs_info.values():
            ncols_max = max(ncols_max, len(rowinfo["colwidth"]))

        return ncols_max

    def _parse_table(self, table: Tag) -> TableData:  # noqa: C901
        """Parse the content of a table tag.

        Args:
            The table element.

        Returns:
            A docling table object.
        """
        tgs_align: list[XmlTable.ColInfo] = []
        tg_secs = table("tgroup")
        if tg_secs:
            for tg_sec in tg_secs:
                if not isinstance(tg_sec, Tag):
                    continue
                col_val = tg_sec.get("cols")
                ncols = (
                    int(col_val)
                    if isinstance(col_val, str) and col_val.isnumeric()
                    else 1
                )
                tg_align: XmlTable.ColInfo = {"ncols": ncols, "colinfo": []}
                cs_secs = tg_sec("colspec")
                if cs_secs:
                    for cs_sec in cs_secs:
                        if not isinstance(cs_sec, Tag):
                            continue
                        colname = cs_sec.get("colname")
                        colwidth = cs_sec.get("colwidth")
                        tg_align["colinfo"].append(
                            {"colname": colname, "colwidth": colwidth}
                        )

                tgs_align.append(tg_align)

        # create unified range along the table groups
        tgs_range = self._create_tg_range(tgs_align)

        # if the structure is broken, return an empty table
        if not tgs_range:
            dl_table = TableData(num_rows=0, num_cols=0, table_cells=[])
            return dl_table

        ncols_max = self._get_max_ncols(tgs_range)

        # extract table data
        table_data: list[TableCell] = []
        i_row_global = 0
        is_row_empty: bool = True
        tg_secs = table("tgroup")
        if tg_secs:
            for itg, tg_sec in enumerate(tg_secs):
                if not isinstance(tg_sec, Tag):
                    continue
                tg_range = tgs_range[itg]
                row_secs = tg_sec(["row", "tr"])

                if row_secs:
                    for row_sec in row_secs:
                        if not isinstance(row_sec, Tag):
                            continue
                        entry_secs = row_sec(["entry", "td"])
                        is_header: bool = (
                            row_sec.parent is not None
                            and row_sec.parent.name == "thead"
                        )

                        ncols = 0
                        local_row: list[TableCell] = []
                        is_row_empty = True
                        if entry_secs:
                            wrong_nbr_cols = False
                            for ientry, entry_sec in enumerate(entry_secs):
                                if not isinstance(entry_sec, Tag):
                                    continue
                                text = entry_sec.get_text().strip()

                                # start-end
                                namest = entry_sec.get("namest")
                                nameend = entry_sec.get("nameend")
                                start = (
                                    int(namest)
                                    if isinstance(namest, str) and namest.isnumeric()
                                    else ientry + 1
                                )
                                if isinstance(nameend, str) and nameend.isnumeric():
                                    end = int(nameend)
                                    shift = 0
                                else:
                                    end = ientry + 2
                                    shift = 1

                                if end > len(tg_range["cell_offst"]):
                                    wrong_nbr_cols = True
                                    self.nbr_messages += 1
                                    if self.nbr_messages <= self.max_nbr_messages:
                                        _log.debug(
                                            "USPTO table has # entries != # columns"
                                        )
                                    break

                                range_ = [
                                    tg_range["cell_offst"][start - 1],
                                    tg_range["cell_offst"][end - 1] - shift,
                                ]

                                # add row and replicate cell if needed
                                cell_text = text if text else self.empty_text
                                if cell_text != self.empty_text:
                                    is_row_empty = False
                                for irep in range(range_[0], range_[1] + 1):
                                    ncols += 1
                                    local_row.append(
                                        TableCell(
                                            column_header=is_header,
                                            text=cell_text,
                                            start_row_offset_idx=i_row_global,
                                            end_row_offset_idx=i_row_global + 1,
                                            row_span=1,
                                            start_col_offset_idx=range_[0],
                                            end_col_offset_idx=range_[1] + 1,
                                            col_span=range_[1] - range_[0] + 1,
                                        )
                                    )

                            if wrong_nbr_cols:
                                # keep empty text, not to introduce noise
                                local_row = []
                                ncols = 0

                            # add empty cell up to ncols_max
                            for irep in range(ncols, ncols_max):
                                local_row.append(
                                    TableCell(
                                        column_header=is_header,
                                        text=self.empty_text,
                                        start_row_offset_idx=i_row_global,
                                        end_row_offset_idx=i_row_global + 1,
                                        row_span=1,
                                        start_col_offset_idx=irep,
                                        end_col_offset_idx=irep + 1,
                                        col_span=1,
                                    )
                                )
                        # do not add empty rows
                        if not is_row_empty:
                            table_data.extend(local_row)
                            i_row_global += 1

        dl_table = TableData(
            num_rows=i_row_global, num_cols=ncols_max, table_cells=table_data
        )

        return dl_table

    def parse(self) -> Optional[TableData]:
        """Parse the first table from an xml content.

        Returns:
            A docling table data.
        """
        section = self._soup.find("table")
        if isinstance(section, Tag):
            table = self._parse_table(section)
            if table.num_rows == 0 or table.num_cols == 0:
                _log.warning("The parsed USPTO table is empty")
            return table
        else:
            return None


class HtmlEntity:
    """Provide utility functions to get the HTML entities of styled characters.

    This class has been developed from:
    https://unicode-table.com/en/html-entities/
    https://www.w3.org/TR/WD-math-970515/table03.html
    """

    def __init__(self):
        """Initialize this class by loading the HTML entity dictionaries."""
        self.superscript = str.maketrans(
            {
                "1": "&sup1;",
                "2": "&sup2;",
                "3": "&sup3;",
                "4": "&#8308;",
                "5": "&#8309;",
                "6": "&#8310;",
                "7": "&#8311;",
                "8": "&#8312;",
                "9": "&#8313;",
                "0": "&#8304;",
                "+": "&#8314;",
                "-": "&#8315;",
                "−": "&#8315;",  # noqa: RUF001
                "=": "&#8316;",
                "(": "&#8317;",
                ")": "&#8318;",
                "a": "&#170;",
                "o": "&#186;",
                "i": "&#8305;",
                "n": "&#8319;",
            }
        )
        self.subscript = str.maketrans(
            {
                "1": "&#8321;",
                "2": "&#8322;",
                "3": "&#8323;",
                "4": "&#8324;",
                "5": "&#8325;",
                "6": "&#8326;",
                "7": "&#8327;",
                "8": "&#8328;",
                "9": "&#8329;",
                "0": "&#8320;",
                "+": "&#8330;",
                "-": "&#8331;",
                "−": "&#8331;",  # noqa: RUF001
                "=": "&#8332;",
                "(": "&#8333;",
                ")": "&#8334;",
                "a": "&#8336;",
                "e": "&#8337;",
                "o": "&#8338;",
                "x": "&#8339;",
            }
        )
        self.mathematical_italic = str.maketrans(
            {
                "A": "&#119860;",
                "B": "&#119861;",
                "C": "&#119862;",
                "D": "&#119863;",
                "E": "&#119864;",
                "F": "&#119865;",
                "G": "&#119866;",
                "H": "&#119867;",
                "I": "&#119868;",
                "J": "&#119869;",
                "K": "&#119870;",
                "L": "&#119871;",
                "M": "&#119872;",
                "N": "&#119873;",
                "O": "&#119874;",
                "P": "&#119875;",
                "Q": "&#119876;",
                "R": "&#119877;",
                "S": "&#119878;",
                "T": "&#119879;",
                "U": "&#119880;",
                "V": "&#119881;",
                "W": "&#119882;",
                "Y": "&#119884;",
                "Z": "&#119885;",
                "a": "&#119886;",
                "b": "&#119887;",
                "c": "&#119888;",
                "d": "&#119889;",
                "e": "&#119890;",
                "f": "&#119891;",
                "g": "&#119892;",
                "h": "&#119893;",
                "i": "&#119894;",
                "j": "&#119895;",
                "k": "&#119896;",
                "l": "&#119897;",
                "m": "&#119898;",
                "n": "&#119899;",
                "o": "&#119900;",
                "p": "&#119901;",
                "q": "&#119902;",
                "r": "&#119903;",
                "s": "&#119904;",
                "t": "&#119905;",
                "u": "&#119906;",
                "v": "&#119907;",
                "w": "&#119908;",
                "x": "&#119909;",
                "y": "&#119910;",
                "z": "&#119911;",
            }
        )

        self.lookup_iso8879 = {
            "&Agr;": "&Alpha;",
            "&Bgr;": "&Beta;",
            "&Ggr;": "&Gamma;",
            "&Dgr;": "&Delta;",
            "&Egr;": "&Epsilon;",
            "&Zgr;": "&Zeta;",
            "&EEgr;": "&Eta;",
            "&THgr;": "&Theta;",
            "&Igr;": "&Iota;",
            "&Kgr;": "&Kappa;",
            "&Lgr;": "&Lambda;",
            "&Mgr;": "&Mu;",
            "&Ngr;": "&Nu;",
            "&Xgr;": "&Xi;",
            "&Ogr;": "&Omicron;",
            "&Pgr;": "&Pi;",
            "&Rgr;": "&Rho;",
            "&Sgr;": "&Sigma;",
            "&Tgr;": "&Tau;",
            "&Ugr;": "&Upsilon;",
            "&PHgr;": "&Phi;",
            "&KHgr;": "&Chi;",
            "&PSgr;": "&Psi;",
            "&OHgr;": "&Omega;",
            "&agr;": "&alpha;",
            "&bgr;": "&beta;",
            "&ggr;": "&gamma;",
            "&dgr;": "&delta;",
            "&egr;": "&epsilon;",
            "&zgr;": "&zeta;",
            "&eegr;": "&eta;",
            "&thgr;": "&theta;",
            "&igr;": "&iota;",
            "&kgr;": "&kappa;",
            "&lgr;": "&lambda;",
            "&mgr;": "&mu;",
            "&ngr;": "&nu;",
            "&xgr;": "&xi;",
            "&ogr;": "&omicron;",
            "&pgr;": "&pi;",
            "&rgr;": "&rho;",
            "&sgr;": "&sigmaf;",
            "&tgr;": "&tau;",
            "&ugr;": "&upsilon;",
            "&phgr;": "&phi;",
            "&khgr;": "&chi;",
            "&psgr;": "&psi;",
            "&ohgr;": "&omega;",
        }

    def get_superscript(self, text: str) -> str:
        """Get a text in superscript as HTML entities.

        Args:
            text: The text to transform.

        Returns:
            The text in superscript as HTML entities.
        """
        return text.translate(self.superscript)

    def get_subscript(self, text: str) -> str:
        """Get a text in subscript as HTML entities.

        Args:
            The text to transform.

        Returns:
            The text in subscript as HTML entities.
        """
        return text.translate(self.subscript)

    def get_math_italic(self, text: str) -> str:
        """Get a text in italic as HTML entities.

        Args:
            The text to transform.

        Returns:
            The text in italics as HTML entities.
        """
        return text.translate(self.mathematical_italic)

    def get_greek_from_iso8879(self, text: str) -> str:
        """Get an HTML entity of a greek letter in ISO 8879.

        Args:
            The text to transform, as an ISO 8879 entity.

        Returns:
            The HTML entity representing a greek letter. If the input text is not
              supported, the original text is returned.
        """
        return self.lookup_iso8879.get(text, text)
