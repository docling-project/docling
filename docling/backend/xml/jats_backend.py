import logging
import traceback
from io import BytesIO
from pathlib import Path
from typing import Final, Optional, Union

from bs4 import BeautifulSoup, Tag
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupItem,
    GroupLabel,
    NodeItem,
    TextItem,
)
from lxml import etree
from typing_extensions import TypedDict, override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)

JATS_DTD_URL: Final = ["JATS-journalpublishing", "JATS-archive"]
DEFAULT_HEADER_ACKNOWLEDGMENTS: Final = "Acknowledgments"
DEFAULT_HEADER_ABSTRACT: Final = "Abstract"
DEFAULT_HEADER_REFERENCES: Final = "References"
DEFAULT_TEXT_ETAL: Final = "et al."


class Abstract(TypedDict):
    label: str
    content: str


class Author(TypedDict):
    name: str
    affiliation_names: list[str]


class Citation(TypedDict):
    author_names: str
    title: str
    source: str
    year: str
    volume: str
    page: str
    pub_id: str
    publisher_name: str
    publisher_loc: str


class Table(TypedDict):
    label: str
    caption: str
    content: str


class XMLComponents(TypedDict):
    title: str
    authors: list[Author]
    abstract: list[Abstract]


class JatsDocumentBackend(DeclarativeDocumentBackend):
    """Backend to parse articles in XML format tagged according to JATS definition.

    The Journal Article Tag Suite (JATS) is an definition standard for the
    representation of journal articles in XML format. Several publishers and journal
    archives provide content in JATS format, including PubMed Central® (PMC), bioRxiv,
    medRxiv, or Springer Nature.

    Refer to https://jats.nlm.nih.gov for more details on JATS.

    The code from this document backend has been developed by modifying parts of the
    PubMed Parser library (version 0.5.0, released on 12.08.2024):
    Achakulvisut et al., (2020).
    Pubmed Parser: A Python Parser for PubMed Open-Access XML Subset and MEDLINE XML
      Dataset XML Dataset.
    Journal of Open Source Software, 5(46), 1979,
    https://doi.org/10.21105/joss.01979
    """

    @override
    def __init__(
        self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]
    ) -> None:
        super().__init__(in_doc, path_or_stream)
        self.path_or_stream = path_or_stream

        # Initialize the root of the document hierarchy
        self.root: Optional[NodeItem] = None

        self.valid = False
        try:
            if isinstance(self.path_or_stream, BytesIO):
                self.path_or_stream.seek(0)
            self.tree: etree._ElementTree = etree.parse(self.path_or_stream)

            doc_info: etree.DocInfo = self.tree.docinfo
            if doc_info.system_url and any(
                kwd in doc_info.system_url for kwd in JATS_DTD_URL
            ):
                self.valid = True
                return
            for ent in doc_info.internalDTD.iterentities():
                if ent.system_url and any(
                    kwd in ent.system_url for kwd in JATS_DTD_URL
                ):
                    self.valid = True
                    return
        except Exception as exc:
            raise RuntimeError(
                f"Could not initialize JATS backend for file with hash {self.document_hash}."
            ) from exc

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
        return {InputFormat.XML_JATS}

    @override
    def convert(self) -> DoclingDocument:
        try:
            # Create empty document
            origin = DocumentOrigin(
                filename=self.file.name or "file",
                mimetype="application/xml",
                binary_hash=self.document_hash,
            )
            doc = DoclingDocument(name=self.file.stem or "file", origin=origin)

            # Get metadata XML components
            xml_components: XMLComponents = self._parse_metadata()

            # Add metadata to the document
            self._add_metadata(doc, xml_components)

            # walk over the XML body
            body = self.tree.xpath("//body")
            if self.root and len(body) > 0:
                self._walk_linear(doc, self.root, body[0])

            # walk over the XML back matter
            back = self.tree.xpath("//back")
            if self.root and len(back) > 0:
                self._walk_linear(doc, self.root, back[0])
        except Exception:
            _log.error(traceback.format_exc())

        return doc

    @staticmethod
    def _get_text(node: etree._Element, sep: Optional[str] = None) -> str:
        skip_tags = ["term", "disp-formula", "inline-formula"]
        text: str = (
            node.text.replace("\n", " ")
            if (node.tag not in skip_tags and node.text)
            else ""
        )
        for child in list(node):
            if child.tag not in skip_tags:
                # TODO: apply styling according to child.tag when supported by docling-core
                text += JatsDocumentBackend._get_text(child, sep)
            if sep:
                text = text.rstrip(sep) + sep
            text += child.tail.replace("\n", " ") if child.tail else ""

        return text

    def _find_metadata(self) -> Optional[etree._Element]:
        meta_names: list[str] = ["article-meta", "book-part-meta"]
        meta: Optional[etree._Element] = None
        for name in meta_names:
            node = self.tree.xpath(f".//{name}")
            if len(node) > 0:
                meta = node[0]
                break

        return meta

    def _parse_abstract(self) -> list[Abstract]:
        # TODO: address cases with multiple sections
        abs_list: list[Abstract] = []

        for abs_node in self.tree.xpath(".//abstract"):
            abstract: Abstract = dict(label="", content="")
            texts = []
            for abs_par in abs_node.xpath("p"):
                texts.append(JatsDocumentBackend._get_text(abs_par).strip())
            abstract["content"] = " ".join(texts)

            label_node = abs_node.xpath("title|label")
            if len(label_node) > 0:
                abstract["label"] = label_node[0].text.strip()

            abs_list.append(abstract)

        return abs_list

    def _parse_authors(self) -> list[Author]:
        # Get mapping between affiliation ids and names
        authors: list[Author] = []
        meta: Optional[etree._Element] = self._find_metadata()
        if meta is None:
            return authors

        affiliation_names = []
        for affiliation_node in meta.xpath(".//aff[@id]"):
            aff = ", ".join([t for t in affiliation_node.itertext() if t.strip()])
            aff = aff.replace("\n", " ")
            label = affiliation_node.xpath("label")
            if label:
                # TODO: once superscript is supported, add label with formatting
                aff = aff.removeprefix(f"{label[0].text}, ")
            affiliation_names.append(aff)
        affiliation_ids_names = dict(
            zip(meta.xpath(".//aff[@id]/@id"), affiliation_names)
        )

        # Get author names and affiliation names
        for author_node in meta.xpath(
            './/contrib-group/contrib[@contrib-type="author"]'
        ):
            author: Author = {
                "name": "",
                "affiliation_names": [],
            }

            # Affiliation names
            affiliation_ids = [
                a.attrib["rid"] for a in author_node.xpath('xref[@ref-type="aff"]')
            ]
            for id in affiliation_ids:
                if id in affiliation_ids_names:
                    author["affiliation_names"].append(affiliation_ids_names[id])

            # Name
            author["name"] = (
                author_node.xpath("name/given-names")[0].text
                + " "
                + author_node.xpath("name/surname")[0].text
            )

            authors.append(author)

        return authors

    def _parse_title(self) -> str:
        meta_names: list[str] = [
            "article-meta",
            "collection-meta",
            "book-meta",
            "book-part-meta",
        ]
        title_names: list[str] = ["article-title", "subtitle", "title", "label"]
        titles: list[str] = [
            " ".join(
                elem.text.replace("\n", " ").strip()
                for elem in list(title_node)
                if elem.tag in title_names
            ).strip()
            for title_node in self.tree.xpath(
                "|".join([f".//{item}/title-group" for item in meta_names])
            )
        ]

        text = " - ".join(titles)

        return text

    def _parse_metadata(self) -> XMLComponents:
        """Parsing JATS document metadata."""
        xml_components: XMLComponents = {
            "title": self._parse_title(),
            "authors": self._parse_authors(),
            "abstract": self._parse_abstract(),
        }
        return xml_components

    def _add_abstract(
        self, doc: DoclingDocument, xml_components: XMLComponents
    ) -> None:
        for abstract in xml_components["abstract"]:
            text: str = abstract["content"]
            title: str = abstract["label"] or DEFAULT_HEADER_ABSTRACT
            if not text:
                continue
            parent = doc.add_heading(parent=self.root, text=title)
            doc.add_text(
                parent=parent,
                text=text,
                label=DocItemLabel.TEXT,
            )

        return

    def _add_authors(self, doc: DoclingDocument, xml_components: XMLComponents) -> None:
        # TODO: once docling supports text formatting, add affiliation reference to
        # author names through superscripts
        authors: list = [item["name"] for item in xml_components["authors"]]
        authors_str = ", ".join(authors)
        affiliations: list = [
            item
            for author in xml_components["authors"]
            for item in author["affiliation_names"]
        ]
        affiliations_str = "; ".join(list(dict.fromkeys(affiliations)))
        if authors_str:
            doc.add_text(
                parent=self.root,
                text=authors_str,
                label=DocItemLabel.PARAGRAPH,
            )
        if affiliations_str:
            doc.add_text(
                parent=self.root,
                text=affiliations_str,
                label=DocItemLabel.PARAGRAPH,
            )

        return

    def _add_citation(self, doc: DoclingDocument, parent: NodeItem, text: str) -> None:
        if isinstance(parent, GroupItem) and parent.label == GroupLabel.LIST:
            doc.add_list_item(text=text, enumerated=False, parent=parent)
        else:
            doc.add_text(text=text, label=DocItemLabel.TEXT, parent=parent)

        return

    def _parse_element_citation(self, node: etree._Element) -> str:  # noqa: C901
        citation: Citation = {
            "author_names": "",
            "title": "",
            "source": "",
            "year": "",
            "volume": "",
            "page": "",
            "pub_id": "",
            "publisher_name": "",
            "publisher_loc": "",
        }

        _log.debug("Citation parsing started")

        # Author names
        names = []
        for name_node in node.xpath(".//name"):
            name_str = (
                name_node.xpath("surname")[0].text.replace("\n", " ").strip()
                + " "
                + name_node.xpath("given-names")[0].text.replace("\n", " ").strip()
            )
            names.append(name_str)
        etal_node = node.xpath(".//etal")
        if len(etal_node) > 0:
            etal_text = etal_node[0].text or DEFAULT_TEXT_ETAL
            names.append(etal_text)
        citation["author_names"] = ", ".join(names)

        titles: list[str] = [
            "article-title",
            "chapter-title",
            "data-title",
            "issue-title",
            "part-title",
            "trans-title",
        ]
        title_node: Optional[etree._Element] = None
        for name in titles:
            name_node = node.xpath(name)
            if len(name_node) > 0:
                title_node = name_node[0]
                break
        citation["title"] = (
            JatsDocumentBackend._get_text(title_node)
            if title_node is not None
            else node.text.replace("\n", " ").strip()
        )

        # Journal, year, publisher name, publisher location, volume, elocation
        fields: list[str] = [
            "source",
            "year",
            "publisher-name",
            "publisher-loc",
            "volume",
        ]
        for item in fields:
            item_node = node.xpath(item)
            if len(item_node) > 0:
                citation[item.replace("-", "_")] = (  # type: ignore[literal-required]
                    item_node[0].text.replace("\n", " ").strip()
                )

        # Publication identifier
        if len(node.xpath("pub-id")) > 0:
            pub_id: list[str] = []
            for id_node in node.xpath("pub-id"):
                id_type = id_node.get("assigning-authority") or id_node.get(
                    "pub-id-type"
                )
                id_text = id_node.text
                if id_type and id_text:
                    pub_id.append(
                        id_type.replace("\n", " ").strip().upper()
                        + ": "
                        + id_text.replace("\n", " ").strip()
                    )
            if pub_id:
                citation["pub_id"] = ", ".join(pub_id)

        # Pages
        if len(node.xpath("elocation-id")) > 0:
            citation["page"] = (
                node.xpath("elocation-id")[0].text.replace("\n", " ").strip()
            )
        elif len(node.xpath("fpage")) > 0:
            citation["page"] = node.xpath("fpage")[0].text.replace("\n", " ").strip()
            if len(node.xpath("lpage")) > 0:
                citation["page"] += (
                    "–" + node.xpath("lpage")[0].text.replace("\n", " ").strip()  # noqa: RUF001
                )

        # Flatten the citation to string

        text = ""
        if citation["author_names"]:
            text += citation["author_names"].rstrip(".") + ". "
        if citation["title"]:
            text += citation["title"] + ". "
        if citation["source"]:
            text += citation["source"] + ". "
        if citation["publisher_name"]:
            if citation["publisher_loc"]:
                text += f"{citation['publisher_loc']}: "
            text += citation["publisher_name"] + ". "
        if citation["volume"]:
            text = text.rstrip(". ")
            text += f" {citation['volume']}. "
        if citation["page"]:
            text = text.rstrip(". ")
            if citation["volume"]:
                text += ":"
            text += citation["page"] + ". "
        if citation["year"]:
            text = text.rstrip(". ")
            text += f" ({citation['year']})."
        if citation["pub_id"]:
            text = text.rstrip(".") + ". "
            text += citation["pub_id"]

        _log.debug("Citation flattened")

        return text

    def _add_equation(
        self, doc: DoclingDocument, parent: NodeItem, node: etree._Element
    ) -> None:
        math_text = node.text
        math_parts = math_text.split("$$")
        if len(math_parts) == 3:
            math_formula = math_parts[1]
            doc.add_text(label=DocItemLabel.FORMULA, text=math_formula, parent=parent)

        return

    def _add_figure_captions(
        self, doc: DoclingDocument, parent: NodeItem, node: etree._Element
    ) -> None:
        label_node = node.xpath("label")
        label: Optional[str] = (
            JatsDocumentBackend._get_text(label_node[0]).strip() if label_node else ""
        )

        caption_node = node.xpath("caption")
        caption: Optional[str]
        if len(caption_node) > 0:
            caption = ""
            for caption_par in list(caption_node[0]):
                if caption_par.xpath(".//supplementary-material"):
                    continue
                caption += JatsDocumentBackend._get_text(caption_par).strip() + " "
            caption = caption.strip()
        else:
            caption = None

        # TODO: format label vs caption once styling is supported
        fig_text: str = f"{label}{' ' if label and caption else ''}{caption}"
        fig_caption: Optional[TextItem] = (
            doc.add_text(label=DocItemLabel.CAPTION, text=fig_text)
            if fig_text
            else None
        )

        doc.add_picture(parent=parent, caption=fig_caption)

        return

    # TODO: add footnotes when DocItemLabel.FOOTNOTE and styling are supported
    # def _add_footnote_group(self, doc: DoclingDocument, parent: NodeItem, node: etree._Element) -> None:
    #     new_parent = doc.add_group(label=GroupLabel.LIST, name="footnotes", parent=parent)
    #     for child in node.iterchildren(tag="fn"):
    #         text = JatsDocumentBackend._get_text(child)
    #         doc.add_list_item(text=text, parent=new_parent)

    def _add_metadata(
        self, doc: DoclingDocument, xml_components: XMLComponents
    ) -> None:
        self._add_title(doc, xml_components)
        self._add_authors(doc, xml_components)
        self._add_abstract(doc, xml_components)

        return

    def _add_table(
        self, doc: DoclingDocument, parent: NodeItem, table_xml_component: Table
    ) -> None:
        soup = BeautifulSoup(table_xml_component["content"], "html.parser")
        table_tag = soup.find("table")
        if not isinstance(table_tag, Tag):
            return

        data = HTMLDocumentBackend.parse_table_data(table_tag)

        # TODO: format label vs caption once styling is supported
        label = table_xml_component["label"]
        caption = table_xml_component["caption"]
        table_text: str = f"{label}{' ' if label and caption else ''}{caption}"
        table_caption: Optional[TextItem] = (
            doc.add_text(label=DocItemLabel.CAPTION, text=table_text)
            if table_text
            else None
        )

        if data is not None:
            doc.add_table(data=data, parent=parent, caption=table_caption)

        return

    def _add_tables(
        self, doc: DoclingDocument, parent: NodeItem, node: etree._Element
    ) -> None:
        table: Table = {"label": "", "caption": "", "content": ""}

        # Content
        if len(node.xpath("table")) > 0:
            table_content_node = node.xpath("table")[0]
        elif len(node.xpath("alternatives/table")) > 0:
            table_content_node = node.xpath("alternatives/table")[0]
        else:
            table_content_node = None
        if table_content_node is not None:
            table["content"] = etree.tostring(table_content_node).decode("utf-8")

        # Caption
        caption_node = node.xpath("caption")
        caption: Optional[str]
        if caption_node:
            caption = ""
            for caption_par in list(caption_node[0]):
                if caption_par.xpath(".//supplementary-material"):
                    continue
                caption += JatsDocumentBackend._get_text(caption_par).strip() + " "
            caption = caption.strip()
        else:
            caption = None
        if caption is not None:
            table["caption"] = caption

        # Label
        if len(node.xpath("label")) > 0:
            table["label"] = node.xpath("label")[0].text

        try:
            self._add_table(doc, parent, table)
        except Exception:
            _log.warning(f"Skipping unsupported table in {self.file!s}")

        return

    def _add_title(self, doc: DoclingDocument, xml_components: XMLComponents) -> None:
        self.root = doc.add_text(
            parent=None,
            text=xml_components["title"],
            label=DocItemLabel.TITLE,
        )
        return

    def _walk_linear(  # noqa: C901
        self, doc: DoclingDocument, parent: NodeItem, node: etree._Element
    ) -> str:
        skip_tags = ["term"]
        flush_tags = ["ack", "sec", "list", "boxed-text", "disp-formula", "fig"]
        new_parent: NodeItem = parent
        node_text: str = (
            node.text.replace("\n", " ")
            if (node.tag not in skip_tags and node.text)
            else ""
        )

        for child in list(node):
            stop_walk: bool = False

            # flush text into TextItem for some tags in paragraph nodes
            if node.tag == "p" and node_text.strip() and child.tag in flush_tags:
                doc.add_text(
                    label=DocItemLabel.TEXT, text=node_text.strip(), parent=parent
                )
                node_text = ""

            # add elements and decide whether to stop walking
            if child.tag in ("sec", "ack"):
                header = child.xpath("title|label")
                text: Optional[str] = None
                if len(header) > 0:
                    text = JatsDocumentBackend._get_text(header[0])
                elif child.tag == "ack":
                    text = DEFAULT_HEADER_ACKNOWLEDGMENTS
                if text:
                    new_parent = doc.add_heading(text=text, parent=parent)
            elif child.tag == "list":
                new_parent = doc.add_group(
                    label=GroupLabel.LIST, name="list", parent=parent
                )
            elif child.tag == "list-item":
                # TODO: address any type of content (another list, formula,...)
                # TODO: address list type and item label
                text = JatsDocumentBackend._get_text(child).strip()
                new_parent = doc.add_list_item(text=text, parent=parent)
                stop_walk = True
            elif child.tag == "fig":
                self._add_figure_captions(doc, parent, child)
                stop_walk = True
            elif child.tag == "table-wrap":
                self._add_tables(doc, parent, child)
                stop_walk = True
            elif child.tag == "suplementary-material":
                stop_walk = True
            elif child.tag == "fn-group":
                # header = child.xpath(".//title") or child.xpath(".//label")
                # if header:
                #     text = JatsDocumentBackend._get_text(header[0])
                #     fn_parent = doc.add_heading(text=text, parent=new_parent)
                # self._add_footnote_group(doc, fn_parent, child)
                stop_walk = True
            elif child.tag == "ref-list" and node.tag != "ref-list":
                header = child.xpath("title|label")
                text = (
                    JatsDocumentBackend._get_text(header[0])
                    if len(header) > 0
                    else DEFAULT_HEADER_REFERENCES
                )
                new_parent = doc.add_heading(text=text, parent=parent)
                new_parent = doc.add_group(
                    parent=new_parent, label=GroupLabel.LIST, name="list"
                )
            elif child.tag == "element-citation":
                text = self._parse_element_citation(child)
                self._add_citation(doc, parent, text)
                stop_walk = True
            elif child.tag == "mixed-citation":
                text = JatsDocumentBackend._get_text(child).strip()
                self._add_citation(doc, parent, text)
                stop_walk = True
            elif child.tag == "tex-math":
                self._add_equation(doc, parent, child)
                stop_walk = True
            elif child.tag == "inline-formula":
                # TODO: address inline formulas when supported by docling-core
                stop_walk = True

            # step into child
            if not stop_walk:
                new_text = self._walk_linear(doc, new_parent, child)
                if not (node.getparent().tag == "p" and node.tag in flush_tags):
                    node_text += new_text

            # pick up the tail text
            node_text += child.tail.replace("\n", " ") if child.tail else ""

        # create paragraph
        if node.tag == "p" and node_text.strip():
            doc.add_text(label=DocItemLabel.TEXT, text=node_text.strip(), parent=parent)
            return ""
        else:
            # backpropagate the text
            return node_text
