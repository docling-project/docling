import re
from collections.abc import Generator
from contextlib import contextmanager
from functools import cached_property
from io import BytesIO
from logging import Logger
from pathlib import Path, PurePath
from typing import Optional, Union

from docling_core.types.doc import BoundingBox, DocItemLabel, ListItem, TextItem
from docling_ibm_models.reading_order.reading_order_rb import (
    PageElement as ReadingOrderPageElement,
)
from docling_parse.pdf_parser import PdfTableOfContents

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import DocumentStream, PageElement, TextElement
from docling.datamodel.document import ConversionResult
from docling.models.header_hierarchy.types.hierarchical_header import HierarchicalHeader

logger = Logger(__name__)


class HeaderNotFoundException(Exception):
    def __init__(self, heading: str):
        super().__init__(f"Following heading was not found in the document: {heading}")


class ImplausibleHeadingStructureException(Exception):
    def __init__(self) -> None:
        super().__init__(
            "Hierarchy demands equal level heading, but no common parent was found!"
        )


class PdfBackendIncompatible(Exception):
    def __init__(self, backend) -> None:
        super().__init__(
            f"The selected backend is '{type(backend)}' instead of 'DoclingParseV4DocumentBackend'."
        )


class InvalidSourceTypeException(Exception):
    pass


class HierarchyBuilderMetadata:
    def __init__(
        self,
        conv_res: ConversionResult,
        sorted_elements: list[ReadingOrderPageElement],
        raise_on_error: bool = False,
    ):
        # if not isinstance(conv_res.input._backend, (DoclingParseV4DocumentBackend, PyPdfiumDocumentBackend)):
        if not isinstance(conv_res.input._backend, DoclingParseV4DocumentBackend):
            raise PdfBackendIncompatible(conv_res.input._backend)
        backend: DoclingParseV4DocumentBackend = conv_res.input._backend
        self.toc_meta: Optional[PdfTableOfContents] = backend.get_table_of_contents()
        self.conv_res: ConversionResult = conv_res
        self.all_elements: list[PageElement] = conv_res.assembled.elements
        self.all_cids: list[str] = [
            f"#/{element.page_no}/{element.cluster.id}" for element in self.all_elements
        ]
        self.sorted_ro_elements: list[ReadingOrderPageElement] = sorted_elements
        self.raise_on_error: bool = raise_on_error
        self.cid_to_page_element: dict[str, PageElement] = dict(
            zip(self.all_cids, self.all_elements)
        )

    def _iterate_toc(
        self, toc_element: Optional[PdfTableOfContents] = None, level: int = 0
    ):
        if toc_element is None:
            toc_element = self.toc_meta
        if toc_element:
            if toc_element.text != "<root>":
                yield level, toc_element.text
            for child in toc_element.children:
                yield from self._iterate_toc(child, level + 1)

    def infer(self) -> HierarchicalHeader:
        root = HierarchicalHeader()
        current = root

        # my problem is that I will need the font information in PdfTextCell, but at the same time I need the ordered text elements (with self refs ideally)

        for level, title in self._iterate_toc():
            new_parent = None
            this_element = None
            orig_text = ""
            ref = None
            last_i: int = 0
            # identify the text item in the document
            for _i, ro_element in enumerate(self.sorted_ro_elements[last_i:]):
                element = self.cid_to_page_element[ro_element.ref.cref]
                # skip all page elements that are before the last ("current") header
                # if element.page_no < last_page or element.cluster.id <= last_cid:
                #     continue
                # Future to do: fixme - better to look for an overlap with the 'to' pointer if possible...
                if not isinstance(element, TextElement):
                    continue
                orig_text = "".join([c.orig for c in element.cluster.cells])

                if re.sub(r"[^A-Za-z0-9]", "", title) == re.sub(
                    r"[^A-Za-z0-9]", "", orig_text
                ):
                    this_element = element
                    last_i = last_i + _i
                    ref = ro_element.ref.cref
                    break
            if this_element is None:
                if self.raise_on_error:
                    raise HeaderNotFoundException(title)
                else:
                    logger.warning(HeaderNotFoundException(title))
                    continue

            if this_element.label != DocItemLabel.SECTION_HEADER:
                this_element.label = DocItemLabel.SECTION_HEADER

            if current.level_toc is None or level > current.level_toc:
                # print(f"gt: {this_fs_level, this_style_attr} VS: {current.level_fontsize, current.style_attrs}")
                new_parent = current
            elif level == current.level_toc:
                # print(f"eq: {this_fs_level, this_style_attr} VS: {current.level_fontsize, current.style_attrs}")
                if current.parent is not None:
                    new_parent = current.parent
                else:
                    raise ImplausibleHeadingStructureException()
            else:
                # go back up in hierarchy and try to find a new parent
                new_parent = current
                while new_parent.parent is not None and (level <= new_parent.level_toc):
                    new_parent = new_parent.parent
                # print(f"fit parent for : {this_fs_level, this_style_attr} parent: {new_parent.level_fontsize, new_parent.style_attrs}")
            new_obj = HierarchicalHeader(
                text=orig_text,
                parent=new_parent,
                level_toc=level,
                doc_ref=ref,
            )
            new_parent.children.append(new_obj)
            current = new_obj

        return root
