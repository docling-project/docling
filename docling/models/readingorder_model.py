from pathlib import Path
from typing import Dict, List

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupLabel,
    NodeItem,
    ProvenanceItem,
    RefItem,
    TableData,
)
from docling_core.types.doc.document import ContentLayer
from docling_ibm_models.reading_order.reading_order_rb import (
    PageElement as ReadingOrderPageElement,
    ReadingOrderPredictor,
)
from pydantic import BaseModel, ConfigDict

from docling.datamodel.base_models import (
    BasePageElement,
    Cluster,
    ContainerElement,
    FigureElement,
    Table,
    TextElement,
)
from docling.datamodel.document import ConversionResult
from docling.utils.profiling import ProfilingScope, TimeRecorder


class ReadingOrderOptions(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_names: str = ""  # e.g. "language;term;reference"


class ReadingOrderModel:
    def __init__(self, options: ReadingOrderOptions):
        self.options = options
        self.ro_model = ReadingOrderPredictor()

    def _assembled_to_readingorder_elements(
        self, conv_res: ConversionResult
    ) -> List[ReadingOrderPageElement]:
        elements: List[ReadingOrderPageElement] = []
        page_no_to_pages = {p.page_no: p for p in conv_res.pages}

        for element in conv_res.assembled.elements:
            page_height = page_no_to_pages[element.page_no].size.height  # type: ignore
            bbox = element.cluster.bbox.to_bottom_left_origin(page_height)
            text = element.text or ""

            elements.append(
                ReadingOrderPageElement(
                    cid=len(elements),
                    ref=RefItem(cref=f"#/{element.page_no}/{element.cluster.id}"),
                    text=text,
                    page_no=element.page_no,
                    page_size=page_no_to_pages[element.page_no].size,
                    label=element.label,
                    l=bbox.l,
                    r=bbox.r,
                    b=bbox.b,
                    t=bbox.t,
                    coord_origin=bbox.coord_origin,
                )
            )

        return elements

    def _add_child_elements(
        self, element: BasePageElement, doc_item: NodeItem, doc: DoclingDocument
    ):
        child: Cluster
        for child in element.cluster.children:
            c_label = child.label
            c_bbox = child.bbox.to_bottom_left_origin(
                doc.pages[element.page_no + 1].size.height
            )
            c_text = " ".join(
                [
                    cell.text.replace("\x02", "-").strip()
                    for cell in child.cells
                    if len(cell.text.strip()) > 0
                ]
            )

            c_prov = ProvenanceItem(
                page_no=element.page_no + 1, charspan=(0, len(c_text)), bbox=c_bbox
            )
            if c_label == DocItemLabel.LIST_ITEM:
                # TODO: Infer if this is a numbered or a bullet list item
                doc.add_list_item(parent=doc_item, text=c_text, prov=c_prov)
            elif c_label == DocItemLabel.SECTION_HEADER:
                doc.add_heading(parent=doc_item, text=c_text, prov=c_prov)
            else:
                doc.add_text(parent=doc_item, label=c_label, text=c_text, prov=c_prov)

    def _readingorder_elements_to_docling_doc(  # noqa: C901
        self,
        conv_res: ConversionResult,
        ro_elements: List[ReadingOrderPageElement],
        el_to_captions_mapping: Dict[int, List[int]],
        el_to_footnotes_mapping: Dict[int, List[int]],
        el_merges_mapping: Dict[int, List[int]],
    ) -> DoclingDocument:
        id_to_elem = {
            RefItem(cref=f"#/{elem.page_no}/{elem.cluster.id}").cref: elem
            for elem in conv_res.assembled.elements
        }
        cid_to_rels = {rel.cid: rel for rel in ro_elements}

        origin = DocumentOrigin(
            mimetype="application/pdf",
            filename=conv_res.input.file.name,
            binary_hash=conv_res.input.document_hash,
        )
        doc_name = Path(origin.filename).stem
        out_doc: DoclingDocument = DoclingDocument(name=doc_name, origin=origin)

        for page in conv_res.pages:
            page_no = page.page_no + 1
            size = page.size

            assert size is not None

            out_doc.add_page(page_no=page_no, size=size)

        current_list = None
        skippable_cids = {
            cid
            for mapping in (
                el_to_captions_mapping,
                el_to_footnotes_mapping,
                el_merges_mapping,
            )
            for lst in mapping.values()
            for cid in lst
        }

        page_no_to_pages = {p.page_no: p for p in conv_res.pages}

        for rel in ro_elements:
            if rel.cid in skippable_cids:
                continue
            element = id_to_elem[rel.ref.cref]

            page_height = page_no_to_pages[element.page_no].size.height  # type: ignore

            if isinstance(element, TextElement):
                if element.label == DocItemLabel.CODE:
                    cap_text = element.text
                    prov = ProvenanceItem(
                        page_no=element.page_no + 1,
                        charspan=(0, len(cap_text)),
                        bbox=element.cluster.bbox.to_bottom_left_origin(page_height),
                    )
                    code_item = out_doc.add_code(text=cap_text, prov=prov)

                    if rel.cid in el_to_captions_mapping.keys():
                        for caption_cid in el_to_captions_mapping[rel.cid]:
                            caption_elem = id_to_elem[cid_to_rels[caption_cid].ref.cref]
                            new_cap_item = self._add_caption_or_footnote(
                                caption_elem, out_doc, code_item, page_height
                            )

                            code_item.captions.append(new_cap_item.get_ref())

                    if rel.cid in el_to_footnotes_mapping.keys():
                        for footnote_cid in el_to_footnotes_mapping[rel.cid]:
                            footnote_elem = id_to_elem[
                                cid_to_rels[footnote_cid].ref.cref
                            ]
                            new_footnote_item = self._add_caption_or_footnote(
                                footnote_elem, out_doc, code_item, page_height
                            )

                            code_item.footnotes.append(new_footnote_item.get_ref())
                else:
                    new_item, current_list = self._handle_text_element(
                        element, out_doc, current_list, page_height
                    )

                    if rel.cid in el_merges_mapping.keys():
                        for merged_cid in el_merges_mapping[rel.cid]:
                            merged_elem = id_to_elem[cid_to_rels[merged_cid].ref.cref]

                            self._merge_elements(
                                element, merged_elem, new_item, page_height
                            )

            elif isinstance(element, Table):
                tbl_data = TableData(
                    num_rows=element.num_rows,
                    num_cols=element.num_cols,
                    table_cells=element.table_cells,
                )

                prov = ProvenanceItem(
                    page_no=element.page_no + 1,
                    charspan=(0, 0),
                    bbox=element.cluster.bbox.to_bottom_left_origin(page_height),
                )

                tbl = out_doc.add_table(
                    data=tbl_data, prov=prov, label=element.cluster.label
                )

                if rel.cid in el_to_captions_mapping.keys():
                    for caption_cid in el_to_captions_mapping[rel.cid]:
                        caption_elem = id_to_elem[cid_to_rels[caption_cid].ref.cref]
                        new_cap_item = self._add_caption_or_footnote(
                            caption_elem, out_doc, tbl, page_height
                        )

                        tbl.captions.append(new_cap_item.get_ref())

                if rel.cid in el_to_footnotes_mapping.keys():
                    for footnote_cid in el_to_footnotes_mapping[rel.cid]:
                        footnote_elem = id_to_elem[cid_to_rels[footnote_cid].ref.cref]
                        new_footnote_item = self._add_caption_or_footnote(
                            footnote_elem, out_doc, tbl, page_height
                        )

                        tbl.footnotes.append(new_footnote_item.get_ref())

                # TODO: Consider adding children of Table.

            elif isinstance(element, FigureElement):
                cap_text = ""
                prov = ProvenanceItem(
                    page_no=element.page_no + 1,
                    charspan=(0, len(cap_text)),
                    bbox=element.cluster.bbox.to_bottom_left_origin(page_height),
                )
                pic = out_doc.add_picture(prov=prov)

                if rel.cid in el_to_captions_mapping.keys():
                    for caption_cid in el_to_captions_mapping[rel.cid]:
                        caption_elem = id_to_elem[cid_to_rels[caption_cid].ref.cref]
                        new_cap_item = self._add_caption_or_footnote(
                            caption_elem, out_doc, pic, page_height
                        )

                        pic.captions.append(new_cap_item.get_ref())

                if rel.cid in el_to_footnotes_mapping.keys():
                    for footnote_cid in el_to_footnotes_mapping[rel.cid]:
                        footnote_elem = id_to_elem[cid_to_rels[footnote_cid].ref.cref]
                        new_footnote_item = self._add_caption_or_footnote(
                            footnote_elem, out_doc, pic, page_height
                        )

                        pic.footnotes.append(new_footnote_item.get_ref())

                self._add_child_elements(element, pic, out_doc)

            elif isinstance(element, ContainerElement):  # Form, KV region
                label = element.label
                group_label = GroupLabel.UNSPECIFIED
                if label == DocItemLabel.FORM:
                    group_label = GroupLabel.FORM_AREA
                elif label == DocItemLabel.KEY_VALUE_REGION:
                    group_label = GroupLabel.KEY_VALUE_AREA

                container_el = out_doc.add_group(label=group_label)

                self._add_child_elements(element, container_el, out_doc)

        return out_doc

    def _add_caption_or_footnote(self, elem, out_doc, parent, page_height):
        assert isinstance(elem, TextElement)
        text = elem.text
        prov = ProvenanceItem(
            page_no=elem.page_no + 1,
            charspan=(0, len(text)),
            bbox=elem.cluster.bbox.to_bottom_left_origin(page_height),
        )
        new_item = out_doc.add_text(
            label=elem.label, text=text, prov=prov, parent=parent
        )
        return new_item

    def _handle_text_element(self, element, out_doc, current_list, page_height):
        cap_text = element.text

        prov = ProvenanceItem(
            page_no=element.page_no + 1,
            charspan=(0, len(cap_text)),
            bbox=element.cluster.bbox.to_bottom_left_origin(page_height),
        )
        label = element.label
        if label == DocItemLabel.LIST_ITEM:
            if current_list is None:
                current_list = out_doc.add_group(label=GroupLabel.LIST, name="list")

            # TODO: Infer if this is a numbered or a bullet list item
            new_item = out_doc.add_list_item(
                text=cap_text, enumerated=False, prov=prov, parent=current_list
            )
        elif label == DocItemLabel.SECTION_HEADER:
            current_list = None

            new_item = out_doc.add_heading(text=cap_text, prov=prov)
        elif label == DocItemLabel.FORMULA:
            current_list = None

            new_item = out_doc.add_text(
                label=DocItemLabel.FORMULA, text="", orig=cap_text, prov=prov
            )
        else:
            current_list = None

            content_layer = ContentLayer.BODY
            if element.label in [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER]:
                content_layer = ContentLayer.FURNITURE

            new_item = out_doc.add_text(
                label=element.label,
                text=cap_text,
                prov=prov,
                content_layer=content_layer,
            )
        return new_item, current_list

    def _merge_elements(self, element, merged_elem, new_item, page_height):
        assert isinstance(merged_elem, type(element)), (
            "Merged element must be of same type as element."
        )
        assert merged_elem.label == new_item.label, (
            "Labels of merged elements must match."
        )
        prov = ProvenanceItem(
            page_no=element.page_no + 1,
            charspan=(
                len(new_item.text) + 1,
                len(new_item.text) + 1 + len(merged_elem.text),
            ),
            bbox=element.cluster.bbox.to_bottom_left_origin(page_height),
        )
        new_item.text += f" {merged_elem.text}"
        new_item.orig += f" {merged_elem.text}"  # TODO: This is incomplete, we don't have the `orig` field of the merged element.
        new_item.prov.append(prov)

    def __call__(self, conv_res: ConversionResult) -> DoclingDocument:
        with TimeRecorder(conv_res, "reading_order", scope=ProfilingScope.DOCUMENT):
            page_elements = self._assembled_to_readingorder_elements(conv_res)

            # Apply reading order
            sorted_elements = self.ro_model.predict_reading_order(
                page_elements=page_elements
            )
            el_to_captions_mapping = self.ro_model.predict_to_captions(
                sorted_elements=sorted_elements
            )
            el_to_footnotes_mapping = self.ro_model.predict_to_footnotes(
                sorted_elements=sorted_elements
            )
            el_merges_mapping = self.ro_model.predict_merges(
                sorted_elements=sorted_elements
            )

            docling_doc: DoclingDocument = self._readingorder_elements_to_docling_doc(
                conv_res,
                sorted_elements,
                el_to_captions_mapping,
                el_to_footnotes_mapping,
                el_merges_mapping,
            )

        return docling_doc
