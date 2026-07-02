from pathlib import Path

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupLabel,
    NodeItem,
    ProvenanceItem,
    RefItem,
    RichTableCell,
    TableData,
)
from docling_core.types.doc.document import ContentLayer
from docling_ibm_models.list_item_normalizer.list_marker_processor import (
    ListItemMarkerProcessor,
)
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
    # Flowing-text labels a left-margin section header can attach to (#3643).
    _SIDE_HEADING_BODY_LABELS = (DocItemLabel.TEXT, DocItemLabel.LIST_ITEM)
    # Two blocks share a row when they overlap vertically by at least this
    # fraction of the shorter block's height (#3648).
    _SIDE_HEADING_MIN_ROW_OVERLAP = 0.5
    # A section header that introduces a column sits directly above its own
    # column's body; a side-heading does not. Two structural signals identify
    # the body a header introduces, so that header is left untouched (#3648):
    #   * left-edge alignment: a heading and its own paragraph share a left edge
    #     within a few points (sub-pixel rendering jitter); a full-width rule
    #     below a margin heading starts clearly further left.
    _SIDE_HEADING_COLUMN_ALIGN_TOL = 3.0
    #   * vertical proximity: a heading's body follows directly beneath it,
    #     within a small multiple of the heading's own height. This catches
    #     headings indented relative to their body, while still ignoring a
    #     full-width line far below a margin heading.
    _SIDE_HEADING_MAX_GAP_RATIO = 6.0

    def __init__(self, options: ReadingOrderOptions):
        self.options = options
        self.ro_model = ReadingOrderPredictor()
        self.list_item_processor = ListItemMarkerProcessor()

    @staticmethod
    def _reorder_side_headings(
        sorted_elements: list[ReadingOrderPageElement],
    ) -> list[ReadingOrderPageElement]:
        """Re-interleave left-margin section headers with their body block.

        The rule-based predictor reads a full column top-to-bottom before moving
        right, so a "side-heading" layout (narrow ``SECTION_HEADER``s in a left
        column, paragraphs in a right column) is emitted as every header first
        and then every paragraph. This pass detects that pattern and moves each
        side-heading to sit immediately before the body block it is row-aligned
        with, restoring ``header -> paragraph`` reading order.

        A header is treated as a side-heading when it does *not* introduce a
        column (no body text sits directly beneath it, left-aligned within its
        own x-band) *and* a body block sits row-aligned to its right. The header
        is then placed before the topmost line of that block, which keeps big
        multi-line headers and line-fragmented paragraphs in order. Ordinary
        single-column and true multi-column headers each sit above their own
        column's body, so they are left untouched.
        See https://github.com/docling-project/docling/issues/3643 and #3648.
        """
        body_labels = ReadingOrderModel._SIDE_HEADING_BODY_LABELS
        min_row_overlap = ReadingOrderModel._SIDE_HEADING_MIN_ROW_OVERLAP
        align_tol = ReadingOrderModel._SIDE_HEADING_COLUMN_ALIGN_TOL
        max_gap_ratio = ReadingOrderModel._SIDE_HEADING_MAX_GAP_RATIO

        # Group element indices by page; relative order is otherwise preserved.
        page_to_indices: dict[int, list[int]] = {}
        for idx, el in enumerate(sorted_elements):
            page_to_indices.setdefault(el.page_no, []).append(idx)

        anchor: dict[int, int] = {}  # header index -> body index it precedes
        for indices in page_to_indices.values():
            body_indices = [
                i for i in indices if sorted_elements[i].label in body_labels
            ]
            for i in indices:
                header = sorted_elements[i]
                if header.label != DocItemLabel.SECTION_HEADER:
                    continue

                header_height = header.t - header.b

                # A header that introduces a column has body text in its own
                # x-band directly beneath it; a side-heading does not (its column
                # holds only headings, the body sits to the right). The body it
                # introduces is recognised by sharing the header's left edge or
                # by sitting directly below it. This structural test replaces the
                # old width-ratio heuristic, which misfired on short in-column
                # headings like "1 INTRODUCTION"; a full-width rule below a margin
                # heading is neither left-aligned nor close, so it does not count.
                heads_a_column = any(
                    header.is_above(body)
                    and header.overlaps_horizontally(body)
                    and (
                        abs(body.l - header.l) <= align_tol
                        or (header.b - body.t) <= max_gap_ratio * header_height
                    )
                    for body in (sorted_elements[j] for j in body_indices)
                )
                if heads_a_column:
                    continue

                # Anchor before the topmost body block lying entirely to the
                # header's right that shares its row. Comparing the row overlap
                # to the shorter height keeps tall headers matching short
                # paragraph lines.
                best_top: float | None = None
                best_body: int | None = None
                for b in body_indices:
                    body = sorted_elements[b]
                    if not header.is_strictly_left_of(body):
                        continue  # body must lie to the right of the header
                    body_height = body.t - body.b
                    overlap = header.y_overlap_with(body)
                    if overlap < min_row_overlap * min(header_height, body_height):
                        continue  # not on the same row
                    if best_top is None or body.t > best_top:
                        best_top, best_body = body.t, b
                if best_body is not None:
                    anchor[i] = best_body

        if not anchor:
            return sorted_elements

        # body index -> headers to place before it (higher on the page first)
        headers_before: dict[int, list[int]] = {}
        for header_idx, body_idx in anchor.items():
            headers_before.setdefault(body_idx, []).append(header_idx)
        for header_list in headers_before.values():
            header_list.sort(key=lambda i: sorted_elements[i].t, reverse=True)

        moved = set(anchor)
        reordered: list[ReadingOrderPageElement] = []
        for idx, el in enumerate(sorted_elements):
            if idx in moved:
                continue  # emitted right before its anchor body block
            for header_idx in headers_before.get(idx, []):
                reordered.append(sorted_elements[header_idx])
            reordered.append(el)
        return reordered

    def _assembled_to_readingorder_elements(
        self, conv_res: ConversionResult
    ) -> list[ReadingOrderPageElement]:
        elements: list[ReadingOrderPageElement] = []
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
                doc.pages[element.page_no].size.height
            )
            c_text = " ".join(
                [
                    cell.text.replace("\x02", "-").strip()
                    for cell in child.cells
                    if len(cell.text.strip()) > 0
                ]
            )

            c_prov = ProvenanceItem(
                page_no=element.page_no, charspan=(0, len(c_text)), bbox=c_bbox
            )
            if c_label == DocItemLabel.LIST_ITEM:
                # TODO: Infer if this is a numbered or a bullet list item
                l_item = doc.add_list_item(parent=doc_item, text=c_text, prov=c_prov)
                self.list_item_processor.process_list_item(l_item)
            elif c_label == DocItemLabel.SECTION_HEADER:
                doc.add_heading(parent=doc_item, text=c_text, prov=c_prov)
            else:
                content_layer = ContentLayer.BODY
                if c_label in (
                    DocItemLabel.PAGE_HEADER,
                    DocItemLabel.PAGE_FOOTER,
                ):
                    content_layer = ContentLayer.FURNITURE
                doc.add_text(
                    parent=doc_item,
                    label=c_label,
                    text=c_text,
                    prov=c_prov,
                    content_layer=content_layer,
                )

    def _create_rich_cell_group(
        self, element: BasePageElement, doc: DoclingDocument, table_item: NodeItem
    ) -> RefItem:
        """Create a group containing all child elements for a rich table cell."""
        group_name = f"rich_cell_group_{len(doc.tables)}_0_0"
        group_element = doc.add_group(
            label=GroupLabel.UNSPECIFIED,
            name=group_name,
            parent=table_item,
        )

        # Add all child elements to the group
        self._add_child_elements(element, group_element, doc)

        return group_element.get_ref()

    @staticmethod
    def _table_data_from_table(element: Table) -> TableData:
        if element.num_rows == 0 and element.num_cols == 0:
            num_rows = 1 if element.cluster.children else 0
            num_cols = 1 if element.cluster.children else 0
            return TableData(
                num_rows=num_rows,
                num_cols=num_cols,
                table_cells=[],
                orientation=element.orientation,
            )

        return TableData(
            num_rows=element.num_rows,
            num_cols=element.num_cols,
            table_cells=element.table_cells,
            orientation=element.orientation,
        )

    def _readingorder_elements_to_docling_doc(
        self,
        conv_res: ConversionResult,
        ro_elements: list[ReadingOrderPageElement],
        el_to_captions_mapping: dict[int, list[int]],
        el_to_footnotes_mapping: dict[int, list[int]],
        el_merges_mapping: dict[int, list[int]],
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
            page_no = page.page_no
            size = page.size

            assert size is not None, "Page size is not initialized."

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
                        page_no=element.page_no,
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
                tbl_data = self._table_data_from_table(element)

                prov = ProvenanceItem(
                    page_no=element.page_no,
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

                # Handle case where table has no structure prediction but has children
                if (
                    element.num_rows == 0
                    and element.num_cols == 0
                    and element.cluster.children
                ):
                    # Create rich cell containing all child elements
                    rich_cell_ref = self._create_rich_cell_group(element, out_doc, tbl)

                    # Create rich table cell spanning the entire 1x1 table
                    rich_cell = RichTableCell(
                        text="",  # Empty text since content is in the group
                        row_span=1,
                        col_span=1,
                        start_row_offset_idx=0,
                        end_row_offset_idx=1,
                        start_col_offset_idx=0,
                        end_col_offset_idx=1,
                        column_header=False,
                        row_header=False,
                        ref=rich_cell_ref,
                    )
                    out_doc.add_table_cell(table_item=tbl, cell=rich_cell)

                # TODO: Consider adding children of Table.

            elif isinstance(element, FigureElement):
                cap_text = ""
                prov = ProvenanceItem(
                    page_no=element.page_no,
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
            page_no=elem.page_no,
            charspan=(0, len(text)),
            bbox=elem.cluster.bbox.to_bottom_left_origin(page_height),
        )
        new_item = out_doc.add_text(
            label=elem.label,
            text=text,
            prov=prov,
            parent=parent,
            hyperlink=elem.hyperlink,
        )
        return new_item

    def _handle_text_element(self, element, out_doc, current_list, page_height):
        cap_text = element.text

        prov = ProvenanceItem(
            page_no=element.page_no,
            charspan=(0, len(cap_text)),
            bbox=element.cluster.bbox.to_bottom_left_origin(page_height),
        )
        label = element.label
        if label == DocItemLabel.LIST_ITEM:
            if current_list is None:
                current_list = out_doc.add_group(label=GroupLabel.LIST, name="list")

            # TODO: Infer if this is a numbered or a bullet list item
            new_item = out_doc.add_list_item(
                text=cap_text,
                enumerated=False,
                prov=prov,
                parent=current_list,
                hyperlink=element.hyperlink,
            )
            self.list_item_processor.process_list_item(new_item)

        elif label == DocItemLabel.SECTION_HEADER:
            current_list = None

            new_item = out_doc.add_heading(
                text=cap_text, prov=prov, hyperlink=element.hyperlink
            )
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
                hyperlink=element.hyperlink,
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
            page_no=merged_elem.page_no,
            charspan=(
                len(new_item.text) + 1,
                len(new_item.text) + 1 + len(merged_elem.text),
            ),
            bbox=merged_elem.cluster.bbox.to_bottom_left_origin(page_height),
        )
        if new_item.text.endswith("\u00ad"):
            # Soft hyphen (U+00AD): strip it and join without space (hyphenated word split across lines)
            new_item.text = new_item.text[:-1] + merged_elem.text
            new_item.orig = (
                new_item.orig[:-1] + merged_elem.text
            )  # TODO: This is incomplete, we don't have the `orig` field of the merged element.
        else:
            new_item.text += f" {merged_elem.text}"
            new_item.orig += f" {merged_elem.text}"  # TODO: This is incomplete, we don't have the `orig` field of the merged element.
        new_item.prov.append(prov)

        if new_item.hyperlink != merged_elem.hyperlink:
            new_item.hyperlink = None

    def __call__(self, conv_res: ConversionResult) -> DoclingDocument:
        with TimeRecorder(conv_res, "reading_order", scope=ProfilingScope.DOCUMENT):
            page_elements = self._assembled_to_readingorder_elements(conv_res)

            # Apply reading order
            sorted_elements = self.ro_model.predict_reading_order(
                page_elements=page_elements
            )
            # Re-interleave left-margin section headers with the body block on
            # their right (side-heading layouts, see #3643).
            sorted_elements = self._reorder_side_headings(sorted_elements)
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
