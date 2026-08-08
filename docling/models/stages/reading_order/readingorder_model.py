from pathlib import Path
from statistics import median

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
    TableItem,
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
from docling.utils.text_normalization import normalize_arabic_presentation_forms


class ReadingOrderOptions(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_names: str = ""  # e.g. "language;term;reference"


class ReadingOrderModel:
    _RICH_CELL_PICTURE_COVERAGE_THRESHOLD = 0.8

    def __init__(self, options: ReadingOrderOptions):
        self.options = options
        self.ro_model = ReadingOrderPredictor()
        self.list_item_processor = ListItemMarkerProcessor()

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
                    ref=RefItem(cref=self._element_ref(element)),
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
            c_text = normalize_arabic_presentation_forms(
                " ".join(
                    [
                        cell.text.replace("\x02", "-").strip()
                        for cell in child.cells
                        if len(cell.text.strip()) > 0
                    ]
                )
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

    @staticmethod
    def _element_ref(element: BasePageElement) -> str:
        return f"#/{element.page_no}/{element.cluster.id}"

    @classmethod
    def _match_table_pictures(
        cls,
        elements: list[BasePageElement],
        excluded_picture_refs: set[str],
    ) -> dict[str, dict[int, list[FigureElement]]]:
        tables = [element for element in elements if isinstance(element, Table)]
        matches: dict[str, dict[int, list[FigureElement]]] = {}

        for picture in (
            element for element in elements if isinstance(element, FigureElement)
        ):
            if cls._element_ref(picture) in excluded_picture_refs:
                continue

            best_match: tuple[float, Table, int] | None = None
            for table in tables:
                if table.page_no != picture.page_no:
                    continue
                if (
                    picture.cluster.bbox.intersection_over_self(table.cluster.bbox)
                    < cls._RICH_CELL_PICTURE_COVERAGE_THRESHOLD
                ):
                    continue

                cell_match = cls._match_picture_to_table_cell(table, picture)
                if cell_match is not None and (
                    best_match is None or cell_match[0] > best_match[0]
                ):
                    best_match = (cell_match[0], table, cell_match[1])

            if best_match is None:
                continue

            _, table, cell_index = best_match
            matches.setdefault(cls._element_ref(table), {}).setdefault(
                cell_index, []
            ).append(picture)

        return matches

    @classmethod
    def _match_picture_to_table_cell(
        cls, table: Table, picture: FigureElement
    ) -> tuple[float, int] | None:
        eligible = [
            (
                picture.cluster.bbox.intersection_over_self(cell.bbox),
                cell_index,
                cell,
            )
            for cell_index, cell in enumerate(table.table_cells)
            if cell.bbox is not None
            and picture.cluster.bbox.intersection_over_self(cell.bbox)
            >= cls._RICH_CELL_PICTURE_COVERAGE_THRESHOLD
        ]
        if not eligible:
            return None

        # Cell boxes can overlap across logical rows and columns. Infer the
        # picture's grid position before choosing among containing cells.
        row_centers: dict[int, list[float]] = {}
        column_centers: dict[int, list[float]] = {}
        for cell in table.table_cells:
            if cell.bbox is None:
                continue
            for row in range(cell.start_row_offset_idx, cell.end_row_offset_idx):
                row_centers.setdefault(row, []).append((cell.bbox.t + cell.bbox.b) / 2)
            for column in range(cell.start_col_offset_idx, cell.end_col_offset_idx):
                column_centers.setdefault(column, []).append(
                    (cell.bbox.l + cell.bbox.r) / 2
                )

        picture_x = (picture.cluster.bbox.l + picture.cluster.bbox.r) / 2
        picture_y = (picture.cluster.bbox.t + picture.cluster.bbox.b) / 2
        row = min(
            row_centers,
            key=lambda index: abs(median(row_centers[index]) - picture_y),
        )
        column = min(
            column_centers,
            key=lambda index: abs(median(column_centers[index]) - picture_x),
        )
        logical_matches = [
            match
            for match in eligible
            if match[2].start_row_offset_idx <= row < match[2].end_row_offset_idx
            and match[2].start_col_offset_idx <= column < match[2].end_col_offset_idx
        ]
        coverage, cell_index, _ = max(logical_matches or eligible)
        return coverage, cell_index

    def _add_rich_table_pictures(
        self,
        *,
        table_item: TableItem,
        pictures_by_cell: dict[int, list[FigureElement]],
        doc: DoclingDocument,
        page_height: float,
    ) -> None:
        for cell_index, pictures in pictures_by_cell.items():
            cell = table_item.data.table_cells[cell_index]
            group = doc.add_group(
                label=GroupLabel.UNSPECIFIED,
                name=(
                    f"rich_cell_group_{len(doc.tables)}_"
                    f"{cell.start_col_offset_idx}_{cell.start_row_offset_idx}"
                ),
                parent=table_item,
            )

            if cell.text:
                cell_bbox = (
                    cell.bbox if cell.bbox is not None else pictures[0].cluster.bbox
                )
                doc.add_text(
                    label=DocItemLabel.TEXT,
                    text=cell.text,
                    prov=ProvenanceItem(
                        page_no=pictures[0].page_no,
                        charspan=(0, len(cell.text)),
                        bbox=cell_bbox.to_bottom_left_origin(page_height),
                    ),
                    parent=group,
                )

            for picture in pictures:
                picture_item = doc.add_picture(
                    annotations=picture.annotations,
                    prov=ProvenanceItem(
                        page_no=picture.page_no,
                        charspan=(0, 0),
                        bbox=picture.cluster.bbox.to_bottom_left_origin(page_height),
                    ),
                    parent=group,
                )
                self._add_child_elements(picture, picture_item, doc)

            table_item.data.table_cells[cell_index] = RichTableCell(
                **cell.model_dump(exclude={"ref"}),
                ref=group.get_ref(),
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
            self._element_ref(elem): elem for elem in conv_res.assembled.elements
        }
        cid_to_rels = {rel.cid: rel for rel in ro_elements}
        excluded_picture_refs = {
            rel.ref.cref
            for rel in ro_elements
            if rel.cid in el_to_captions_mapping or rel.cid in el_to_footnotes_mapping
        }
        rich_table_pictures = self._match_table_pictures(
            conv_res.assembled.elements,
            excluded_picture_refs=excluded_picture_refs,
        )
        rich_picture_refs = {
            self._element_ref(picture)
            for cells in rich_table_pictures.values()
            for pictures in cells.values()
            for picture in pictures
        }

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
        skippable_cids.update(
            rel.cid for rel in ro_elements if rel.ref.cref in rich_picture_refs
        )

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
                if rel.ref.cref in rich_table_pictures:
                    self._add_rich_table_pictures(
                        table_item=tbl,
                        pictures_by_cell=rich_table_pictures[rel.ref.cref],
                        doc=out_doc,
                        page_height=page_height,
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
        continuation_text = merged_elem.text.lstrip()
        if new_item.text.endswith("\u00ad") or (
            new_item.text.endswith("-")
            and continuation_text
            and continuation_text[0].islower()
        ):
            # A soft hyphen or hard-hyphenated lowercase continuation is a split word.
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
