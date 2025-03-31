import re
from pathlib import Path
from typing import List

import pandas as pd
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupLabel,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from docling_core.types.doc.document import ContentLayer


def resolve_item(paths, obj):
    """Find item in document from a reference path"""

    if len(paths) == 0:
        return obj

    if paths[0] == "#":
        return resolve_item(paths[1:], obj)

    try:
        key = int(paths[0])
    except:
        key = paths[0]

    if len(paths) == 1:
        if isinstance(key, str) and key in obj:
            return obj[key]
        elif isinstance(key, int) and key < len(obj):
            return obj[key]
        else:
            return None

    elif len(paths) > 1:
        if isinstance(key, str) and key in obj:
            return resolve_item(paths[1:], obj[key])
        elif isinstance(key, int) and key < len(obj):
            return resolve_item(paths[1:], obj[key])
        else:
            return None

    else:
        return None


def _flatten_table_grid(grid: List[List[dict]]) -> List[dict]:
    unique_objects = []
    seen_spans = set()

    for sublist in grid:
        for obj in sublist:
            # Convert the spans list to a tuple of tuples for hashing
            spans_tuple = tuple(tuple(span) for span in obj["spans"])
            if spans_tuple not in seen_spans:
                seen_spans.add(spans_tuple)
                unique_objects.append(obj)

    return unique_objects


def to_docling_document(doc_glm, update_name_label=False) -> DoclingDocument:
    origin = DocumentOrigin(
        mimetype="application/pdf",
        filename=doc_glm["file-info"]["filename"],
        binary_hash=doc_glm["file-info"]["document-hash"],
    )
    doc_name = Path(origin.filename).stem

    doc: DoclingDocument = DoclingDocument(name=doc_name, origin=origin)

    for page_dim in doc_glm["page-dimensions"]:
        page_no = int(page_dim["page"])
        size = Size(width=page_dim["width"], height=page_dim["height"])

        doc.add_page(page_no=page_no, size=size)

    if "properties" in doc_glm:
        props = pd.DataFrame(
            doc_glm["properties"]["data"], columns=doc_glm["properties"]["headers"]
        )
    else:
        props = pd.DataFrame()

    current_list = None

    for ix, pelem in enumerate(doc_glm["page-elements"]):
        ptype = pelem["type"]
        span_i = pelem["span"][0]
        span_j = pelem["span"][1]

        if "iref" not in pelem:
            # print(json.dumps(pelem, indent=2))
            continue

        iref = pelem["iref"]

        if re.match("#/figures/(\\d+)/captions/(.+)", iref):
            # print(f"skip {iref}")
            continue

        if re.match("#/tables/(\\d+)/captions/(.+)", iref):
            # print(f"skip {iref}")
            continue

        path = iref.split("/")
        obj = resolve_item(path, doc_glm)

        if obj is None:
            current_list = None
            print(f"warning: undefined {path}")
            continue

        if ptype == "figure":
            current_list = None
            text = ""
            caption_refs = []
            for caption in obj["captions"]:
                text += caption["text"]

                for nprov in caption["prov"]:
                    npaths = nprov["$ref"].split("/")
                    nelem = resolve_item(npaths, doc_glm)

                    if nelem is None:
                        # print(f"warning: undefined caption {npaths}")
                        continue

                    span_i = nelem["span"][0]
                    span_j = nelem["span"][1]

                    cap_text = caption["text"][span_i:span_j]

                    # doc_glm["page-elements"].remove(nelem)

                    prov = ProvenanceItem(
                        page_no=nelem["page"],
                        charspan=tuple(nelem["span"]),
                        bbox=BoundingBox.from_tuple(
                            nelem["bbox"], origin=CoordOrigin.BOTTOMLEFT
                        ),
                    )

                    caption_obj = doc.add_text(
                        label=DocItemLabel.CAPTION, text=cap_text, prov=prov
                    )
                    caption_refs.append(caption_obj.get_ref())

            prov = ProvenanceItem(
                page_no=pelem["page"],
                charspan=(0, len(text)),
                bbox=BoundingBox.from_tuple(
                    pelem["bbox"], origin=CoordOrigin.BOTTOMLEFT
                ),
            )

            pic = doc.add_picture(prov=prov)
            pic.captions.extend(caption_refs)
            _add_child_elements(pic, doc, obj, pelem)

        elif ptype == "table":
            current_list = None
            text = ""
            caption_refs = []
            item_label = DocItemLabel(pelem["name"])

            for caption in obj["captions"]:
                text += caption["text"]

                for nprov in caption["prov"]:
                    npaths = nprov["$ref"].split("/")
                    nelem = resolve_item(npaths, doc_glm)

                    if nelem is None:
                        # print(f"warning: undefined caption {npaths}")
                        continue

                    span_i = nelem["span"][0]
                    span_j = nelem["span"][1]

                    cap_text = caption["text"][span_i:span_j]

                    # doc_glm["page-elements"].remove(nelem)

                    prov = ProvenanceItem(
                        page_no=nelem["page"],
                        charspan=tuple(nelem["span"]),
                        bbox=BoundingBox.from_tuple(
                            nelem["bbox"], origin=CoordOrigin.BOTTOMLEFT
                        ),
                    )

                    caption_obj = doc.add_text(
                        label=DocItemLabel.CAPTION, text=cap_text, prov=prov
                    )
                    caption_refs.append(caption_obj.get_ref())

            table_cells_glm = _flatten_table_grid(obj["data"])

            table_cells = []
            for tbl_cell_glm in table_cells_glm:
                if tbl_cell_glm["bbox"] is not None:
                    bbox = BoundingBox.from_tuple(
                        tbl_cell_glm["bbox"], origin=CoordOrigin.BOTTOMLEFT
                    )
                else:
                    bbox = None

                is_col_header = False
                is_row_header = False
                is_row_section = False

                if tbl_cell_glm["type"] == "col_header":
                    is_col_header = True
                elif tbl_cell_glm["type"] == "row_header":
                    is_row_header = True
                elif tbl_cell_glm["type"] == "row_section":
                    is_row_section = True

                table_cells.append(
                    TableCell(
                        row_span=tbl_cell_glm["row-span"][1]
                        - tbl_cell_glm["row-span"][0],
                        col_span=tbl_cell_glm["col-span"][1]
                        - tbl_cell_glm["col-span"][0],
                        start_row_offset_idx=tbl_cell_glm["row-span"][0],
                        end_row_offset_idx=tbl_cell_glm["row-span"][1],
                        start_col_offset_idx=tbl_cell_glm["col-span"][0],
                        end_col_offset_idx=tbl_cell_glm["col-span"][1],
                        text=tbl_cell_glm["text"],
                        bbox=bbox,
                        column_header=is_col_header,
                        row_header=is_row_header,
                        row_section=is_row_section,
                    )
                )

            tbl_data = TableData(
                num_rows=obj.get("#-rows", 0),
                num_cols=obj.get("#-cols", 0),
                table_cells=table_cells,
            )

            prov = ProvenanceItem(
                page_no=pelem["page"],
                charspan=(0, 0),
                bbox=BoundingBox.from_tuple(
                    pelem["bbox"], origin=CoordOrigin.BOTTOMLEFT
                ),
            )

            tbl = doc.add_table(data=tbl_data, prov=prov, label=item_label)
            tbl.captions.extend(caption_refs)

        elif ptype in [DocItemLabel.FORM.value, DocItemLabel.KEY_VALUE_REGION.value]:
            label = DocItemLabel(ptype)
            group_label = GroupLabel.UNSPECIFIED
            if label == DocItemLabel.FORM:
                group_label = GroupLabel.FORM_AREA
            elif label == DocItemLabel.KEY_VALUE_REGION:
                group_label = GroupLabel.KEY_VALUE_AREA

            container_el = doc.add_group(label=group_label)

            _add_child_elements(container_el, doc, obj, pelem)
        elif "text" in obj:
            text = obj["text"][span_i:span_j]

            type_label = pelem["type"]
            name_label = pelem["name"]
            if update_name_label and len(props) > 0 and type_label == "paragraph":
                prop = props[
                    (props["type"] == "semantic") & (props["subj_path"] == iref)
                ]
                if len(prop) == 1 and prop.iloc[0]["confidence"] > 0.85:
                    name_label = prop.iloc[0]["label"]

            prov = ProvenanceItem(
                page_no=pelem["page"],
                charspan=(0, len(text)),
                bbox=BoundingBox.from_tuple(
                    pelem["bbox"], origin=CoordOrigin.BOTTOMLEFT
                ),
            )
            label = DocItemLabel(name_label)

            if label == DocItemLabel.LIST_ITEM:
                if current_list is None:
                    current_list = doc.add_group(label=GroupLabel.LIST, name="list")

                # TODO: Infer if this is a numbered or a bullet list item
                doc.add_list_item(
                    text=text, enumerated=False, prov=prov, parent=current_list
                )
            elif label == DocItemLabel.SECTION_HEADER:
                current_list = None

                doc.add_heading(text=text, prov=prov)
            elif label == DocItemLabel.CODE:
                current_list = None

                doc.add_code(text=text, prov=prov)
            elif label == DocItemLabel.FORMULA:
                current_list = None

                doc.add_text(label=DocItemLabel.FORMULA, text="", orig=text, prov=prov)
            elif label in [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER]:
                current_list = None

                doc.add_text(
                    label=DocItemLabel(name_label),
                    text=text,
                    prov=prov,
                    content_layer=ContentLayer.FURNITURE,
                )
            else:
                current_list = None

                doc.add_text(label=DocItemLabel(name_label), text=text, prov=prov)

    return doc


def _add_child_elements(container_el, doc, obj, pelem):
    payload = obj.get("payload")
    if payload is not None:
        children = payload.get("children", [])

        for child in children:
            c_label = DocItemLabel(child["label"])
            c_bbox = BoundingBox.model_validate(child["bbox"]).to_bottom_left_origin(
                doc.pages[pelem["page"]].size.height
            )
            c_text = " ".join(
                [
                    cell["text"].replace("\x02", "-").strip()
                    for cell in child["cells"]
                    if len(cell["text"].strip()) > 0
                ]
            )

            c_prov = ProvenanceItem(
                page_no=pelem["page"], charspan=(0, len(c_text)), bbox=c_bbox
            )
            if c_label == DocItemLabel.LIST_ITEM:
                # TODO: Infer if this is a numbered or a bullet list item
                doc.add_list_item(parent=container_el, text=c_text, prov=c_prov)
            elif c_label == DocItemLabel.SECTION_HEADER:
                doc.add_heading(parent=container_el, text=c_text, prov=c_prov)
            else:
                doc.add_text(
                    parent=container_el, label=c_label, text=c_text, prov=c_prov
                )
