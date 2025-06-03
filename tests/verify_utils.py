import json
import os
import warnings
from pathlib import Path
from typing import List, Optional

import pytest
from docling_core.types.doc import (
    DocItem,
    DoclingDocument,
    PictureItem,
    TableItem,
    TextItem,
)
from docling_core.types.legacy_doc.document import ExportedCCSDocument as DsDocument
from PIL import Image as PILImage
from pydantic import TypeAdapter
from pydantic.json import pydantic_encoder

from docling.datamodel.base_models import ConversionStatus, Page
from docling.datamodel.document import ConversionResult


def levenshtein(str1: str, str2: str) -> int:
    # Ensure str1 is the shorter string to optimize memory usage
    if len(str1) > len(str2):
        str1, str2 = str2, str1

    # Previous and current row buffers
    previous_row = list(range(len(str2) + 1))
    current_row = [0] * (len(str2) + 1)

    # Compute the Levenshtein distance row by row
    for i, c1 in enumerate(str1, start=1):
        current_row[0] = i
        for j, c2 in enumerate(str2, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (c1 != c2)
            current_row[j] = min(insertions, deletions, substitutions)
        # Swap rows for the next iteration
        previous_row, current_row = current_row, previous_row

    # The result is in the last element of the previous row
    return previous_row[-1]


def verify_text(gt: str, pred: str, fuzzy: bool, fuzzy_threshold: float = 0.4):
    if len(gt) == 0 or not fuzzy:
        assert gt == pred, f"{gt}!={pred}"
    else:
        dist = levenshtein(gt, pred)
        diff = dist / len(gt)
        assert diff < fuzzy_threshold, f"{gt}!~{pred}"
    return True


def verify_cells(doc_pred_pages: List[Page], doc_true_pages: List[Page]):
    assert len(doc_pred_pages) == len(doc_true_pages), (
        "pred- and true-doc do not have the same number of pages"
    )

    for pid, page_true_item in enumerate(doc_true_pages):
        num_true_cells = len(page_true_item.cells)
        num_pred_cells = len(doc_pred_pages[pid].cells)

        assert num_true_cells == num_pred_cells, (
            f"num_true_cells!=num_pred_cells {num_true_cells}!={num_pred_cells}"
        )

        for cid, cell_true_item in enumerate(page_true_item.cells):
            cell_pred_item = doc_pred_pages[pid].cells[cid]

            true_text = cell_true_item.text
            pred_text = cell_pred_item.text
            assert true_text == pred_text, f"{true_text}!={pred_text}"

            true_bbox = cell_true_item.rect.to_bounding_box().as_tuple()
            pred_bbox = cell_pred_item.rect.to_bounding_box().as_tuple()
            assert true_bbox == pred_bbox, (
                f"bbox is not the same: {true_bbox} != {pred_bbox}"
            )

    return True


# def verify_maintext(doc_pred: DsDocument, doc_true: DsDocument):
#     assert doc_true.main_text is not None, "doc_true cannot be None"
#     assert doc_pred.main_text is not None, "doc_true cannot be None"
#
#     assert len(doc_true.main_text) == len(
#         doc_pred.main_text
#     ), f"document has different length of main-text than expected. {len(doc_true.main_text)}!={len(doc_pred.main_text)}"
#
#     for l, true_item in enumerate(doc_true.main_text):
#         pred_item = doc_pred.main_text[l]
#         # Validate type
#         assert (
#             true_item.obj_type == pred_item.obj_type
#         ), f"Item[{l}] type does not match. expected[{true_item.obj_type}] != predicted [{pred_item.obj_type}]"
#
#         # Validate text ceels
#         if isinstance(true_item, BaseText):
#             assert isinstance(
#                 pred_item, BaseText
#             ), f"{pred_item} is not a BaseText element, but {true_item} is."
#             assert true_item.text == pred_item.text
#
#     return True


def verify_tables_v1(doc_pred: DsDocument, doc_true: DsDocument, fuzzy: bool):
    if doc_true.tables is None:
        # No tables to check
        assert doc_pred.tables is None, "not expecting any table on this document"
        return True

    assert doc_pred.tables is not None, "no tables predicted, but expected in doc_true"

    # print("Expected number of tables: {}, result: {}".format(len(doc_true.tables), len(doc_pred.tables)))

    assert len(doc_true.tables) == len(doc_pred.tables), (
        "document has different count of tables than expected."
    )

    for ix, true_item in enumerate(doc_true.tables):
        pred_item = doc_pred.tables[ix]

        assert true_item.num_rows == pred_item.num_rows, (
            "table does not have the same #-rows"
        )
        assert true_item.num_cols == pred_item.num_cols, (
            "table does not have the same #-cols"
        )

        assert true_item.data is not None, "documents are expected to have table data"
        assert pred_item.data is not None, "documents are expected to have table data"

        print("True: \n", true_item.export_to_dataframe().to_markdown())
        print("Pred: \n", true_item.export_to_dataframe().to_markdown())

        for i, row in enumerate(true_item.data):
            for j, col in enumerate(true_item.data[i]):
                # print("true: ", true_item.data[i][j].text)
                # print("pred: ", pred_item.data[i][j].text)
                # print("")

                verify_text(
                    true_item.data[i][j].text, pred_item.data[i][j].text, fuzzy=fuzzy
                )

                assert true_item.data[i][j].obj_type == pred_item.data[i][j].obj_type, (
                    "table-cell does not have the same type"
                )

    return True


def verify_table_v2(true_item: TableItem, pred_item: TableItem, fuzzy: bool):
    assert true_item.data.num_rows == pred_item.data.num_rows, (
        "table does not have the same #-rows"
    )
    assert true_item.data.num_cols == pred_item.data.num_cols, (
        "table does not have the same #-cols"
    )

    assert true_item.data is not None, "documents are expected to have table data"
    assert pred_item.data is not None, "documents are expected to have table data"

    # print("True: \n", true_item.export_to_dataframe().to_markdown())
    # print("Pred: \n", true_item.export_to_dataframe().to_markdown())

    for i, row in enumerate(true_item.data.grid):
        for j, col in enumerate(true_item.data.grid[i]):
            # print("true: ", true_item.data[i][j].text)
            # print("pred: ", pred_item.data[i][j].text)
            # print("")

            verify_text(
                true_item.data.grid[i][j].text,
                pred_item.data.grid[i][j].text,
                fuzzy=fuzzy,
            )

            assert (
                true_item.data.grid[i][j].column_header
                == pred_item.data.grid[i][j].column_header
            ), "table-cell should be a column_header but prediction isn't"

            assert (
                true_item.data.grid[i][j].row_header
                == pred_item.data.grid[i][j].row_header
            ), "table-cell should be a row_header but prediction isn't"

            assert (
                true_item.data.grid[i][j].row_section
                == pred_item.data.grid[i][j].row_section
            ), "table-cell should be a row_section but prediction isn't"

    return True


def verify_picture_image_v2(
    true_image: PILImage.Image, pred_item: Optional[PILImage.Image]
):
    assert pred_item is not None, "predicted image is None"
    assert true_image.size == pred_item.size
    assert true_image.mode == pred_item.mode
    # assert true_image.tobytes() == pred_item.tobytes()
    return True


# def verify_output(doc_pred: DsDocument, doc_true: DsDocument):
#     #assert verify_maintext(doc_pred, doc_true), "verify_maintext(doc_pred, doc_true)"
#     assert verify_tables_v1(doc_pred, doc_true), "verify_tables(doc_pred, doc_true)"
#     return True


def verify_docitems(doc_pred: DoclingDocument, doc_true: DoclingDocument, fuzzy: bool):
    assert len(doc_pred.texts) == len(doc_true.texts), "Text lengths do not match."

    assert len(doc_true.tables) == len(doc_pred.tables), (
        "document has different count of tables than expected."
    )

    for (true_item, _true_level), (pred_item, _pred_level) in zip(
        doc_true.iterate_items(), doc_pred.iterate_items()
    ):
        if not isinstance(true_item, DocItem):
            continue
        assert isinstance(pred_item, DocItem), "Test item is not a DocItem"

        # Validate type
        assert true_item.label == pred_item.label, "Object label does not match."

        # Validate provenance
        assert len(true_item.prov) == len(pred_item.prov), "Length of prov mismatch"
        if len(true_item.prov) > 0:
            true_prov = true_item.prov[0]
            pred_prov = pred_item.prov[0]

            assert true_prov.page_no == pred_prov.page_no, "Page provenance mistmatch"

            # TODO: add bbox check with tolerance

        # Validate text content
        if isinstance(true_item, TextItem):
            assert isinstance(pred_item, TextItem), (
                "Test item is not a TextItem as the expected one "
                f"{true_item=} "
                f"{pred_item=} "
            )

            assert verify_text(true_item.text, pred_item.text, fuzzy=fuzzy)

        # Validate table content
        if isinstance(true_item, TableItem):
            assert isinstance(pred_item, TableItem), (
                "Test item is not a TableItem as the expected one"
            )
            assert verify_table_v2(true_item, pred_item, fuzzy=fuzzy), (
                "Tables not matching"
            )

        # Validate picture content
        if isinstance(true_item, PictureItem):
            assert isinstance(pred_item, PictureItem), (
                "Test item is not a PictureItem as the expected one"
            )

            true_image = true_item.get_image(doc=doc_true)
            pred_image = true_item.get_image(doc=doc_pred)
            if true_image is not None:
                assert verify_picture_image_v2(true_image, pred_image), (
                    "Picture image mismatch"
                )

            # TODO: check picture annotations

    return True


def verify_md(doc_pred_md: str, doc_true_md: str, fuzzy: bool):
    return verify_text(doc_true_md, doc_pred_md, fuzzy)


def verify_dt(doc_pred_dt: str, doc_true_dt: str, fuzzy: bool):
    return verify_text(doc_true_dt, doc_pred_dt, fuzzy)


def verify_conversion_result_v1(
    input_path: Path,
    doc_result: ConversionResult,
    generate: bool = False,
    ocr_engine: Optional[str] = None,
    fuzzy: bool = False,
    indent: int = 2,
):
    PageList = TypeAdapter(List[Page])

    assert doc_result.status == ConversionStatus.SUCCESS, (
        f"Doc {input_path} did not convert successfully."
    )

    doc_pred_pages: List[Page] = doc_result.pages
    with pytest.warns(DeprecationWarning, match="Use document instead"):
        doc_pred: DsDocument = doc_result.legacy_document
        doc_pred_md = doc_result.legacy_document.export_to_markdown()
        doc_pred_dt = doc_result.legacy_document.export_to_document_tokens()

    engine_suffix = "" if ocr_engine is None else f".{ocr_engine}"

    gt_subpath = input_path.parent / "groundtruth" / "docling_v1" / input_path.name
    if str(input_path.parent).endswith("pdf"):
        gt_subpath = (
            input_path.parent.parent / "groundtruth" / "docling_v1" / input_path.name
        )

    pages_path = gt_subpath.with_suffix(f"{engine_suffix}.pages.json")
    json_path = gt_subpath.with_suffix(f"{engine_suffix}.json")
    md_path = gt_subpath.with_suffix(f"{engine_suffix}.md")
    dt_path = gt_subpath.with_suffix(f"{engine_suffix}.doctags.txt")

    if generate:  # only used when re-generating truth
        pages_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pages_path, mode="w", encoding="utf-8") as fw:
            fw.write(
                json.dumps(doc_pred_pages, default=pydantic_encoder, indent=indent)
            )

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, mode="w", encoding="utf-8") as fw:
            fw.write(json.dumps(doc_pred, default=pydantic_encoder, indent=indent))

        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, mode="w", encoding="utf-8") as fw:
            fw.write(doc_pred_md)

        dt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dt_path, mode="w", encoding="utf-8") as fw:
            fw.write(doc_pred_dt)
    else:  # default branch in test
        with open(pages_path, encoding="utf-8") as fr:
            doc_true_pages = PageList.validate_json(fr.read())

        with open(json_path, encoding="utf-8") as fr:
            doc_true: DsDocument = DsDocument.model_validate_json(fr.read())

        with open(md_path, encoding="utf-8") as fr:
            doc_true_md = fr.read()

        with open(dt_path, encoding="utf-8") as fr:
            doc_true_dt = fr.read()

        if not fuzzy:
            assert verify_cells(doc_pred_pages, doc_true_pages), (
                f"Mismatch in PDF cell prediction for {input_path}"
            )

        # assert verify_output(
        #    doc_pred, doc_true
        # ), f"Mismatch in JSON prediction for {input_path}"

        assert verify_tables_v1(doc_pred, doc_true, fuzzy=fuzzy), (
            f"verify_tables(doc_pred, doc_true) mismatch for {input_path}"
        )

        assert verify_md(doc_pred_md, doc_true_md, fuzzy=fuzzy), (
            f"Mismatch in Markdown prediction for {input_path}"
        )

        assert verify_dt(doc_pred_dt, doc_true_dt, fuzzy=fuzzy), (
            f"Mismatch in DocTags prediction for {input_path}"
        )


def verify_conversion_result_v2(
    input_path: Path,
    doc_result: ConversionResult,
    generate: bool = False,
    ocr_engine: Optional[str] = None,
    fuzzy: bool = False,
    indent: int = 2,
):
    PageList = TypeAdapter(List[Page])

    assert doc_result.status == ConversionStatus.SUCCESS, (
        f"Doc {input_path} did not convert successfully."
    )

    doc_pred_pages: List[Page] = doc_result.pages
    doc_pred: DoclingDocument = doc_result.document
    doc_pred_md = doc_result.document.export_to_markdown()
    doc_pred_dt = doc_result.document.export_to_doctags()

    engine_suffix = "" if ocr_engine is None else f".{ocr_engine}"

    gt_subpath = input_path.parent / "groundtruth" / "docling_v2" / input_path.name
    if str(input_path.parent).endswith("pdf"):
        gt_subpath = (
            input_path.parent.parent / "groundtruth" / "docling_v2" / input_path.name
        )

    pages_path = gt_subpath.with_suffix(f"{engine_suffix}.pages.json")
    json_path = gt_subpath.with_suffix(f"{engine_suffix}.json")
    md_path = gt_subpath.with_suffix(f"{engine_suffix}.md")
    dt_path = gt_subpath.with_suffix(f"{engine_suffix}.doctags.txt")

    if generate:  # only used when re-generating truth
        pages_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pages_path, mode="w", encoding="utf-8") as fw:
            fw.write(
                json.dumps(doc_pred_pages, default=pydantic_encoder, indent=indent)
            )

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, mode="w", encoding="utf-8") as fw:
            fw.write(json.dumps(doc_pred, default=pydantic_encoder, indent=indent))

        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, mode="w", encoding="utf-8") as fw:
            fw.write(doc_pred_md)

        dt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dt_path, mode="w", encoding="utf-8") as fw:
            fw.write(doc_pred_dt)
    else:  # default branch in test
        with open(pages_path, encoding="utf-8") as fr:
            doc_true_pages = PageList.validate_json(fr.read())

        with open(json_path, encoding="utf-8") as fr:
            doc_true: DoclingDocument = DoclingDocument.model_validate_json(fr.read())

        with open(md_path, encoding="utf-8") as fr:
            doc_true_md = fr.read()

        with open(dt_path, encoding="utf-8") as fr:
            doc_true_dt = fr.read()

        if not fuzzy:
            assert verify_cells(doc_pred_pages, doc_true_pages), (
                f"Mismatch in PDF cell prediction for {input_path}"
            )

        # assert verify_output(
        #    doc_pred, doc_true
        # ), f"Mismatch in JSON prediction for {input_path}"

        assert verify_docitems(doc_pred, doc_true, fuzzy=fuzzy), (
            f"verify_docling_document(doc_pred, doc_true) mismatch for {input_path}"
        )

        assert verify_md(doc_pred_md, doc_true_md, fuzzy=fuzzy), (
            f"Mismatch in Markdown prediction for {input_path}"
        )

        assert verify_dt(doc_pred_dt, doc_true_dt, fuzzy=fuzzy), (
            f"Mismatch in DocTags prediction for {input_path}"
        )


def verify_document(pred_doc: DoclingDocument, gtfile: str, generate: bool = False):
    if not os.path.exists(gtfile) or generate:
        with open(gtfile, mode="w", encoding="utf-8") as fw:
            json.dump(pred_doc.export_to_dict(), fw, ensure_ascii=False, indent=2)

        return True
    else:
        with open(gtfile, encoding="utf-8") as fr:
            true_doc = DoclingDocument.model_validate_json(fr.read())

        return verify_docitems(pred_doc, true_doc, fuzzy=False)


def verify_export(pred_text: str, gtfile: str, generate: bool = False) -> bool:
    file = Path(gtfile)

    if not file.exists() or generate:
        with file.open(mode="w", encoding="utf-8") as fw:
            fw.write(pred_text)
        return True

    with file.open(encoding="utf-8") as fr:
        true_text = fr.read()

    return pred_text == true_text
