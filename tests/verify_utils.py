# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from docling_core.types.doc import (
    CodeItem,
    DocItem,
    DoclingDocument,
    FormulaItem,
    PictureItem,
    TableItem,
    TextItem,
)
from docling_core.types.doc.base import BoundingBox
from PIL import Image as PILImage
from pydantic import BaseModel, TypeAdapter

from docling.datamodel.base_models import ConversionStatus, Page
from docling.datamodel.document import ConversionResult

from .groundtruth_paths import GroundTruthPaths

COORD_PREC = 2  # decimal places for coordinates
CONFID_PREC = 3  # decimal places for confidence
STRICT_BBOX_TOL_RATIO = 0.0025  # allow minor cross-platform layout variance
FUZZY_BBOX_TOL_RATIO = (
    0.08  # OCR/image output varies more, but gross shifts should fail
)
IMAGE_SIZE_TOL_RATIO = 0.015  # allow ~1.5% cross-platform image size variance


def _normalize_newlines(text: str) -> str:
    """Drop stray CR characters extracted from source documents.

    Some backends (e.g. pypdfium2) carry a literal CRLF out of the PDF into item text.
    Ground truth is stored and compared with LF only, so the exported text is normalized
    on both the generate and the verify path.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


class _TestPagesMeta(BaseModel):
    num_cells: int

    @classmethod
    def from_page(cls, page: Page):
        return cls(num_cells=len(page.cells))


def _assert_bbox_close(
    *,
    true_bbox: BoundingBox,
    pred_bbox: BoundingBox,
    fuzzy: bool,
    page_extent: Optional[float],
    pdf_filename: str,
):
    """Compare bbox coordinates at the same precision used in serialized fixtures."""

    tol_ratio = FUZZY_BBOX_TOL_RATIO if fuzzy else STRICT_BBOX_TOL_RATIO
    tol = max(10 ** (-COORD_PREC), (page_extent or 0.0) * tol_ratio)

    assert true_bbox.coord_origin == pred_bbox.coord_origin, (
        f"[{pdf_filename}] BBox coord_origin mismatch"
    )

    for label, true_value, pred_value in (
        ("left", true_bbox.l, pred_bbox.l),
        ("top", true_bbox.t, pred_bbox.t),
        ("right", true_bbox.r, pred_bbox.r),
        ("bottom", true_bbox.b, pred_bbox.b),
    ):
        true_rounded = round(true_value, COORD_PREC)
        pred_rounded = round(pred_value, COORD_PREC)
        diff = abs(true_rounded - pred_rounded)

        assert math.isclose(true_rounded, pred_rounded, rel_tol=0.0, abs_tol=tol), (
            f"[{pdf_filename}] BBox {label} mismatch:"
            f" {true_rounded} vs {pred_rounded}"
            f" (raw pred: {pred_value}, diff: {diff:.2f}, tol: {tol:.2f})"
        )


def _describe_item(item: DocItem) -> str:
    """Compact identification of a doc item, for use in assertion messages."""
    parts: list[str] = [item.self_ref]
    if item.prov:
        prov = item.prov[0]
        parts.append(f"page {prov.page_no}")
        if prov.bbox is not None:
            bbox = prov.bbox
            parts.append(
                f"bbox=({bbox.l:.1f}, {bbox.t:.1f}, {bbox.r:.1f}, {bbox.b:.1f})"
            )
    if isinstance(item, TextItem):
        text = item.text if len(item.text) <= 60 else f"{item.text[:57]}..."
        parts.append(repr(text))
    return ", ".join(parts)


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
        # get a better view where it goes wrong ...
        lines_gt = gt.split("\n")
        lines_pr = pred.split("\n")
        for i, line_gt in enumerate(lines_gt):
            if i < len(lines_pr):
                assert line_gt == lines_pr[i], f"{line_gt}!={lines_pr[i]}"

        assert gt == pred, f"{gt}!={pred}"
    else:
        dist = levenshtein(gt, pred)
        diff = dist / len(gt)
        assert diff < fuzzy_threshold, f"{gt}!~{pred}"
    return True


def verify_cells(
    doc_pred_pages: list[_TestPagesMeta], doc_true_pages: list[_TestPagesMeta]
):
    assert len(doc_pred_pages) == len(doc_true_pages), (
        "pred- and true-doc do not have the same number of pages"
    )

    for pid, page_true_item in enumerate(doc_true_pages):
        num_true_cells = page_true_item.num_cells
        num_pred_cells = doc_pred_pages[pid].num_cells

        assert num_true_cells == num_pred_cells, (
            f"num_true_cells!=num_pred_cells {num_true_cells}!={num_pred_cells}"
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
) -> bool:
    """Compare image properties between a ground-truth image and a predicted image.

    The image mode must match exactly.  The pixel dimensions are compared with a
    percentage-based tolerance (IMAGE_SIZE_TOL_RATIO) to accommodate minor
    cross-platform rendering differences.

    Image bytes are not compared because they can differ significantly across
    platforms even for visually identical images.

    Args:
        true_image: Ground-truth PIL image loaded from the reference fixture.
        pred_item: Predicted PIL image produced by the conversion under test.

    Returns:
        True if all assertions pass.
    """
    assert pred_item is not None, "predicted image is None"

    # Check image mode (should be exact)
    assert true_image.mode == pred_item.mode, (
        f"Image mode mismatch: {true_image.mode} vs {pred_item.mode}"
    )

    # Check image size with a percentage-based tolerance
    true_width, true_height = true_image.size
    pred_width, pred_height = pred_item.size

    width_diff = abs(true_width - pred_width)
    height_diff = abs(true_height - pred_height)

    width_diff_ratio = width_diff / true_width if true_width > 0 else 0
    height_diff_ratio = height_diff / true_height if true_height > 0 else 0

    assert width_diff_ratio <= IMAGE_SIZE_TOL_RATIO, (
        f"Image width mismatch: {true_width} vs {pred_width} "
        f"(diff: {width_diff} pixels, {width_diff_ratio:.1%} vs tolerance {IMAGE_SIZE_TOL_RATIO:.1%})"
    )
    assert height_diff_ratio <= IMAGE_SIZE_TOL_RATIO, (
        f"Image height mismatch: {true_height} vs {pred_height} "
        f"(diff: {height_diff} pixels, {height_diff_ratio:.1%} vs tolerance {IMAGE_SIZE_TOL_RATIO:.1%})"
    )

    return True


def verify_docitems(
    *,
    doc_pred: DoclingDocument,
    doc_true: DoclingDocument,
    fuzzy: bool,
    pdf_filename: str = "",
):
    """Verify that two DoclingDocuments contain equivalent content.

    For every item pair the following properties are checked:

    - Label: item type must match exactly.
    - Provenance: page number and bounding-box coordinates must match.
      BBox tolerance is controlled by the fuzzy flag (STRICT_BBOX_TOL_RATIO vs
      FUZZY_BBOX_TOL_RATIO).
    - Text (TextItem): exact match in strict mode; Levenshtein distance below
      threshold in fuzzy mode.
    - Tables (TableItem): row/column counts and cell text must match, subject to
      the same text-fuzziness rules.
    - Pictures (PictureItem): only checked when the ground-truth image is
      present. When fuzzy is False, image mode and pixel dimensions are verified
      (see verify_picture_image_v2). When fuzzy is True, image sizes are not
      compared — only the existence of the predicted image is asserted. This is
      intentional for backends that rely on LibreOffice (MsWordDocumentBackend,
      MsExcelDocumentBackend, MsPowerPointDocumentBackend), whose rendered pixel
      dimensions are not stable across LibreOffice versions and
      operating-system installations.
    - Code (CodeItem): code_language must match exactly.

    Args:
        doc_pred: The DoclingDocument produced by the conversion under test.
        doc_true: The reference DoclingDocument loaded from the ground-truth fixture.
        fuzzy: When True, apply relaxed tolerances for text and bboxes, and skip
            image size comparison entirely (see Pictures note above).
        pdf_filename: Source filename included in assertion messages for easier
            debugging.

    Returns:
        True if all assertions pass.
    """

    assert len(doc_pred.texts) == len(doc_true.texts), (
        f"[{pdf_filename}] Text lengths do not match: {len(doc_pred.texts)} != {len(doc_true.texts)}"
    )

    assert len(doc_true.tables) == len(doc_pred.tables), (
        f"[{pdf_filename}] document has different count of tables than expected."
    )
    assert len(doc_true.pictures) == len(doc_pred.pictures), (
        f"[{pdf_filename}] Picture lengths do not match: {len(doc_true.pictures)} != {len(doc_pred.pictures)}"
    )

    for item_no, ((true_item, _true_level), (pred_item, _pred_level)) in enumerate(
        zip(doc_true.iterate_items(), doc_pred.iterate_items())
    ):
        if not isinstance(true_item, DocItem):
            continue
        assert isinstance(pred_item, DocItem), (
            f"[{pdf_filename}] Test item is not a DocItem"
        )

        # Validate type
        assert true_item.label == pred_item.label, (
            f"[{pdf_filename}] Object label does not match at item {item_no}:\n"
            f"  groundtruth: {true_item.label.value} ({_describe_item(true_item)})\n"
            f"  predicted  : {pred_item.label.value} ({_describe_item(pred_item)})"
        )

        # Validate provenance
        assert len(true_item.prov) == len(pred_item.prov), (
            f"[{pdf_filename}] Length of prov mismatch at item {item_no} "
            f"({true_item.label.value}): "
            f"groundtruth {len(true_item.prov)} != predicted {len(pred_item.prov)}"
        )
        if len(true_item.prov) > 0:
            true_prov = true_item.prov[0]
            pred_prov = pred_item.prov[0]
            true_page = doc_true.pages.get(true_prov.page_no)
            pred_page = doc_pred.pages.get(pred_prov.page_no)

            assert true_prov.page_no == pred_prov.page_no, (
                f"[{pdf_filename}] Page provenance mismatch at item {item_no} "
                f"({true_item.label.value}): "
                f"groundtruth page {true_prov.page_no} != "
                f"predicted page {pred_prov.page_no}"
            )
            assert (true_prov.bbox is None) == (pred_prov.bbox is None), (
                f"[{pdf_filename}] BBox presence mismatch at item {item_no} "
                f"({true_item.label.value}): "
                f"groundtruth bbox={true_prov.bbox is not None}, "
                f"predicted bbox={pred_prov.bbox is not None}"
            )

            if true_prov.bbox is not None and pred_prov.bbox is not None:
                _assert_bbox_close(
                    true_bbox=true_prov.bbox,
                    pred_bbox=pred_prov.bbox,
                    fuzzy=fuzzy,
                    page_extent=(
                        max(page.size.width, page.size.height)
                        if (page := true_page or pred_page) is not None
                        else None
                    ),
                    pdf_filename=pdf_filename,
                )

        # Validate source
        assert bool(true_item.source) == bool(pred_item.source), (
            "Source exists mismatch"
        )
        if true_item.source:
            true_source = true_item.source[0]
            pred_source = pred_item.source[0]
            assert true_source.start_time == pred_source.start_time, (
                "TrackProvenance start time mismatch"
            )
            assert true_source.end_time == pred_source.end_time, (
                "TrackProvenance end time mismatch"
            )

        # Validate text content
        if isinstance(true_item, TextItem):
            assert isinstance(pred_item, TextItem), (
                f"[{pdf_filename}] Test item should be a TextItem {true_item=} {pred_item=} "
            )

            assert verify_text(true_item.text, pred_item.text, fuzzy=fuzzy)

        # Validate table content
        if isinstance(true_item, TableItem):
            assert isinstance(pred_item, TableItem), (
                f"[{pdf_filename}] Test item should be a TableItem"
            )
            assert verify_table_v2(true_item, pred_item, fuzzy=fuzzy), (
                f"[{pdf_filename}] Tables not matching"
            )

        # Validate picture content
        if isinstance(true_item, PictureItem):
            assert isinstance(pred_item, PictureItem), (
                f"[{pdf_filename}] Test item should be a PictureItem"
            )

            true_image = true_item.get_image(doc=doc_true)
            if true_image is not None:
                if fuzzy:
                    # In fuzzy mode (used for LibreOffice-based backends whose
                    # rendered image dimensions vary across platforms) we only
                    # verify that the predicted image exists, not its size.
                    assert pred_item.get_image(doc=doc_pred) is not None, (
                        f"[{pdf_filename}] Picture image is missing"
                    )
                else:
                    assert verify_picture_image_v2(
                        true_image, pred_item.get_image(doc=doc_pred)
                    ), f"[{pdf_filename}] Picture image mismatch"
        # TODO: check picture annotations

        # Validate code content
        if isinstance(true_item, CodeItem):
            assert isinstance(pred_item, CodeItem), (
                f"[{pdf_filename}] Test item should be a CodeItem"
            )
            assert true_item.code_language == pred_item.code_language, (
                f"[{pdf_filename}] Code language mismatch"
            )

        # Validate formula content
        if isinstance(true_item, FormulaItem):
            assert isinstance(pred_item, FormulaItem), (
                f"[{pdf_filename}] Test item should be a FormulaItem"
            )

    return True


def verify_md(doc_pred_md: str, doc_true_md: str, fuzzy: bool):
    return verify_text(doc_true_md, doc_pred_md, fuzzy)


def verify_dt(doc_pred_dt: str, doc_true_dt: str, fuzzy: bool):
    return verify_text(doc_true_dt, doc_pred_dt, fuzzy)


class VerificationFailure(BaseModel):
    """A single ground-truth check of a conversion result that did not match."""

    check: str
    message: str


def check_conversion_result_v2(
    gt: GroundTruthPaths,
    doc_result: ConversionResult,
    generate: bool = False,
    fuzzy: bool = False,
    verify_doctags: bool = True,
    indent: int = 2,
) -> list[VerificationFailure]:
    """Compare a conversion result against ground truth and collect every mismatch.

    Unlike `verify_conversion_result_v2`, a failing check does not stop the checks
    after it, so a caller can report all the ways a document deviates at once.
    """
    PageMetaList = TypeAdapter(list[_TestPagesMeta])

    input_path = doc_result.input.file
    failures: list[VerificationFailure] = []

    def run_check(check: str, verify: Callable[[], bool], mismatch: str) -> None:
        """Record the outcome of one check instead of raising on mismatch."""
        try:
            matches = verify()
        except AssertionError as exc:
            failures.append(VerificationFailure(check=check, message=str(exc)))
        else:
            if not matches:
                failures.append(VerificationFailure(check=check, message=mismatch))

    if doc_result.status != ConversionStatus.SUCCESS:
        failures.append(
            VerificationFailure(
                check="status",
                message=f"Doc {input_path} did not convert successfully.",
            )
        )
        return failures

    doc_pred_pages: list[Page] = doc_result.pages
    doc_pred_pages_meta: list[_TestPagesMeta] = [
        _TestPagesMeta.from_page(page) for page in doc_pred_pages
    ]
    doc_pred: DoclingDocument = doc_result.document
    doc_pred_md = _normalize_newlines(
        doc_result.document.export_to_markdown(compact_tables=True)
    )
    doc_pred_dt = _normalize_newlines(doc_result.document.export_to_doctags())

    pages_path = gt.pages_meta
    json_path = gt.doc_json
    md_path = gt.md
    dt_path = gt.doctags

    if generate:  # only used when re-generating truth
        pages_path.parent.mkdir(parents=True, exist_ok=True)

        pages_data = PageMetaList.dump_json(doc_pred_pages_meta, indent=indent)
        with open(pages_path, mode="w", encoding="utf-8") as fw:
            fw.write(pages_data.decode())

        json_path.parent.mkdir(parents=True, exist_ok=True)
        doc_pred.save_as_json(
            json_path,
            indent=indent,
            coord_precision=COORD_PREC,
            confid_precision=CONFID_PREC,
        )

        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, mode="w", encoding="utf-8", newline="") as fw:
            fw.write(doc_pred_md)

        if verify_doctags:
            dt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dt_path, mode="w", encoding="utf-8", newline="") as fw:
                fw.write(doc_pred_dt)
    else:  # default branch in test
        with open(pages_path, encoding="utf-8") as fr:
            doc_true_pages_meta = PageMetaList.validate_json(fr.read())

        with open(json_path, encoding="utf-8") as fr:
            doc_true: DoclingDocument = DoclingDocument.model_validate_json(fr.read())

        with open(md_path, encoding="utf-8", newline="") as fr:
            doc_true_md = fr.read()

        if verify_doctags:
            with open(dt_path, encoding="utf-8", newline="") as fr:
                doc_true_dt = fr.read()

        if not fuzzy:
            run_check(
                "cells",
                lambda: verify_cells(doc_pred_pages_meta, doc_true_pages_meta),
                f"Mismatch in PDF cell prediction for {input_path}",
            )

        run_check(
            "docitems",
            lambda: verify_docitems(
                doc_pred=doc_pred,
                doc_true=doc_true,
                fuzzy=fuzzy,
                pdf_filename=input_path.name,
            ),
            f"verify_docling_document(doc_pred, doc_true) mismatch for {input_path}",
        )

        run_check(
            "markdown",
            lambda: verify_md(doc_pred_md, doc_true_md, fuzzy=fuzzy),
            f"Mismatch in Markdown prediction for {input_path}",
        )

        if verify_doctags:
            run_check(
                "doctags",
                lambda: verify_dt(doc_pred_dt, doc_true_dt, fuzzy=fuzzy),
                f"Mismatch in DocTags prediction for {input_path}",
            )

    return failures


def verify_conversion_result_v2(
    gt: GroundTruthPaths,
    doc_result: ConversionResult,
    generate: bool = False,
    fuzzy: bool = False,
    verify_doctags: bool = True,
    indent: int = 2,
):
    failures = check_conversion_result_v2(
        gt=gt,
        doc_result=doc_result,
        generate=generate,
        fuzzy=fuzzy,
        verify_doctags=verify_doctags,
        indent=indent,
    )
    if failures:
        raise AssertionError(
            "\n".join(f"[{failure.check}] {failure.message}" for failure in failures)
        )


def verify_document(
    pred_doc: DoclingDocument, gtfile: str, generate: bool = False, fuzzy: bool = False
):
    if not os.path.exists(gtfile) or generate:
        with open(gtfile, mode="w", encoding="utf-8") as fw:
            pred_dict = pred_doc.export_to_dict(
                coord_precision=COORD_PREC,
                confid_precision=CONFID_PREC,
            )
            json.dump(pred_dict, fw, ensure_ascii=False, indent=2)

        return True
    else:
        with open(gtfile, encoding="utf-8") as fr:
            true_doc = DoclingDocument.model_validate_json(fr.read())

        return verify_docitems(
            doc_pred=pred_doc, doc_true=true_doc, fuzzy=fuzzy, pdf_filename=gtfile
        )


def verify_export(
    pred_text: str, gtfile: str, generate: bool = False, fuzzy: bool = False
) -> bool:
    file = Path(gtfile)

    pred_text = _normalize_newlines(pred_text)

    if not file.exists() or generate:
        with file.open(mode="w", encoding="utf-8", newline="") as fw:
            fw.write(pred_text)
        return True

    with file.open(encoding="utf-8", newline="") as fr:
        true_text = fr.read()

    if fuzzy:
        return verify_text(true_text, pred_text, fuzzy=True)

    return pred_text == true_text
