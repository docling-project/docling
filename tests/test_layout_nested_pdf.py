# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Keep paragraph ownership through PDF extraction and document assembly."""

from collections import Counter
from dataclasses import dataclass
from io import BytesIO

import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel
from docling_core.types.doc.page import TextCell

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
    InputFormat,
    LayoutPrediction,
    Page,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import LayoutPostprocessorOptions
from docling.models.stages.page_assemble.page_assemble_model import (
    PageAssembleModel,
    PageAssembleOptions,
)
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)
from docling.utils.layout_postprocessor import LayoutPostprocessor


@dataclass(frozen=True)
class _Paragraph:
    lines: tuple[str, ...]
    left: float
    baseline: float


def _paragraphs(page_no: int, columns: int, gap: int) -> list[_Paragraph]:
    result = []
    for column in range(columns):
        for paragraph in range(2):
            prefix = f"P{page_no}C{column + 1}A{paragraph + 1}"
            result.append(
                _Paragraph(
                    lines=(
                        f"{prefix} Start.",
                        f"{prefix} longer middle line.",
                        f"{prefix} Finish.",
                    ),
                    left=40 + column * 290,
                    baseline=80 + paragraph * (38 + gap),
                )
            )
    return result


def _pdf_bytes(pages: list[list[_Paragraph]]) -> bytes:
    """Build a real multipage PDF without a renderer or a fixture dependency."""
    kids = " ".join(f"{4 + 2 * index} 0 R" for index in range(len(pages)))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for page_no, paragraphs in enumerate(pages, 1):
        rows = [(40.0, 40.0, f"PAGE {page_no}")]
        rows.extend(
            (paragraph.left, paragraph.baseline + 14 * line_no, text)
            for paragraph in paragraphs
            for line_no, text in enumerate(paragraph.lines)
        )
        content = "\n".join(
            f"BT /F1 10 Tf 1 0 0 1 {left:g} {800 - baseline:g} Tm ({text}) Tj ET"
            for left, baseline, text in rows
        ).encode("ascii")
        objects.extend(
            [
                (
                    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] "
                    "/Resources << /Font << /F1 3 0 R >> >> "
                    f"/Contents {5 + 2 * (page_no - 1)} 0 R >>"
                ).encode(),
                f"<< /Length {len(content)} >>\nstream\n".encode()
                + content
                + b"\nendstream",
            ]
        )

    data = bytearray(b"%PDF-1.7\n")
    offsets = []
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode()
    )
    return bytes(data)


def _bounds(cells: list[TextCell]) -> BoundingBox:
    boxes = [cell.rect.to_bounding_box() for cell in cells]
    return BoundingBox(
        l=min(box.l for box in boxes),
        t=min(box.t for box in boxes),
        r=max(box.r for box in boxes),
        b=max(box.b for box in boxes),
    )


def _layout(
    page: Page,
    paragraphs: list[_Paragraph],
    broad_confidence: float,
    partial_detection: bool,
) -> list[Cluster]:
    by_text = {cell.text.strip(): cell for cell in page.cells}
    expected_lines = [f"PAGE {page.page_no}"] + [
        line for paragraph in paragraphs for line in paragraph.lines
    ]
    assert Counter(cell.text.strip() for cell in page.cells) == Counter(expected_lines)
    clusters = [
        Cluster(
            id=0,
            label=DocItemLabel.SECTION_HEADER,
            confidence=0.95,
            bbox=_bounds([by_text[expected_lines[0]]]),
        )
    ]
    for start in range(0, len(paragraphs), 2):
        column_cells = []
        for paragraph in paragraphs[start : start + 2]:
            cells = [by_text[line] for line in paragraph.lines]
            column_cells.extend(cells)
            box = _bounds(cells)
            # The middle line is slightly wider than its paragraph detection.
            if partial_detection and paragraph is paragraphs[start + 1]:
                continue
            clusters.append(
                Cluster(
                    id=len(clusters),
                    label=DocItemLabel.TEXT,
                    confidence=0.35 if partial_detection else 0.9,
                    bbox=BoundingBox(
                        l=box.l - 1, t=box.t - 1, r=box.r - 1, b=box.b + 1
                    ),
                )
            )
        box = _bounds(column_cells)
        # A competing prediction covers both nearby paragraphs.
        clusters.append(
            Cluster(
                id=len(clusters),
                label=DocItemLabel.TEXT,
                confidence=broad_confidence,
                bbox=BoundingBox(l=box.l - 2, t=box.t - 2, r=box.r + 2, b=box.b + 2),
            )
        )
    return clusters


@pytest.mark.parametrize("page_count", [1, 2])
@pytest.mark.parametrize("columns", [1, 2])
@pytest.mark.parametrize(
    "paragraph_gap,broad_confidence,partial_detection",
    [(4, 0.5, False), (20, 0.5, False), (20, 0.95, False), (20, 0.95, True)],
)
def test_nested_pdf_predictions_preserve_paragraphs(
    page_count: int,
    columns: int,
    paragraph_gap: int,
    broad_confidence: float,
    partial_detection: bool,
) -> None:
    paragraphs_by_page = [
        _paragraphs(page_no, columns, paragraph_gap)
        for page_no in range(1, page_count + 1)
    ]
    in_doc = InputDocument(
        path_or_stream=BytesIO(_pdf_bytes(paragraphs_by_page)),
        filename="nested-text.pdf",
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
    )
    backend = in_doc._backend
    assert isinstance(backend, PyPdfiumDocumentBackend)
    conv_res = ConversionResult(input=in_doc)
    assembled = AssembledUnit()
    expected = []
    try:
        for page_index, paragraphs in enumerate(paragraphs_by_page):
            page_backend = backend.load_page(page_index)
            try:
                page = Page(
                    page_no=page_index + 1,
                    size=page_backend.get_size(),
                    parsed_page=page_backend.get_segmented_page(),
                )
                page._backend = page_backend
                conv_res.pages.append(page)
                clusters = LayoutPostprocessor(
                    page,
                    _layout(page, paragraphs, broad_confidence, partial_detection),
                    LayoutPostprocessorOptions(),
                ).postprocess()
                page.predictions.layout = LayoutPrediction(clusters=clusters)
                list(PageAssembleModel(PageAssembleOptions())(conv_res, [page]))
                assert page.assembled is not None
                assembled.elements.extend(page.assembled.elements)
                assembled.body.extend(page.assembled.body)
                assembled.headers.extend(page.assembled.headers)
                expected.append(
                    (DocItemLabel.SECTION_HEADER, page.page_no, f"PAGE {page.page_no}")
                )
                expected.extend(
                    (DocItemLabel.TEXT, page.page_no, " ".join(paragraph.lines))
                    for paragraph in paragraphs
                )
            finally:
                page_backend.unload()
        conv_res.assembled = assembled
        document = ReadingOrderModel(ReadingOrderOptions())(conv_res)
        actual = [
            (item.label, item.prov[0].page_no, item.text) for item in document.texts
        ]
        # Separate content coverage from paragraph ownership and reading order.
        assert Counter(
            word for _, _, text in actual for word in text.split()
        ) == Counter(word for _, _, text in expected for word in text.split())
        assert actual == expected
    finally:
        backend.unload()
