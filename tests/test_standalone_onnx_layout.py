# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, Size

pytest.importorskip("onnxruntime")
pytest.importorskip("pypdfium2")

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat, Page
from docling.datamodel.document import InputDocument
from docling.datamodel.pipeline_options import LayoutPostprocessorOptions
from docling.utils.layout_postprocessor import LayoutPostprocessor
from docs.examples.standalone_onnx_layout import detections_to_clusters, run


def test_non_square_pixel_boxes_map_to_points_and_clip() -> None:
    # Deliberately unequal axis ratios also cover render-size rounding.
    clusters = detections_to_clusters(
        labels=np.array([9, 8, 9]),
        boxes=np.array([[60, 160, 600, 800], [-20, -40, 1300, 1700], [4, 5, 4, 8]]),
        scores=np.array([0.95, 0.9, 0.8]),
        label_map={9: DocItemLabel.TEXT, 8: DocItemLabel.TABLE},
        page_size=Size(width=600, height=800),
        image_size=(1200, 1601),
    )
    assert len(clusters) == 2
    assert clusters[0].label == DocItemLabel.TEXT
    assert clusters[0].bbox.l == 30
    assert clusters[0].bbox.r == 300
    assert clusters[0].bbox.t == pytest.approx(160 * 800 / 1601)
    assert clusters[0].bbox.b == pytest.approx(800 * 800 / 1601)
    assert clusters[0].bbox.coord_origin == CoordOrigin.TOPLEFT
    assert clusters[1].bbox == BoundingBox(l=0, t=0, r=600, b=800)


def test_native_pdf_cells_are_assigned_after_pixel_conversion() -> None:
    source = Path(__file__).parent / "data/pdf/sources/2305.03393v1-pg9.pdf"
    document = InputDocument(
        path_or_stream=source,
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
    )
    backend = document._backend
    assert isinstance(backend, PyPdfiumDocumentBackend)
    try:
        pdf_page = backend.load_page(0)
        try:
            size = pdf_page.get_size()
            page = Page(page_no=1, size=size, parsed_page=pdf_page.get_segmented_page())
        finally:
            pdf_page.unload()
    finally:
        backend.unload()

    assert size.width != size.height
    native = page.cells[0]
    assert "Optimized Table Tokenization" in native.text
    bbox = native.rect.to_bounding_box()
    # Synthetic detections are a geometry/assignment control, not model inference.
    clusters = detections_to_clusters(
        labels=np.array([5]),
        boxes=np.array([[bbox.l - 1, bbox.t - 1, bbox.r + 1, bbox.b + 1]]) * 2,
        scores=np.array([0.99]),
        label_map={5: DocItemLabel.PAGE_HEADER},
        page_size=size,
        image_size=(int(size.width * 2), int(size.height * 2)),
    )
    processed = LayoutPostprocessor(
        page, clusters, LayoutPostprocessorOptions()
    ).postprocess()
    header = next(c for c in processed if c.label == DocItemLabel.PAGE_HEADER)
    assert [cell.text for cell in header.cells] == [native.text]
    assert header.bbox == bbox


@pytest.mark.parametrize("coordinate", [np.inf, -np.inf, np.nan])
def test_non_finite_detection_is_rejected_before_clipping(coordinate: float) -> None:
    with pytest.raises(ValueError, match="Non-finite"):
        detections_to_clusters(
            labels=np.array([9]),
            boxes=np.array([[coordinate, 10, 100, 200]]),
            scores=np.array([0.9]),
            label_map={9: DocItemLabel.TEXT},
            page_size=Size(width=600, height=800),
            image_size=(1200, 1600),
        )


def test_mismatched_detection_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="shapes"):
        detections_to_clusters(
            labels=np.array([9, 9]),
            boxes=np.array([[10, 10, 100, 200]]),
            scores=np.array([0.9]),
            label_map={9: DocItemLabel.TEXT},
            page_size=Size(width=600, height=800),
            image_size=(1200, 1600),
        )


def test_existing_output_is_preserved_before_loading_inputs(tmp_path: Path) -> None:
    output = tmp_path / "existing-output"
    output.mkdir()
    marker = output / "user-file.txt"
    marker.write_text("keep this file", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        run(tmp_path / "missing.pdf", tmp_path / "missing-model", output, 1, 144)
    assert marker.read_text(encoding="utf-8") == "keep this file"
    assert list(output.iterdir()) == [marker]
