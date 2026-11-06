# %% [markdown]
# <!--
# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT
# -->
# # Standalone PDFium rendering and ONNX layout inference
#
# Render one PDF page, extract its native text cells, run the official Heron
# ONNX export on CPU, and assign cells to layout clusters. This bypasses
# `DocumentConverter` and does not use Torch, Transformers, or PyMuPDF.
# It produces layout clusters, not a complete `DoclingDocument`.
#
# ## Install and run
#
# From a Docling checkout, use a separate environment (Python 3.10+):
#
# ```sh
# python -m venv .venv-onnx
# . .venv-onnx/bin/activate
# pip install -e '.[convert-core,format-pdf-pypdfium2]' onnxruntime huggingface-hub
# python -c 'from huggingface_hub import snapshot_download; snapshot_download("docling-project/docling-layout-heron-onnx", revision="40bde044036bb181c130ddf6c51792187268748f", local_dir="heron-onnx", allow_patterns=["model.onnx", "config.json", "preprocessor_config.json"])'
# python docs/examples/standalone_onnx_layout.py input.pdf heron-onnx output --page 1 --dpi 144
# ```
#
# The model download requires network access. Inference uses only local files.
# Use the CPU `onnxruntime` package explicitly: the `models-onnxruntime` extra
# selects a GPU package on some platforms. Do not install the full `docling`
# bundle or `models-local` for this environment. Common option modules can probe
# for optional Torch/Transformers imports and catch ImportError when absent;
# neither package is required or loaded in this environment.
#
# ## Data and coordinates
#
# `--page` is one-based; `load_page` takes a zero-based index. `Page.size` and
# native `TextCell.rect` use the displayed page frame in PDF points (72/inch).
# The backend handles page rotation and converts cells to a top-left origin.
# `get_segmented_page()` supplies page geometry and textline cells; `Page.cells`
# reads those cells. This backend does not supply word or character cells.
#
# `get_page_image(scale=dpi/72)` returns round(width*scale) by
# round(height*scale) pixels. The backend temporarily renders at 1.5 times this
# scale before resizing. Limit DPI for memory-constrained deployments.
#
# This specific ONNX export takes uint8 RGB NCHW `images` resized to 640x640
# with bilinear interpolation, without rescaling or normalization, and int64
# `orig_target_sizes` in **width, height** order. It returns `labels`, `boxes`,
# `scores`; boxes are top-left `x0,y0,x1,y1` in the original rendered pixels.
# The export repeats `orig_target_sizes` as [width, height, width, height]
# to scale its xyxy boxes. The current Docling ONNX engine also supplies
# sizes in width, height order.
# These conventions are export-specific, not a generic ONNX detection format.
# Labels come from `config.json`'s `id2label`, matched to `DocItemLabel` names.
# Divide x and y by their actual image-to-page ratios, then clip to page bounds.
#
# The output directory must not exist. It contains `page.png` and `layout.json`:
# page size, render size, native cells, raw clusters in PDF points, and processed
# clusters with assigned cells and children. Save raw clusters before processing:
# `LayoutPostprocessor` mutates clusters and can update page cells. It applies
# label-specific thresholds, remaps labels (for example TITLE to SECTION_HEADER),
# resolves overlaps, and can create clusters for unassigned text.
#
# ## Compatibility and limits
#
# This example targets the APIs in Docling 2.148.0 and the pinned model revision
# above. Backend classes, `InputDocument._backend`, `Page`, `Cluster`, and
# `LayoutPostprocessor` are internal integration surfaces, not a new stable public
# standalone API. Pin and test Docling and model versions together.
# No helper API or schema change is needed for this flow.
#
# There is no OCR, table structure recognition, semantic reading order, or document
# assembly here. Scans can have detections but no native text. Text extraction and
# layout predictions can be imperfect. The historical cross-type overlap no-op
# discussed in #2143 is no longer the current implementation; current rules resolve
# specific near-coincident label pairs, not every possible overlap.
# For another model, adapt preprocessing, tensor names, label mapping, and box
# conversion to that model's contract. Do not reuse these assumptions blindly.

# %%
import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, Size
from numpy.typing import NDArray
from onnxruntime import InferenceSession
from PIL import Image

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import Cluster, InputFormat, Page
from docling.datamodel.document import InputDocument
from docling.datamodel.pipeline_options import LayoutPostprocessorOptions
from docling.utils.layout_postprocessor import LayoutPostprocessor

if TYPE_CHECKING:
    # The public re-export is conditional; this is its native definition.
    from onnxruntime.capi._pybind_state import SessionOptions
else:
    from onnxruntime import SessionOptions


def detections_to_clusters(
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    label_map: dict[int, DocItemLabel],
    page_size: Size,
    image_size: tuple[int, int],
) -> list[Cluster]:
    """Convert one page's Heron outputs from pixels to PDF points."""
    if (
        labels.ndim != 1
        or boxes.shape != (len(labels), 4)
        or scores.shape != labels.shape
    ):
        raise ValueError("Expected detection shapes (N,), (N, 4), (N,)")
    scale = np.array([page_size.width, page_size.height] * 2) / np.array(
        list(image_size) * 2
    )
    bounds = np.array([page_size.width, page_size.height] * 2)
    clusters = []
    for label, box, score in zip(labels, boxes, scores):
        if not np.isfinite(box).all() or not math.isfinite(float(score)):
            raise ValueError("Non-finite ONNX detection")
        coords = np.clip(box * scale, 0, bounds)
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            continue
        clusters.append(
            Cluster(
                id=len(clusters),
                label=label_map[int(label)],
                confidence=float(score),
                bbox=BoundingBox(
                    l=float(coords[0]),
                    t=float(coords[1]),
                    r=float(coords[2]),
                    b=float(coords[3]),
                    coord_origin=CoordOrigin.TOPLEFT,
                ),
            )
        )
    return clusters


def run(pdf: Path, model_dir: Path, output: Path, page_no: int, dpi: float) -> None:
    """Process one page with local model files and bounded CPU parallelism."""
    if page_no < 1 or not math.isfinite(dpi) or dpi <= 0:
        raise ValueError("Page must be >= 1 and DPI must be finite and positive")
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    pre = json.loads(
        (model_dir / "preprocessor_config.json").read_text(encoding="utf-8")
    )
    if (
        pre["do_rescale"]
        or pre["do_normalize"]
        or pre["do_pad"]
        or not pre["do_resize"]
        or pre["resample"] != 2
    ):
        raise ValueError("Expected the documented Heron ONNX preprocessing config")
    label_map = {
        int(key): DocItemLabel[value.upper()]
        for key, value in config["id2label"].items()
    }
    options = SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = InferenceSession(
        str(model_dir / "model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    in_doc = InputDocument(
        path_or_stream=pdf, format=InputFormat.PDF, backend=PyPdfiumDocumentBackend
    )
    backend = in_doc._backend
    if not isinstance(backend, PyPdfiumDocumentBackend):
        raise ValueError("Could not open the PDF with the PDFium backend")
    try:
        if not in_doc.valid or page_no > backend.page_count():
            raise ValueError("Invalid PDF or requested page is out of range")
        page_backend = backend.load_page(page_no - 1)
        try:
            if not page_backend.is_valid():
                raise ValueError("Could not load the requested page")
            page_size = page_backend.get_size()
            parsed_page = page_backend.get_segmented_page()
            if parsed_page is None:
                raise ValueError("Could not extract page geometry and text")
            image = page_backend.get_page_image(scale=dpi / 72).convert("RGB")
            page = Page(page_no=page_no, size=page_size, parsed_page=parsed_page)
            native_cells = [cell.model_dump(mode="json") for cell in page.cells]
        finally:
            page_backend.unload()
    finally:
        backend.unload()

    resized = image.resize(
        (pre["size"]["width"], pre["size"]["height"]), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)[None]
    labels, boxes, scores = session.run(
        ["labels", "boxes", "scores"],
        {"images": pixels, "orig_target_sizes": np.array([image.size], dtype=np.int64)},
    )
    if (
        not isinstance(labels, np.ndarray)
        or not isinstance(boxes, np.ndarray)
        or not isinstance(scores, np.ndarray)
    ):
        raise TypeError("Expected dense NumPy ONNX outputs: labels, boxes, scores")
    if (
        labels.dtype != np.int64
        or boxes.dtype != np.float32
        or scores.dtype != np.float32
    ):
        raise TypeError("Expected ONNX output dtypes int64, float32, float32")
    if (
        labels.ndim != 2
        or labels.shape[0] != 1
        or boxes.shape != (*labels.shape, 4)
        or scores.shape != labels.shape
    ):
        raise ValueError("Expected batched ONNX outputs for one page")
    clusters = detections_to_clusters(
        cast(NDArray[np.int64], labels)[0],
        cast(NDArray[np.float32], boxes)[0],
        cast(NDArray[np.float32], scores)[0],
        label_map,
        page_size,
        image.size,
    )
    raw_clusters = [cluster.model_dump(mode="json") for cluster in clusters]
    processed = LayoutPostprocessor(
        page, clusters, LayoutPostprocessorOptions()
    ).postprocess()
    output.mkdir(parents=True, exist_ok=False)
    image.save(output / "page.png")
    result = {
        "page_no": page_no,
        "page_size_points": page_size.model_dump(mode="json"),
        "image_size_pixels": list(image.size),
        "dpi": dpi,
        "native_cells": native_cells,
        "raw_clusters": raw_clusters,
        "clusters": [cluster.model_dump(mode="json") for cluster in processed],
    }
    (output / "layout.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{len(raw_clusters)} raw detections, {len(processed)} processed clusters")
    print(f"{len(native_cells)} native text cells; results: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone PDFium + Heron ONNX layout"
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--dpi", type=float, default=144)
    args = parser.parse_args()
    run(args.pdf, args.model_dir, args.output, args.page, args.dpi)


if __name__ == "__main__":
    main()
