import logging
from pathlib import Path
from typing import Optional

import pytest

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    NemotronOcrOptions,
    OcrMode,
    OcrOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.stages.ocr.nemotron_ocr_model import (
    NemotronOcrModel,
    resolve_nemotronocr_language,
)

from .groundtruth_paths import (
    GroundTruthPaths,
    get_regular_groundtruth_paths,
    resolve_gt_ocr_mode,
)
from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_conversion_result_v2

_log = logging.getLogger(__name__)

GENERATE_V2 = GEN_TEST_DATA


def get_nemotron_ocr_groundtruth_paths(
    input_path: Path,
    *,
    mode: OcrMode,
) -> GroundTruthPaths:
    """Build GT paths for nemotron OCR, organized by OCR mode.

    Each mode maps to a sub-directory named after ``mode.value``; files are tagged
    ``nemotron_ocr.<mode.value>``. DEFAULT shares PDF_AWARE_LAYOUT_REGIONS' ground truth.
    """
    model_name = "nemotron_ocr"
    mode = resolve_gt_ocr_mode(mode)
    gt_dir = input_path.parent.parent / "groundtruth" / model_name / mode.value
    tag = f"{model_name}.{mode.value}"
    return get_regular_groundtruth_paths(input_path, gt_dir=gt_dir, tag=tag)


def _nemotron_available() -> bool:
    """Reuse the model's own runtime gate; only the package import is extra.

    `NemotronOcrModel.validate_runtime` covers the OS/arch/Python/CUDA checks
    but does not verify that the optional `nemotron_ocr` package is installed,
    so that import is probed separately here.
    """
    try:
        import nemotron_ocr.inference.pipeline_v2
    except ImportError as exc:
        _log.warning("Nemotron OCR package is not importable: %s", exc)
        return False
    try:
        NemotronOcrModel.validate_runtime(
            AcceleratorOptions(device=AcceleratorDevice.AUTO)
        )
    except RuntimeError as exc:
        _log.warning("Nemotron OCR runtime validation failed: %s", exc)
        return False
    return True


pytestmark = [
    pytest.mark.ml_ocr,
    pytest.mark.skipif(
        not _nemotron_available(),
        reason="Nemotron OCR requires Linux x86_64, Python 3.12 and CUDA 13.x.",
    ),
]


def get_converter(ocr_options: OcrOptions, ocr_batch_size: Optional[int] = None):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.ocr_options = ocr_options
    # The threaded pipeline delivers pages to the OCR stage in groups of
    # `ocr_batch_size`. Raising it lets a single OCR batch span several pages,
    # which is required to exercise cross-page batching in NemotronOcrModel.
    if ocr_batch_size is not None:
        pipeline_options.ocr_batch_size = ocr_batch_size
    # Nemotron OCR requires a CUDA accelerator.
    pipeline_options.accelerator_options.device = AcceleratorDevice.CUDA

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=DoclingParseDocumentBackend,
            )
        }
    )

    return converter


@pytest.mark.parametrize(
    ("req_languages", "expected"),
    [
        # No request -> english default
        (None, "english"),
        ([], "english"),
        # English aliases (and case / whitespace / region-tag normalization)
        (["en"], "english"),
        (["eng"], "english"),
        (["english"], "english"),
        (["EN"], "english"),
        (["English"], "english"),
        (["  en  "], "english"),
        (["en-US"], "english"),
        (["en_US"], "english"),
        (["en", "english", "eng"], "english"),
        # Any non-english language maps to multilingual
        (["de"], "multilingual"),
        (["fr"], "multilingual"),
        (["zh-CN"], "multilingual"),
        # A single non-english language is enough to promote the whole request
        (["en", "de"], "multilingual"),
        (["de", "en"], "multilingual"),
    ],
)
def test_nemotron_language_resolution(req_languages, expected):
    assert resolve_nemotronocr_language(req_languages) == expected


def test_e2e_nemotron_ocr_conversions():
    directory = Path("./tests/data/ocr/sources")

    # List all PDF files in the directory and its subdirectories
    pdf_paths = sorted(directory.rglob("ocr_test*.pdf"))

    configs: list[OcrOptions] = [
        NemotronOcrOptions(),  # Default options
        NemotronOcrOptions(batch_size=3),
        NemotronOcrOptions(mode=OcrMode.FULL_PAGE),
    ]

    for ocr_options in configs:
        print(
            f"Converting with ocr_engine: {ocr_options.kind}, "
            f"merge_level: {ocr_options.merge_level}, "
            f"mode: {ocr_options.mode}"
        )
        converter = get_converter(ocr_options=ocr_options)
        for pdf_path in pdf_paths:
            print(f"converting {pdf_path}")

            doc_result: ConversionResult = converter.convert(pdf_path)

            verify_conversion_result_v2(
                gt=get_nemotron_ocr_groundtruth_paths(pdf_path, mode=ocr_options.mode),
                doc_result=doc_result,
                generate=GENERATE_V2,
                fuzzy=True,
            )


def test_e2e_nemotron_ocr_multipage_batching():
    """Exercise cross-page batching and the per-page redistribution of results."""
    pdf_path = Path("./tests/data/ocr/sources/nemotron_multipage.pdf")

    # Reference GT is generated with batch_size=1
    # During test the batch_size is chosen not to divide the number of pages, to ensure batches
    # that span across pages
    batch_size = 1 if GENERATE_V2 else 3

    configs: list[OcrOptions] = [
        NemotronOcrOptions(batch_size=batch_size),
        NemotronOcrOptions(batch_size=batch_size, mode=OcrMode.FULL_PAGE),
    ]

    for ocr_options in configs:
        print(
            f"Converting multi-page with batch_size: {ocr_options.batch_size}, "
            f"mode: {ocr_options.mode}"
        )
        converter = get_converter(ocr_options=ocr_options, ocr_batch_size=batch_size)
        doc_result: ConversionResult = converter.convert(pdf_path)

        verify_conversion_result_v2(
            gt=get_nemotron_ocr_groundtruth_paths(pdf_path, mode=ocr_options.mode),
            doc_result=doc_result,
            generate=GENERATE_V2,
            fuzzy=True,
        )
