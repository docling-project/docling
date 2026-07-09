from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    KserveV2OcrOptions,
    OcrMacOptions,
    OcrMode,
    RapidOcrOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)

from .test_data_gen_flag import GEN_TEST_DATA

# The four ground-truth artifacts produced per converted document
_PAGES_META_SUFFIX = ".pages.meta.json"
_JSON_SUFFIX = ".json"
_MD_SUFFIX = ".md"
_DOCTAGS_SUFFIX = ".doctags.txt"


# Maps an OCR engine `kind` to the GT sub-directory (and filename tag) it uses.
_OCR_ENGINE_TO_DIR: dict[str, str] = {
    TesseractOcrOptions.kind: "tesseract",
    TesseractCliOcrOptions.kind: "tesseract",
    EasyOcrOptions.kind: "easyocr",
    RapidOcrOptions.kind: "rapidocr",
    OcrMacOptions.kind: "ocrmac",
    KserveV2OcrOptions.kind: "rapidocr",
}


class GroundTruthPaths(BaseModel):
    """Locations of the ground-truth files for a single converted document"""

    model_config = ConfigDict(frozen=True)

    pages_meta: Path
    doc_json: Path
    md: Path
    doctags: Path


def get_regular_groundtruth_paths(
    input_path: Path,
    *,
    gt_dir: Optional[Path] = None,
    tag: Optional[str] = None,
) -> GroundTruthPaths:
    """
    Build the GT paths.
    An exception is raised if the resolved test dir does not exist, unless GEN_TEST_DATA is set
    """
    base_dir = (
        gt_dir if gt_dir is not None else input_path.parent.parent / "groundtruth"
    )
    if not GEN_TEST_DATA and not base_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory does not exist: {base_dir}")
    base = base_dir / input_path.name
    prefix = "" if tag is None else f".{tag}"

    gt_paths = GroundTruthPaths(
        pages_meta=base.with_suffix(f"{prefix}{_PAGES_META_SUFFIX}"),
        doc_json=base.with_suffix(f"{prefix}{_JSON_SUFFIX}"),
        md=base.with_suffix(f"{prefix}{_MD_SUFFIX}"),
        doctags=base.with_suffix(f"{prefix}{_DOCTAGS_SUFFIX}"),
    )
    return gt_paths


def get_ocr_groundtruth_paths(
    input_path: Path,
    *,
    mode: OcrMode,
    engine: str,
) -> GroundTruthPaths:
    """Build GT paths for an OCR conversion, organized by engine and OCR mode."""
    engine_name = _OCR_ENGINE_TO_DIR[engine]
    mode_dir_name = mode.value
    mode_tag = mode.value
    tag = f"{engine_name}.{mode_tag}"

    engine_gt_dir = (
        input_path.parent.parent / "groundtruth" / engine_name / mode_dir_name
    )

    gt_paths = get_regular_groundtruth_paths(input_path, gt_dir=engine_gt_dir, tag=tag)
    return gt_paths
