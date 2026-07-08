from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from docling.datamodel.pipeline_options import OcrMode

# The four ground-truth artifacts produced per converted document
_PAGES_META_SUFFIX = ".pages.meta.json"
_JSON_SUFFIX = ".json"
_MD_SUFFIX = ".md"
_DOCTAGS_SUFFIX = ".doctags.txt"

_OCR_MODE_TO_DIR_NAME = {
    OcrMode.FULL_PAGE_OCR: OcrMode.FULL_PAGE_OCR.value,
    OcrMode.CLUSTER_OCR: OcrMode.CLUSTER_OCR.value,
    OcrMode.PDF_CLUSTER_OCR: OcrMode.PDF_CLUSTER_OCR.value,
}

_OCR_MODE_TO_TAG = {
    OcrMode.FULL_PAGE_OCR: OcrMode.FULL_PAGE_OCR.value,
    OcrMode.CLUSTER_OCR: OcrMode.CLUSTER_OCR.value,
    OcrMode.PDF_CLUSTER_OCR: OcrMode.PDF_CLUSTER_OCR.value,
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
    """Build the GT paths for ``input_path``.

    Returns:
        The four GT file locations as a :class:`GroundTruthPaths`.
    """
    base_dir = (
        gt_dir if gt_dir is not None else input_path.parent.parent / "groundtruth"
    )
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
    engine: Optional[str] = None,
    mode: Optional[str] = None,
    gt_dir: Optional[Path] = None,
) -> GroundTruthPaths:
    """Build GT paths for an OCR conversion, tagged by engine and (optional) mode.

    Returns:
        The four GT file locations as a :class:`GroundTruthPaths`.
    """
    if engine is None and mode is None:
        tag: Optional[str] = None
    elif mode is None:
        tag = engine
    elif engine is None:
        tag = mode
    else:
        tag = f"{engine}.{mode}"

    gt_paths = get_regular_groundtruth_paths(input_path, gt_dir=gt_dir, tag=tag)
    return gt_paths


def get_nemotron_ocr_groundtruth_paths(
    input_path: Path,
    *,
    mode: OcrMode,
) -> GroundTruthPaths:
    """Build GT paths for nemotron OCR"""
    model_name = "nemotron_ocr"

    mode_dir_name = _OCR_MODE_TO_DIR_NAME.get(mode, ".")
    mode_tag = _OCR_MODE_TO_TAG.get(mode, "")
    tag = f"{model_name}.{mode_tag}"

    nemotron_gt_dir = (
        input_path.parent.parent / "groundtruth" / model_name / mode_dir_name
    )

    gt_paths = get_regular_groundtruth_paths(
        input_path, gt_dir=nemotron_gt_dir, tag=tag
    )
    return gt_paths
