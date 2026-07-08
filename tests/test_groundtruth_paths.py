from pathlib import Path

import pytest

from docling.datamodel.pipeline_options import OcrMode
from tests.groundtruth_paths import (
    get_ocr_groundtruth_paths,
    get_regular_groundtruth_paths,
)


def test_default_paths_cover_dir_suffixes_and_dotted_stem():
    # Dotted stem (arXiv id) exercises "only the extension is replaced".
    # Default GT dir is the `groundtruth` sibling of the source's parent.
    gt = get_regular_groundtruth_paths(Path("tests/data/pdf/sources/2206.01062.pdf"))

    gt_dir = Path("tests/data/pdf/groundtruth")
    assert gt.pages_meta == gt_dir / "2206.01062.pages.meta.json"
    assert gt.doc_json == gt_dir / "2206.01062.json"
    assert gt.md == gt_dir / "2206.01062.md"
    assert gt.doctags == gt_dir / "2206.01062.doctags.txt"


def test_gt_dir_override_and_tag_before_format_suffix():
    override = Path("some/other/groundtruth")
    gt = get_regular_groundtruth_paths(
        Path("tests/data/pdf/sources/report.pdf"),
        gt_dir=override,
        tag="nemotron-ocr.full-page",
    )

    assert gt.doc_json == override / "report.nemotron-ocr.full-page.json"
    assert gt.doctags == override / "report.nemotron-ocr.full-page.doctags.txt"


@pytest.mark.parametrize(
    ("engine", "expected_tag"),
    [
        (None, "full_page_ocr"),
        ("kserve_v2_ocr", "kserve_v2_ocr.full_page_ocr"),
    ],
)
def test_ocr_paths_use_mode_subdir_and_tag(engine, expected_tag):
    input_path = Path("tests/data/scanned/sources/ocr_test.pdf")

    # get_ocr_groundtruth_paths nests GT under `groundtruth/general/<mode>/`
    # and tags each file with the mode (optionally prefixed with the engine).
    gt_dir = input_path.parent.parent / "groundtruth" / "general" / "full_page_ocr"

    ocr_gt = get_ocr_groundtruth_paths(
        input_path, mode=OcrMode.FULL_PAGE_OCR, engine=engine
    )
    assert ocr_gt == get_regular_groundtruth_paths(
        input_path, gt_dir=gt_dir, tag=expected_tag
    )
