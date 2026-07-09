from pathlib import Path

import pytest

from docling.datamodel.pipeline_options import OcrMode
from tests.groundtruth_paths import (
    get_ocr_groundtruth_paths,
    get_regular_groundtruth_paths,
)


# def test_default_paths_cover_dir_suffixes_and_dotted_stem():
#     # Dotted stem (arXiv id) exercises "only the extension is replaced".
#     # Default GT dir is the `groundtruth` sibling of the source's parent.
#     gt = get_regular_groundtruth_paths(Path("tests/data/pdf/sources/2206.01062.pdf"))
#
#     gt_dir = Path("tests/data/pdf/groundtruth")
#     assert gt.pages_meta == gt_dir / "2206.01062.pages.meta.json"
#     assert gt.doc_json == gt_dir / "2206.01062.json"
#     assert gt.md == gt_dir / "2206.01062.md"
#     assert gt.doctags == gt_dir / "2206.01062.doctags.txt"
#
#
# def test_gt_dir_override_and_tag_before_format_suffix():
#     override = Path("some/other/groundtruth")
#     gt = get_regular_groundtruth_paths(
#         Path("tests/data/pdf/sources/report.pdf"),
#         gt_dir=override,
#         tag="nemotron-ocr.full-page",
#     )
#
#     assert gt.doc_json == override / "report.nemotron-ocr.full-page.json"
#     assert gt.doctags == override / "report.nemotron-ocr.full-page.doctags.txt"
#
#
# @pytest.mark.parametrize(
#     ("engine", "expected_dir"),
#     [
#         # Both tesseract variants collapse onto one shared "tesseract" reference.
#         ("tesserocr", "tesseract"),
#         ("tesseract", "tesseract"),
#         ("easyocr", "easyocr"),
#         ("rapidocr", "rapidocr"),
#         ("ocrmac", "ocrmac"),
#         # kserve reuses the rapidocr reference (the endpoint serves a rapidocr model).
#         ("kserve_v2_ocr", "rapidocr"),
#     ],
# )
# def test_ocr_paths_use_engine_dir_and_tag(engine, expected_dir):
#     input_path = Path("tests/data/ocr/sources/ocr_test.pdf")
#
#     # get_ocr_groundtruth_paths nests GT under `groundtruth/<engine>/<mode>/` and tags
#     # each file with `<engine>.<mode>`, where <engine> is the mapped on-disk name (so the
#     # mapped name drives both the sub-dir and the filename tag).
#     gt_dir = input_path.parent.parent / "groundtruth" / expected_dir / "full_page_ocr"
#
#     ocr_gt = get_ocr_groundtruth_paths(
#         input_path, mode=OcrMode.FULL_PAGE_OCR, engine=engine
#     )
#     assert ocr_gt == get_regular_groundtruth_paths(
#         input_path, gt_dir=gt_dir, tag=f"{expected_dir}.full_page_ocr"
#     )
