import logging
import os
from pathlib import Path

import pytest

from docling.backend.docx.drawingml.utils import get_libreoffice_cmd
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import (
    ConversionResult,
    DoclingDocument,
    SectionHeaderItem,
    TextItem,
)
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

_log = logging.getLogger(__name__)

GENERATE = GEN_TEST_DATA
IS_CI = bool(os.getenv("CI"))


def get_docx_paths():
    # Define the directory you want to search
    directory = Path("./tests/data/docx/")

    # List all docx files in the directory and its subdirectories
    docx_files = sorted(directory.rglob("*.docx"))
    return docx_files


def get_converter():
    converter = DocumentConverter(allowed_formats=[InputFormat.DOCX])

    return converter


@pytest.fixture(scope="module")
def documents() -> list[tuple[Path, DoclingDocument]]:
    documents: list[dict[Path, DoclingDocument]] = []

    docx_paths = get_docx_paths()
    converter = get_converter()

    for docx_path in docx_paths:
        _log.debug(f"converting {docx_path}")

        gt_path = (
            docx_path.parent.parent / "groundtruth" / "docling_v2" / docx_path.name
        )

        conv_result: ConversionResult = converter.convert(docx_path)

        doc: DoclingDocument = conv_result.document

        assert doc, f"Failed to convert document from file {gt_path}"
        documents.append((gt_path, doc))

    return documents


def _test_e2e_docx_conversions_impl(docx_paths: list[tuple[Path, DoclingDocument]]):
    has_libreoffice = False
    try:
        cmd = get_libreoffice_cmd(raise_if_unavailable=True)
        if cmd is not None:
            has_libreoffice = True
    except Exception:
        pass

    for docx_path, doc in docx_paths:
        if not IS_CI and not has_libreoffice and docx_path.name == "drawingml.docx":
            print(f"Skipping {docx_path} because no Libreoffice is installed.")
            continue

        pred_md: str = doc.export_to_markdown()
        assert verify_export(pred_md, str(docx_path) + ".md", generate=GENERATE), (
            f"export to markdown failed on {docx_path}"
        )

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(docx_path) + ".itxt", generate=GENERATE), (
            f"export to indented-text failed on {docx_path}"
        )

        assert verify_document(doc, str(docx_path) + ".json", generate=GENERATE), (
            f"DoclingDocument verification failed on {docx_path}"
        )

        if docx_path.name == "word_tables.docx":
            pred_html: str = doc.export_to_html()
            assert verify_export(
                pred_text=pred_html,
                gtfile=str(docx_path) + ".html",
                generate=GENERATE,
            ), f"export to html failed on {docx_path}"


flaky_file = "textbox.docx"


def test_e2e_docx_conversions(documents):
    target = [item for item in documents if item[0].name != flaky_file]
    _test_e2e_docx_conversions_impl(target)


@pytest.mark.xfail(strict=False)
def test_textbox_conversion(documents):
    target = [item for item in documents if item[0].name == flaky_file]
    _test_e2e_docx_conversions_impl(target)


@pytest.mark.xfail(strict=False)
def test_textbox_extraction(documents):
    name = "textbox.docx"
    doc = next(item[1] for item in documents if item[0].name == name)

    # Verify if a particular textbox content is extracted
    textbox_found = False
    for item, _ in doc.iterate_items():
        if item.text[:30] == """Suggested Reportable Symptoms:""":
            textbox_found = True
    assert textbox_found


def test_heading_levels(documents):
    name = "word_sample.docx"
    doc = next(item[1] for item in documents if item[0].name == name)

    found_lvl_1 = found_lvl_2 = False
    for item, _ in doc.iterate_items():
        if isinstance(item, SectionHeaderItem):
            if item.text == "Let\u2019s swim!":
                found_lvl_1 = True
                assert item.level == 1
            elif item.text == "Let\u2019s eat":
                found_lvl_2 = True
                assert item.level == 2
    assert found_lvl_1 and found_lvl_2


def test_text_after_image_anchors(documents):
    """Test to analyse whether text gets parsed after image anchors."""

    name = "word_image_anchors.docx"
    doc = next(item[1] for item in documents if item[0].name == name)

    found_text_after_anchor_1 = found_text_after_anchor_2 = (
        found_text_after_anchor_3
    ) = found_text_after_anchor_4 = False
    for item, _ in doc.iterate_items():
        if isinstance(item, TextItem):
            if item.text == "This is test 1":
                found_text_after_anchor_1 = True
            elif item.text == "0:08\nCorrect, he is not.":
                found_text_after_anchor_2 = True
            elif item.text == "This is test 2":
                found_text_after_anchor_3 = True
            elif item.text == "0:16\nYeah, exactly.":
                found_text_after_anchor_4 = True

    assert (
        found_text_after_anchor_1
        and found_text_after_anchor_2
        and found_text_after_anchor_3
        and found_text_after_anchor_4
    )
