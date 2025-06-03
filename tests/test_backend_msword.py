from pathlib import Path

import pytest

from docling.backend.msword_backend import MsWordDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import (
    ConversionResult,
    DoclingDocument,
    InputDocument,
    SectionHeaderItem,
)
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA


@pytest.mark.xfail(strict=False)
def test_textbox_extraction():
    in_path = Path("tests/data/docx/textbox.docx")
    in_doc = InputDocument(
        path_or_stream=in_path,
        format=InputFormat.DOCX,
        backend=MsWordDocumentBackend,
    )
    backend = MsWordDocumentBackend(
        in_doc=in_doc,
        path_or_stream=in_path,
    )
    doc = backend.convert()

    # Verify if a particular textbox content is extracted
    textbox_found = False
    for item, _ in doc.iterate_items():
        if item.text[:30] == """Suggested Reportable Symptoms:""":
            textbox_found = True
    assert textbox_found


def test_heading_levels():
    in_path = Path("tests/data/docx/word_sample.docx")
    in_doc = InputDocument(
        path_or_stream=in_path,
        format=InputFormat.DOCX,
        backend=MsWordDocumentBackend,
    )
    backend = MsWordDocumentBackend(
        in_doc=in_doc,
        path_or_stream=in_path,
    )
    doc = backend.convert()

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


def get_docx_paths():
    # Define the directory you want to search
    directory = Path("./tests/data/docx/")

    # List all PDF files in the directory and its subdirectories
    pdf_files = sorted(directory.rglob("*.docx"))
    return pdf_files


def get_converter():
    converter = DocumentConverter(allowed_formats=[InputFormat.DOCX])

    return converter


def _test_e2e_docx_conversions_impl(docx_paths: list[Path]):
    converter = get_converter()

    for docx_path in docx_paths:
        # print(f"converting {docx_path}")

        gt_path = (
            docx_path.parent.parent / "groundtruth" / "docling_v2" / docx_path.name
        )

        conv_result: ConversionResult = converter.convert(docx_path)

        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown()
        assert verify_export(pred_md, str(gt_path) + ".md", generate=GENERATE), (
            "export to md"
        )

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", generate=GENERATE), (
            "export to indented-text"
        )

        assert verify_document(doc, str(gt_path) + ".json", generate=GENERATE), (
            "document document"
        )

        if docx_path.name == "word_tables.docx":
            pred_html: str = doc.export_to_html()
            assert verify_export(
                pred_text=pred_html,
                gtfile=str(gt_path) + ".html",
                generate=GENERATE,
            ), "export to html"


flaky_path = Path("tests/data/docx/textbox.docx")


def test_e2e_docx_conversions():
    _test_e2e_docx_conversions_impl(
        docx_paths=[path for path in get_docx_paths() if path != flaky_path]
    )


@pytest.mark.xfail(strict=False)
def test_textbox_conversion():
    _test_e2e_docx_conversions_impl(docx_paths=[flaky_path])
