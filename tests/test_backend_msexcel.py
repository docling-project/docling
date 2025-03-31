import os
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA


def get_xlsx_paths():

    # Define the directory you want to search
    directory = Path("./tests/data/xlsx/")

    # List all PDF files in the directory and its subdirectories
    pdf_files = sorted(directory.rglob("*.xlsx"))
    return pdf_files


def get_converter():

    converter = DocumentConverter(allowed_formats=[InputFormat.XLSX])

    return converter


def test_e2e_xlsx_conversions():

    xlsx_paths = get_xlsx_paths()
    converter = get_converter()

    for xlsx_path in xlsx_paths:
        print(f"converting {xlsx_path}")

        gt_path = (
            xlsx_path.parent.parent / "groundtruth" / "docling_v2" / xlsx_path.name
        )

        conv_result: ConversionResult = converter.convert(xlsx_path)

        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown()
        assert verify_export(pred_md, str(gt_path) + ".md"), "export to md"

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(
            pred_itxt, str(gt_path) + ".itxt"
        ), "export to indented-text"

        assert verify_document(
            doc, str(gt_path) + ".json", GENERATE
        ), "document document"
