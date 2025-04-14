import logging
from pathlib import Path

import pytest

from docling.backend.msexcel_backend import MsExcelDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument, InputDocument
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

_log = logging.getLogger(__name__)

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


@pytest.fixture(scope="module")
def documents() -> list[tuple[Path, DoclingDocument]]:
    documents: list[dict[Path, DoclingDocument]] = []

    xlsx_paths = get_xlsx_paths()
    converter = get_converter()

    for xlsx_path in xlsx_paths:
        _log.debug(f"converting {xlsx_path}")

        gt_path = (
            xlsx_path.parent.parent / "groundtruth" / "docling_v2" / xlsx_path.name
        )

        conv_result: ConversionResult = converter.convert(xlsx_path)

        doc: DoclingDocument = conv_result.document

        assert doc, f"Failed to convert document from file {gt_path}"
        documents.append((gt_path, doc))

    return documents


def test_e2e_xlsx_conversions(documents) -> None:
    for gt_path, doc in documents:
        pred_md: str = doc.export_to_markdown()
        assert verify_export(pred_md, str(gt_path) + ".md"), "export to md"

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt"), (
            "export to indented-text"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE), (
            "document document"
        )


def test_pages(documents) -> None:
    """Test the page count and page size of converted documents.

    Args:
        documents: The paths and converted documents.
    """
    # number of pages from the backend method
    path = next(item for item in get_xlsx_paths() if item.stem == "test-01")
    in_doc = InputDocument(
        path_or_stream=path,
        format=InputFormat.XLSX,
        filename=path.stem,
        backend=MsExcelDocumentBackend,
    )
    backend = MsExcelDocumentBackend(in_doc=in_doc, path_or_stream=path)
    assert backend.page_count() == 3

    # number of pages from the converted document
    doc = next(item for path, item in documents if path.stem == "test-01")
    assert len(doc.pages) == 3

    # page sizes as number of cells
    assert doc.pages.get(1).size.as_tuple() == (3.0, 7.0)
    assert doc.pages.get(2).size.as_tuple() == (9.0, 18.0)
    assert doc.pages.get(3).size.as_tuple() == (13.0, 36.0)
