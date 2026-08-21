from pathlib import Path

import pytest
from docling_core.types.doc import GroupLabel
from docling_core.types.doc.document import PictureItem, TableItem

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

pytestmark = pytest.mark.ml_pdf_model

PDF_PATH = Path("tests/data/pdf/sources/table_misidentified_as_form.pdf")


def _get_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.accelerator_options.device = AcceleratorDevice.CPU
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )


def test_tables_and_pictures_are_nested_in_form_area() -> None:
    doc = _get_converter().convert(PDF_PATH).document

    form = next(group for group in doc.groups if group.label == GroupLabel.FORM_AREA)
    children = [child.resolve(doc) for child in form.children]
    nested_tables = [child for child in children if isinstance(child, TableItem)]
    nested_pictures = [child for child in children if isinstance(child, PictureItem)]

    assert {table.self_ref for table in nested_tables} == {
        table.self_ref for table in doc.tables
    }
    assert len(nested_tables) == 3
    assert len(nested_pictures) == 3
    assert all(table.parent == form.get_ref() for table in nested_tables)
    assert all(picture.parent == form.get_ref() for picture in nested_pictures)
