import json
from pathlib import Path

import pytest

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


@pytest.fixture
def test_doc_paths():
    return [
        Path("tests/data/html/wiki_duck.html"),
        Path("tests/data/docx/word_sample.docx"),
        Path("tests/data/docx/lorem_ipsum.docx"),
        Path("tests/data/pptx/powerpoint_sample.pptx"),
        Path("tests/data/2305.03393v1-pg9-img.png"),
        Path("tests/data/pdf/2206.01062.pdf"),
    ]


def get_converter():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        }
    )

    return converter


def test_compare_legacy_output(test_doc_paths):
    converter = get_converter()
    res = converter.convert_all(test_doc_paths, raises_on_error=True)
    for conv_res in res:
        print(f"Results for {conv_res.input.file}")
        with pytest.warns(DeprecationWarning, match="Use document instead"):
            print(
                json.dumps(
                    conv_res.legacy_document.model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    )
                )
            )
    # assert res.legacy_output == res.legacy_output_transformed
