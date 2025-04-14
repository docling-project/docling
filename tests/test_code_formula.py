from pathlib import Path

from docling_core.types.doc import CodeItem, TextItem
from docling_core.types.doc.labels import CodeLanguageLabel, DocItemLabel

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


def get_converter():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True

    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    pipeline_options.do_code_enrichment = True
    pipeline_options.do_formula_enrichment = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    return converter


def test_code_and_formula_conversion():
    pdf_path = Path("tests/data/pdf/code_and_formula.pdf")
    converter = get_converter()

    print(f"converting {pdf_path}")

    doc_result: ConversionResult = converter.convert(pdf_path)

    results = doc_result.document.texts

    code_blocks = [el for el in results if isinstance(el, CodeItem)]
    assert len(code_blocks) == 1

    gt = "function add(a, b) {\n    return a + b;\n}\nconsole.log(add(3, 5));"

    predicted = code_blocks[0].text.strip()
    assert predicted == gt, f"mismatch in text {predicted=}, {gt=}"
    assert code_blocks[0].code_language == CodeLanguageLabel.JAVASCRIPT

    formula_blocks = [
        el
        for el in results
        if isinstance(el, TextItem) and el.label == DocItemLabel.FORMULA
    ]
    assert len(formula_blocks) == 1

    gt = "a ^ { 2 } + 8 = 1 2"
    predicted = formula_blocks[0].text
    assert predicted == gt, f"mismatch in text {predicted=}, {gt=}"
