from pathlib import Path

from docling_core.types.doc import PictureClassificationMetaField

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
    pipeline_options.do_code_enrichment = False
    pipeline_options.do_formula_enrichment = False
    pipeline_options.generate_picture_images = False
    pipeline_options.generate_page_images = False
    pipeline_options.do_picture_classification = True
    pipeline_options.images_scale = 2

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    return converter


def test_picture_classifier():
    pdf_path = Path("tests/data/pdf/picture_classification.pdf")
    converter = get_converter()

    print(f"converting {pdf_path}")

    doc_result: ConversionResult = converter.convert(pdf_path)

    results = doc_result.document.pictures

    assert len(results) == 2

    res = results[0]
    assert res.meta is not None
    assert isinstance(res.meta.classification, PictureClassificationMetaField)
    classification_data = res.meta.classification
    assert len(classification_data.predictions) == 16, (
        "Number of predicted classes is not equal to 16"
    )
    confidences = [pred.confidence for pred in classification_data.predictions]
    assert confidences == sorted(confidences, reverse=True), (
        "Predictions are not sorted in descending order of confidence"
    )
    assert classification_data.predictions[0].class_name == "bar_chart", (
        "The prediction is wrong for the bar chart image."
    )
    assert classification_data.predictions[0].created_by == "DocumentPictureClassifier"

    res = results[1]
    assert res.meta is not None
    assert isinstance(res.meta.classification, PictureClassificationMetaField)
    classification_data = res.meta.classification
    assert len(classification_data.predictions) == 16, (
        "Number of predicted classes is not equal to 16"
    )
    confidences = [pred.confidence for pred in classification_data.predictions]
    assert confidences == sorted(confidences, reverse=True), (
        "Predictions are not sorted in descending order of confidence"
    )
    assert classification_data.predictions[0].class_name == "map", (
        "The prediction is wrong for the bar chart image."
    )
    assert classification_data.predictions[0].created_by == "DocumentPictureClassifier"
