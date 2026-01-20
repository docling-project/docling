import warnings
from pathlib import Path

from docling_core.types.doc import (
    PictureClassificationData,
    PictureClassificationMetaField,
)

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

    # Test first picture (bar chart)
    res = results[0]

    # Test old format (.annotations) - expect DeprecationWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert len(res.annotations) == 1
        # Verify that a DeprecationWarning was raised
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "annotations" in str(w[0].message).lower()

    # Now silence the deprecation warnings for the rest of the test
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, message=".*annotations.*"
        )

        assert isinstance(res.annotations[0], PictureClassificationData)
        classification_data = res.annotations[0]
        assert classification_data.provenance == "DocumentPictureClassifier"
        assert len(classification_data.predicted_classes) == 26, (
            "Number of predicted classes is not equal to 26"
        )
        confidences = [
            pred.confidence for pred in classification_data.predicted_classes
        ]
        assert confidences == sorted(confidences, reverse=True), (
            "Predictions are not sorted in descending order of confidence"
        )
        assert classification_data.predicted_classes[0].class_name == "bar_chart", (
            "The prediction is wrong for the bar chart image."
        )

        # Test new format (.meta.classification)
        assert res.meta is not None, "Picture meta should not be None"
        assert res.meta.classification is not None, (
            "Classification meta should not be None"
        )
        assert isinstance(res.meta.classification, PictureClassificationMetaField)
        meta_classification = res.meta.classification
        assert len(meta_classification.predictions) == 26, (
            "Number of predictions in meta is not equal to 26"
        )
        meta_confidences = [pred.confidence for pred in meta_classification.predictions]
        assert meta_confidences == sorted(meta_confidences, reverse=True), (
            "Meta predictions are not sorted in descending order of confidence"
        )
        assert meta_classification.predictions[0].class_name == "bar_chart", (
            "The meta prediction is wrong for the bar chart image."
        )
        assert (
            meta_classification.predictions[0].created_by == "DocumentPictureClassifier"
        ), "The created_by field should be DocumentPictureClassifier"

        # Test second picture (map)
        res = results[1]

        # Test old format (.annotations)
        assert len(res.annotations) == 1
        assert isinstance(res.annotations[0], PictureClassificationData)
        classification_data = res.annotations[0]
        assert classification_data.provenance == "DocumentPictureClassifier"
        assert len(classification_data.predicted_classes) == 26, (
            "Number of predicted classes is not equal to 26"
        )
        confidences = [
            pred.confidence for pred in classification_data.predicted_classes
        ]
        assert confidences == sorted(confidences, reverse=True), (
            "Predictions are not sorted in descending order of confidence"
        )
        assert (
            classification_data.predicted_classes[0].class_name == "geographical_map"
        ), "The prediction is wrong for the map image."

        # Test new format (.meta.classification)
        assert res.meta is not None, "Picture meta should not be None"
        assert res.meta.classification is not None, (
            "Classification meta should not be None"
        )
        assert isinstance(res.meta.classification, PictureClassificationMetaField)
        meta_classification = res.meta.classification
        assert len(meta_classification.predictions) == 26, (
            "Number of predictions in meta is not equal to 26"
        )
        meta_confidences = [pred.confidence for pred in meta_classification.predictions]
        assert meta_confidences == sorted(meta_confidences, reverse=True), (
            "Meta predictions are not sorted in descending order of confidence"
        )
        assert meta_classification.predictions[0].class_name == "geographical_map", (
            "The meta prediction is wrong for the map image."
        )
        assert (
            meta_classification.predictions[0].created_by == "DocumentPictureClassifier"
        ), "The created_by field should be DocumentPictureClassifier"
