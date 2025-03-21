from docling.datamodel.pipeline_options import OnnxtrOcrOptions, PdfPipelineOptions
from docling.document_converter import (
    ConversionResult,
    DocumentConverter,
    InputFormat,
    PdfFormatOption,
)


def main():
    # Source document to convert
    source = "https://arxiv.org/pdf/2408.09869v4"

    ocr_options = OnnxtrOcrOptions(
        det_arch="db_mobilenet_v3_large",
        reco_arch="Felix92/onnxtr-parseq-multilingual-v1",  # Model will be downloaded from Hugging Face Hub
        auto_correct_orientation=True,  # This can be used to correct the orientation of the pages
    )

    pipeline_options = PdfPipelineOptions(
        ocr_options=ocr_options,
    )

    # Convert the document
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        },
    )

    conversion_result: ConversionResult = converter.convert(source=source)
    doc = conversion_result.document
    md = doc.export_to_markdown()
    print(md)


if __name__ == "__main__":
    main()
