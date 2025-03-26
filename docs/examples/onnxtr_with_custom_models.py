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

    # Available detection & recognition models can be found at
    # https://github.com/felixdittrich92/OnnxTR

    # Or you choose a model from Hugging Face Hub
    # Collection: https://huggingface.co/collections/Felix92/onnxtr-66bf213a9f88f7346c90e842

    ocr_options = OnnxtrOcrOptions(
        # Text detection model
        det_arch="db_mobilenet_v3_large",
        # Text recognition model - from Hugging Face Hub
        reco_arch="Felix92/onnxtr-parseq-multilingual-v1",
        # This can be set to `True` to auto-correct the orientation of the pages
        auto_correct_orientation=False,
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
