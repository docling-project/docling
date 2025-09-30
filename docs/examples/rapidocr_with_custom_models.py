# %% [markdown]
# Use RapidOCR with custom ONNX models to OCR a PDF page and print Markdown.
#
# What this example does
# - Downloads RapidOCR models from Hugging Face via ModelScope.
# - Configures `RapidOcrOptions` with explicit det/rec/cls model paths.
# - Runs the PDF pipeline with RapidOCR and prints Markdown output.
#
# Prerequisites
# - Install Docling, `modelscope`, and have network access to download models.
# - Ensure your environment can import `docling` and `modelscope`.
#
# How to run
# - From the repo root: `python docs/examples/rapidocr_with_custom_models.py`.
# - The script prints the recognized text as Markdown to stdout.
#
# Notes
# - The default `source` points to an arXiv PDF URL; replace with a local path if desired.
# - Model paths are derived from the downloaded snapshot directory.
# - ModelScope caches downloads (typically under `~/.cache/modelscope`); set a proxy
#   or pre-download models if running in a restricted network environment.

# %%

import os

from modelscope import snapshot_download

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def main():
    # Source document to convert
    source = "https://arxiv.org/pdf/2408.09869v4"

    # Download RapidOCR models from Hugging Face
    print("Downloading RapidOCR models")
    download_path = snapshot_download(repo_id="RapidAI/RapidOCR")

    # Setup RapidOcrOptions for English detection
    det_model_path = os.path.join(
        download_path, "onnx", "PP-OCRv5", "det", "ch_PP-OCRv5_server_det.onnx"
    )
    rec_model_path = os.path.join(
        download_path, "onnx", "PP-OCRv5", "rec", "ch_PP-OCRv5_rec_server_infer.onnx"
    )
    cls_model_path = os.path.join(
        download_path, "onnx", "PP-OCRv4", "cls", "ch_ppocr_mobile_v2.0_cls_infer.onnx"
    )
    ocr_options = RapidOcrOptions(
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        cls_model_path=cls_model_path,
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
