# Example: Integrating SuryaOCR with Docling for PDF OCR and Markdown Export
#
# Overview:
# - Configures SuryaOCR options for OCR.
# - Executes PDF pipeline with SuryaOCR integration.
# - Models auto-download from Hugging Face on first run.
#
# Prerequisites:
# - Install: `pip install docling-surya`
# - Ensure `docling` imports successfully.
#
# Execution:
# - Run from repo root: `python docs/examples/suryaocr_with_custom_models.py`
# - Outputs Markdown to stdout.
#
# Notes:
# - Default source: EPA PDF URL; substitute with local path as needed.
# - Models cached in `~/.cache/huggingface`; override with HF_HOME env var.
# - Use proxy config for restricted networks.
# - **Important Licensing Note**: The `docling-surya` package integrates SuryaOCR, which is licensed under the GNU General Public License (GPL).
#   Using this integration may impose GPL obligations on your project. Review the license terms carefully.

# Requires `pip install docling-surya`
# See https://pypi.org/project/docling-surya/
from docling_surya import SuryaOcrOptions

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def main():
    source = "https://19january2021snapshot.epa.gov/sites/static/files/2016-02/documents/epa_sample_letter_sent_to_commissioners_dated_february_29_2015.pdf"

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        ocr_model="suryaocr",
        allow_external_plugins=True,
        ocr_options=SuryaOcrOptions(lang=["en"]),
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )

    result = converter.convert(source)
    print(result.document.export_to_markdown())


if __name__ == "__main__":
    main()
