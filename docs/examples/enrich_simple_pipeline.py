import logging
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import ConvertPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    HTMLFormatOption,
    WordFormatOption,
)

_log = logging.getLogger(__name__)


def main():
    input_path = Path("tests/data/docx/word_sample.docx")

    pipeline_options = ConvertPipelineOptions()
    pipeline_options.do_picture_classification = True
    pipeline_options.do_picture_description = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.DOCX: WordFormatOption(pipeline_options=pipeline_options),
            InputFormat.HTML: HTMLFormatOption(pipeline_options=pipeline_options),
        },
    )

    res = doc_converter.convert(input_path)

    print(res.document.export_to_markdown())


if __name__ == "__main__":
    main()
