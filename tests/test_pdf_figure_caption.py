# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


@pytest.mark.ml_pdf_model
def test_long_figure_caption_is_labeled_and_linked() -> None:
    # A synthetic chart with a 2,190-character caption that starts with a
    # production code. Run the default layout model on its rendered page.
    source = Path(__file__).parent / "data/pdf/fixtures/long_figure_caption.pdf"
    options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=False,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )

    result = converter.convert(source)
    doc = result.document

    (picture,) = doc.pictures
    (caption,) = doc.texts
    assert (caption.label, picture.captions) == (
        DocItemLabel.CAPTION,
        [caption.get_ref()],
    )
    expected_text = (
        "AX19 Figure 15-35 The monitoring stations. "
        + (
            "Each group shows measurements from a different sensor. "
            "The colored bars show the readings from the three trials. "
            "The first group gives the baseline readings for all three trials. "
        )
        * 12
    )
    assert caption.text == expected_text.rstrip()
    assert caption.parent == picture.get_ref()
    assert caption.text in doc.export_to_markdown()
