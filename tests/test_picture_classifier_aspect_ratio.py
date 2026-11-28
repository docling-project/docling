# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path
from unittest.mock import Mock

import pypdfium2 as pdfium
import pytest
from PIL import Image

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import NativePdfPipelineOptions
from docling.document_converter import DocumentConverter, NativePdfFormatOption
from docling.models.inference_engines.image_classification.base import (
    BaseImageClassificationEngine,
    ImageClassificationEngineOutput,
)
from docling.models.stages.picture_classifier import document_picture_classifier


@pytest.mark.parametrize("embedded", [False, True], ids=["page-crops", "embedded"])
def test_picture_classifier_limits_aspect_ratio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedded: bool
) -> None:
    sizes = [(242, 80), (80, 242), (160, 80), (80, 80)]
    source = tmp_path / "pictures.pdf"
    with pdfium.PdfDocument.new() as pdf:
        for width, height in sizes:
            page = pdf.new_page(200, 200)
            bitmap = pdfium.PdfImage.new(pdf)
            bitmap.set_bitmap(
                pdfium.PdfBitmap.from_pil(Image.new("RGB", (width, height), "black"))
            )
            bitmap.set_matrix(
                pdfium.PdfMatrix().scale(width / 2, height / 2).translate(20, 20)
            )
            page.insert_obj(bitmap)
            page.gen_content()
        pdf.save(source)

    engine = Mock(spec=BaseImageClassificationEngine)
    engine.get_label_mapping.return_value = {0: "signature"}
    engine.predict_batch.side_effect = lambda batch: [
        ImageClassificationEngineOutput(label_ids=[0], scores=[1.0]) for _ in batch
    ]
    monkeypatch.setattr(
        document_picture_classifier,
        "create_image_classification_engine",
        lambda **_: engine,
    )
    options = NativePdfPipelineOptions(
        do_picture_classification=True,
        generate_picture_images=embedded,
        generate_page_images=not embedded,
        images_scale=2.0,
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: NativePdfFormatOption(pipeline_options=options)
        }
    )

    document = converter.convert(source).document

    images = [
        item.image
        for call in engine.predict_batch.call_args_list
        for item in call.args[0]
    ]
    assert [image.size for image in images] == [(242, 121), (121, 242), *sizes[2:]]
    assert images[0].getpixel((0, 19)) == (255, 255, 255)
    assert images[0].getpixel((0, 20)) == (0, 0, 0)
    assert images[0].getpixel((0, 99)) == (0, 0, 0)
    assert images[0].getpixel((0, 100)) == (255, 255, 255)
    assert images[1].getpixel((19, 0)) == (255, 255, 255)
    assert images[1].getpixel((20, 0)) == (0, 0, 0)
    assert images[1].getpixel((99, 0)) == (0, 0, 0)
    assert images[1].getpixel((100, 0)) == (255, 255, 255)
    for picture, size in zip(document.pictures, sizes):
        assert (picture.image is not None) == embedded
        if picture.image is not None:
            assert picture.image.pil_image.size == size
        assert picture.meta is not None
        assert picture.meta.classification is not None
        assert picture.meta.classification.predictions[0].class_name == "signature"
