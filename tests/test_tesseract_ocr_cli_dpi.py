# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from PIL import Image, ImageDraw

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    OcrMode,
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.stages.ocr.tesseract_ocr_cli_model import TesseractOcrCliModel

pytestmark = [
    pytest.mark.ml_ocr,
    pytest.mark.skipif(
        shutil.which("tesseract") is None, reason="tesseract not installed"
    ),
]


@pytest.mark.parametrize("scale", [3.0, 4.0])
@pytest.mark.parametrize("orientation", [0, 90])
@pytest.mark.parametrize(
    ("ocr_overrides", "expected_embed_dpi"),
    [
        pytest.param({}, True, id="default-dpi"),
        pytest.param({"embed_render_dpi": False}, False, id="estimate-dpi"),
    ],
)
def test_rendered_images_use_configured_dpi(
    tmp_path: Path,
    scale: float,
    orientation: int,
    ocr_overrides: dict[str, bool],
    expected_embed_dpi: bool,
) -> None:
    image = Image.new("RGB", (600, 300), "white")
    ImageDraw.Draw(image).multiline_text(
        (30, 30),
        "First line of text\nSecond line of text\nThird line of text",
        fill="black",
    )
    source = tmp_path / "scan.pdf"
    image.save(source)

    options = PdfPipelineOptions(do_ocr=True, do_table_structure=False)
    options.ocr_options = TesseractCliOcrOptions.model_validate(
        {"lang": ["eng"], "mode": OcrMode.FULL_PAGE, "scale": scale, **ocr_overrides}
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )

    run_tesseract = TesseractOcrCliModel._run_tesseract
    seen_dpi: list[tuple[float, float] | None] = []

    def perform_osd(model: TesseractOcrCliModel, filename: str) -> pd.DataFrame:
        with Image.open(filename) as rendered:
            seen_dpi.append(rendered.info.get("dpi"))
        return pd.DataFrame(
            {"key": ["Orientation in degrees"], "value": [str(orientation)]}
        )

    def run_ocr(
        model: TesseractOcrCliModel, filename: str, osd: pd.DataFrame | None
    ) -> pd.DataFrame:
        with Image.open(filename) as rendered:
            seen_dpi.append(rendered.info.get("dpi"))
        return run_tesseract(model, filename, osd)

    # Select the rotation path without depending on OSD accuracy.
    with (
        patch.object(
            TesseractOcrCliModel, "_perform_osd", autospec=True, side_effect=perform_osd
        ),
        patch.object(
            TesseractOcrCliModel, "_run_tesseract", autospec=True, side_effect=run_ocr
        ),
    ):
        result = converter.convert(source)

    assert result.status == ConversionStatus.SUCCESS
    expected_dpi = (
        pytest.approx((72 * scale, 72 * scale), abs=0.02)
        if expected_embed_dpi
        else None
    )
    assert seen_dpi == [expected_dpi, expected_dpi]
