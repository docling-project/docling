# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import openpyxl
import pytest
from docx import Document
from docx.shared import Inches
from openpyxl.drawing.image import Image as XlImage
from PIL import Image, ImageCms
from pptx import Presentation

from docling.backend.msexcel_backend import MsExcelDocumentBackend
from docling.backend.utils.image import normalize_image_for_png
from docling.datamodel.base_models import ConversionStatus
from docling.document_converter import DocumentConverter


def _cmyk_jpeg(profile: bytes | None = None) -> BytesIO:
    jpeg = BytesIO()
    Image.new("CMYK", (200, 100), (0, 255, 255, 0)).save(
        jpeg, format="JPEG", icc_profile=profile
    )
    jpeg.seek(0)
    return jpeg


def _save_xlsx_with_picture(path: Path, picture: BytesIO) -> None:
    book = openpyxl.Workbook()
    sheet = book.active
    assert sheet is not None
    sheet["A1"] = "label"
    sheet.add_image(XlImage(picture), "B2")
    book.save(path)


@pytest.mark.parametrize("profile", [None, b"invalid ICC profile"])
@pytest.mark.parametrize("suffix", ["pptx", "docx", "xlsx"])
def test_cmyk_office_picture_without_libreoffice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str, profile: bytes | None
) -> None:
    # Exercise the real unavailable-converter path without changing the system.
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("DOCLING_LIBREOFFICE_CMD", raising=False)
    monkeypatch.setattr(
        "docling.backend.msword_backend.get_docx_to_pdf_converter", lambda: None
    )
    monkeypatch.setattr(
        "docling.backend.msexcel_backend.get_docx_to_pdf_converter", lambda: None
    )
    jpeg = _cmyk_jpeg(profile)
    path = tmp_path / f"cmyk.{suffix}"
    if suffix == "pptx":
        deck = Presentation()
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        slide.shapes.add_picture(jpeg, Inches(1), Inches(1))
        deck.save(path)
    elif suffix == "xlsx":
        _save_xlsx_with_picture(path, jpeg)
    else:
        # python-docx rejects CMYK JPEG headers, so replace the embedded RGB JPEG.
        rgb = BytesIO()
        Image.new("RGB", (200, 100), "red").save(rgb, format="JPEG")
        rgb.seek(0)
        source = Document()
        source.add_picture(rgb, width=Inches(2))
        template = BytesIO()
        source.save(template)
        template.seek(0)
        with ZipFile(template) as src, ZipFile(path, "w") as dst:
            for entry in src.infolist():
                data = (
                    jpeg.getvalue()
                    if entry.filename.startswith("word/media/")
                    else src.read(entry.filename)
                )
                dst.writestr(entry, data)

    result = DocumentConverter().convert(path)
    assert result.status == ConversionStatus.SUCCESS
    assert len(result.document.pictures) == 1
    image_ref = result.document.pictures[0].image
    assert image_ref is not None
    image = image_ref.pil_image
    assert image is not None
    assert image.size == (200, 100)
    assert image.convert("RGB").getpixel((0, 0)) == (255, 0, 0)


def test_cmyk_xlsx_picture_is_not_rendered_by_libreoffice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stand in for an installed LibreOffice: it would re-render the picture at
    # another size, so the stored picture must not be routed to it.
    monkeypatch.setattr(
        MsExcelDocumentBackend,
        "_convert_emf_to_pil",
        lambda self, image_bytes: Image.new("RGB", (401, 201), "white"),
    )
    path = tmp_path / "cmyk.xlsx"
    _save_xlsx_with_picture(path, _cmyk_jpeg())

    result = DocumentConverter().convert(path)
    assert result.status == ConversionStatus.SUCCESS
    assert len(result.document.pictures) == 1
    image_ref = result.document.pictures[0].image
    assert image_ref is not None
    image = image_ref.pil_image
    assert image is not None
    assert image.size == (200, 100)


def test_embedded_icc_profile_is_used() -> None:
    # LAB also requires conversion for PNG and has a profile Pillow can generate.
    image = Image.new("LAB", (2, 2), (128, 180, 80))
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB"))
    image.info["icc_profile"] = profile.tobytes()
    expected = ImageCms.profileToProfile(
        image, profile, ImageCms.createProfile("sRGB"), outputMode="RGB"
    )
    actual = normalize_image_for_png(image)
    assert actual.mode == "RGB"
    assert actual.tobytes() == expected.tobytes()


def test_pa_preserves_per_pixel_alpha_in_png() -> None:
    image = Image.new("PA", (2, 1))
    image.putpalette([10, 20, 30] + [0, 0, 0] * 255)
    image.putpixel((0, 0), (0, 40))
    image.putpixel((1, 0), (0, 200))

    buffer = BytesIO()
    normalize_image_for_png(image).save(buffer, format="PNG")
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        assert decoded.mode == "RGBA"
        assert decoded.getpixel((0, 0)) == (10, 20, 30, 40)
        assert decoded.getpixel((1, 0)) == (10, 20, 30, 200)


@pytest.mark.parametrize(
    "mode, color", [("RGB", (10, 20, 30)), ("RGBA", (10, 20, 30, 40))]
)
def test_compatible_pixels_survive_png_serialization(mode: str, color: tuple) -> None:
    image = Image.new(mode, (2, 2), color)
    buffer = BytesIO()
    normalize_image_for_png(image).save(buffer, format="PNG")
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        assert decoded.mode == mode
        assert decoded.getpixel((0, 0)) == color
