# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import base64
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from docling.datamodel.backend_options import (
    AsciiDocBackendOptions,
    HTMLBackendOptions,
    MarkdownBackendOptions,
)
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.document_converter import (
    AsciiDocFormatOption,
    DocumentConverter,
    HTMLFormatOption,
    MarkdownFormatOption,
)


def _cmyk_jpeg() -> bytes:
    buffer = BytesIO()
    Image.new("CMYK", (200, 100), (0, 255, 255, 0)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _assert_single_red_picture(result) -> None:
    assert result.status == ConversionStatus.SUCCESS
    assert len(result.document.pictures) == 1
    image_ref = result.document.pictures[0].image
    assert image_ref is not None
    image = image_ref.pil_image
    assert image is not None
    assert image.size == (200, 100)
    assert image.convert("RGB").getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize(
    "suffix, content, format_name, format_option, backend_options",
    [
        (
            "html",
            '<html><body><img src="cmyk.jpg"></body></html>',
            InputFormat.HTML,
            HTMLFormatOption,
            HTMLBackendOptions,
        ),
        (
            "md",
            "![cmyk](cmyk.jpg)\n",
            InputFormat.MD,
            MarkdownFormatOption,
            MarkdownBackendOptions,
        ),
        (
            "adoc",
            "image::cmyk.jpg[]\n",
            InputFormat.ASCIIDOC,
            AsciiDocFormatOption,
            AsciiDocBackendOptions,
        ),
    ],
)
def test_cmyk_picture_is_kept(
    tmp_path: Path,
    suffix: str,
    content: str,
    format_name: InputFormat,
    format_option: type,
    backend_options: type,
) -> None:
    (tmp_path / "cmyk.jpg").write_bytes(_cmyk_jpeg())
    source = tmp_path / f"source.{suffix}"
    source.write_text(content)

    options = backend_options(
        enable_local_fetch=True, fetch_images=True, source_uri=str(source)
    )
    converter = DocumentConverter(
        allowed_formats=[format_name],
        format_options={format_name: format_option(backend_options=options)},
    )

    _assert_single_red_picture(converter.convert(source))


def test_cmyk_picture_in_mhtml_is_kept(tmp_path: Path) -> None:
    payload = base64.b64encode(_cmyk_jpeg()).decode("ascii")
    source = tmp_path / "source.mhtml"
    source.write_bytes(
        (
            "MIME-Version: 1.0\r\n"
            'Content-Type: multipart/related; boundary="B"\r\n\r\n'
            "--B\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            "Content-ID: <root@example>\r\n"
            "Content-Location: https://example.com/docs/page.html\r\n\r\n"
            '<html><body><img src="cid:image1@example"></body></html>\r\n'
            "--B\r\n"
            "Content-Type: image/jpeg\r\n"
            "Content-Transfer-Encoding: base64\r\n"
            "Content-ID: <image1@example>\r\n\r\n"
            f"{payload}\r\n"
            "--B--\r\n"
        ).encode()
    )

    _assert_single_red_picture(
        DocumentConverter(
            allowed_formats=[InputFormat.MHTML],
            format_options={
                InputFormat.MHTML: HTMLFormatOption(
                    backend_options=HTMLBackendOptions(fetch_images=True)
                )
            },
        ).convert(source)
    )
