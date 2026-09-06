# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import base64
import mimetypes
from io import BytesIO
from pathlib import Path, PureWindowsPath
from unittest.mock import Mock, patch

import pytest
from PIL import Image
from pydantic import AnyUrl

from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.backend_options import HTMLBackendOptions
from docling.datamodel.base_models import (
    ConversionStatus,
    DocumentStream,
    InputFormat,
)
from docling.datamodel.document import _DocumentConversionInput
from docling.document_converter import DocumentConverter, HTMLFormatOption

MHTML_DATA_DIR = Path("tests/data/mhtml/sources")


def _red_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _blue_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (1, 1), color=(0, 0, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def _archive(
    html: str,
    extra_parts: str = "",
    related_options: str = "",
    root_location: str = "https://example.com/docs/page.html",
) -> bytes:
    return (
        "MIME-Version: 1.0\r\n"
        f'Content-Type: multipart/related; boundary="B"{related_options}\r\n\r\n'
        "--B\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "Content-ID: <root@example>\r\n"
        f"Content-Location: {root_location}\r\n\r\n"
        f"{html}\r\n"
        f"{extra_parts}"
        "--B--\r\n"
    ).encode()


def _convert_stream(
    data: bytes,
    name: str = "sample.mhtml",
    options: HTMLBackendOptions | None = None,
):
    converter = (
        DocumentConverter(allowed_formats=[InputFormat.MHTML])
        if options is None
        else DocumentConverter(
            allowed_formats=[InputFormat.MHTML],
            format_options={
                InputFormat.MHTML: HTMLFormatOption(backend_options=options)
            },
        )
    )
    result = converter.convert(DocumentStream(name=name, stream=BytesIO(data)))
    assert result.status == ConversionStatus.SUCCESS
    return result.document


@pytest.mark.parametrize("extension", ["mhtml", "mht"])
def test_mhtml_extension_detection(tmp_path: Path, extension: str):
    path = tmp_path / f"sample.{extension}"
    path.write_bytes(_archive("<html><body><p>Detected</p></body></html>"))
    conversion_input = _DocumentConversionInput(path_or_stream_iterator=[path])

    assert conversion_input._guess_format(path) == InputFormat.MHTML
    stream = DocumentStream(name=path.name, stream=BytesIO(path.read_bytes()))
    assert conversion_input._guess_format(stream) == InputFormat.MHTML
    assert mimetypes.guess_type(path.name)[0] == "application/x-mimearchive"


def test_root_html_preserves_standard_html_semantics():
    html = """
    <html><body>
      <h1>Пример</h1>
      <p>Текст with <a href="https://example.com/details">a link</a>.</p>
      <ul><li>One</li><li>Two</li></ul>
      <table><tr><th>Key</th><th>Value</th></tr><tr><td>A</td><td>1</td></tr></table>
    </body></html>
    """
    doc = _convert_stream(_archive(html))
    markdown = doc.export_to_markdown()

    assert "# Пример" in markdown
    assert "[a link](https://example.com/details)" in markdown
    assert "- One" in markdown and "- Two" in markdown
    assert len(doc.tables) == 1
    assert [cell.text for cell in doc.tables[0].data.table_cells] == [
        "Key",
        "Value",
        "A",
        "1",
    ]


def test_related_start_selects_root_inside_nested_multipart():
    data = (
        b"MIME-Version: 1.0\r\n"
        b'Content-Type: multipart/mixed; boundary="OUT"\r\n\r\n'
        b"--OUT\r\n"
        b'Content-Type: multipart/related; boundary="IN"; start="<wanted@example>"\r\n\r\n'
        b"--IN\r\nContent-Type: text/html\r\nContent-ID: <wrong@example>\r\n\r\n"
        b"<p>Wrong root</p>\r\n"
        b"--IN\r\nContent-Type: text/html\r\nContent-ID: <wanted@example>\r\n\r\n"
        b"<h1>Selected root</h1>\r\n"
        b"--IN--\r\n--OUT--\r\n"
    )

    markdown = _convert_stream(data).export_to_markdown()
    assert "Selected root" in markdown
    assert "Wrong root" not in markdown


def test_related_start_selects_html_from_multipart_alternative():
    data = (
        b"MIME-Version: 1.0\r\n"
        b'Content-Type: multipart/related; boundary="B"; start="<root@example>"\r\n\r\n'
        b'--B\r\nContent-Type: multipart/alternative; boundary="A"\r\n'
        b"Content-ID: <root@example>\r\n\r\n"
        b"--A\r\nContent-Type: text/plain\r\n\r\nPlain fallback\r\n"
        b"--A\r\nContent-Type: text/html\r\n\r\n<h1>HTML alternative</h1>\r\n"
        b"--A--\r\n--B--\r\n"
    )

    markdown = _convert_stream(data).export_to_markdown()

    assert "# HTML alternative" in markdown
    assert "Plain fallback" not in markdown


def test_related_without_start_does_not_select_later_html():
    data = (
        b"MIME-Version: 1.0\r\n"
        b'Content-Type: multipart/related; boundary="B"\r\n\r\n'
        b"--B\r\nContent-Type: text/plain\r\n\r\nFirst root\r\n"
        b"--B\r\nContent-Type: text/html\r\n\r\n<h1>Wrong root</h1>\r\n"
        b"--B--\r\n"
    )
    converter = DocumentConverter(allowed_formats=[InputFormat.MHTML])
    result = converter.convert(
        DocumentStream(name="invalid.mhtml", stream=BytesIO(data)),
        raises_on_error=False,
    )

    assert result.status == ConversionStatus.FAILURE
    assert result.errors


def test_declared_non_utf8_charset_is_decoded():
    html = "<html><body><p>Привет, мир</p></body></html>".encode("windows-1251")
    encoded = base64.b64encode(html).decode()
    data = (
        "MIME-Version: 1.0\r\n"
        "Content-Type: text/html; charset=windows-1251\r\n"
        "Content-Transfer-Encoding: base64\r\n\r\n"
        f"{encoded}\r\n"
    ).encode()

    assert "Привет, мир" in _convert_stream(data).export_to_markdown()


def test_real_blink_fixture_decodes_quoted_printable_html():
    path = MHTML_DATA_DIR / "example.mhtml"
    converter = DocumentConverter(allowed_formats=[InputFormat.MHTML])
    result = converter.convert(path)

    assert result.status == ConversionStatus.SUCCESS
    markdown = result.document.export_to_markdown()
    assert "# Example Domain" in markdown
    assert "[Learn more](https://iana.org/domains/example)" in markdown


def test_embedded_images_resolve_by_location_and_cid():
    png = base64.b64encode(_red_png()).decode()
    parts = (
        "--B\r\nContent-Type: image/png\r\n"
        "Content-Location: https://example.com/images/location.png\r\n"
        f"Content-Transfer-Encoding: base64\r\n\r\n{png}\r\n"
        "--B\r\nContent-Type: image/png\r\n"
        "Content-ID: <CID-IMAGE@EXAMPLE>\r\n"
        f"Content-Transfer-Encoding: base64\r\n\r\n{png}\r\n"
    )
    html = (
        '<html><body><img src="../images/location.png">'
        '<img src="cid:cid-image@example"></body></html>'
    )
    doc = _convert_stream(
        _archive(html, parts), options=HTMLBackendOptions(fetch_images=True)
    )

    assert len(doc.pictures) == 2
    assert all(picture.image is not None for picture in doc.pictures)
    assert all(picture.get_image(doc).size == (1, 1) for picture in doc.pictures)


def test_embedded_relative_location_resolves_with_file_root():
    png = base64.b64encode(_red_png()).decode()
    parts = (
        "--B\r\nContent-Type: image/png\r\n"
        "Content-Location: images/file.png\r\n"
        f"Content-Transfer-Encoding: base64\r\n\r\n{png}\r\n"
    )
    data = _archive('<html><body><img src="images/file.png"></body></html>', parts)
    data = data.replace(
        b"https://example.com/docs/page.html", b"file:///C:/saved/page.html"
    )

    doc = _convert_stream(data, options=HTMLBackendOptions(fetch_images=True))

    assert doc.pictures[0].image is not None


def test_default_fetch_images_leaves_archive_image_as_placeholder():
    png = base64.b64encode(_red_png()).decode()
    parts = (
        "--B\r\nContent-Type: image/png\r\nContent-ID: <image@example>\r\n"
        f"Content-Transfer-Encoding: base64\r\n\r\n{png}\r\n"
    )
    doc = _convert_stream(
        _archive('<html><body><img src="cid:image@example"></body></html>', parts)
    )

    assert doc.pictures[0].image is None


def test_archive_image_precedes_explicit_remote_fetch():
    png = base64.b64encode(_red_png()).decode()
    parts = (
        "--B\r\nContent-Type: image/png\r\n"
        "Content-Location: https://example.com/image.png\r\n"
        f"Content-Transfer-Encoding: base64\r\n\r\n{png}\r\n"
    )
    html = '<html><body><img src="https://example.com/image.png"></body></html>'

    with patch(
        "docling.backend.utils.image_resource_loader.requests.Session.get"
    ) as mocked_get:
        doc = _convert_stream(
            _archive(html, parts),
            options=HTMLBackendOptions(fetch_images=True, enable_remote_fetch=True),
        )

    mocked_get.assert_not_called()
    assert doc.pictures[0].get_image(doc).getpixel((0, 0)) == (255, 0, 0)


def test_resources_are_limited_to_selected_related_scope():
    red = base64.b64encode(_red_png()).decode()
    blue = base64.b64encode(_blue_png()).decode()
    data = (
        "MIME-Version: 1.0\r\n"
        'Content-Type: multipart/mixed; boundary="OUT"\r\n\r\n'
        '--OUT\r\nContent-Type: multipart/related; boundary="ONE"\r\n\r\n'
        "--ONE\r\nContent-Type: text/html\r\n\r\n"
        '<html><body><img src="cid:shared@example"></body></html>\r\n'
        "--ONE\r\nContent-Type: image/png\r\nContent-ID: <shared@example>\r\n"
        f"Content-Transfer-Encoding: base64\r\n\r\n{red}\r\n"
        "--ONE--\r\n"
        '--OUT\r\nContent-Type: multipart/related; boundary="TWO"\r\n\r\n'
        "--TWO\r\nContent-Type: text/html\r\n\r\n<p>Other scope</p>\r\n"
        "--TWO\r\nContent-Type: image/png\r\nContent-ID: <shared@example>\r\n"
        f"Content-Transfer-Encoding: base64\r\n\r\n{blue}\r\n"
        "--TWO--\r\n--OUT--\r\n"
    ).encode()

    doc = _convert_stream(data, options=HTMLBackendOptions(fetch_images=True))

    assert doc.pictures[0].get_image(doc).getpixel((0, 0)) == (255, 0, 0)


def test_duplicate_resource_labels_keep_first_part():
    red = base64.b64encode(_red_png()).decode()
    blue = base64.b64encode(_blue_png()).decode()
    parts = (
        "--B\r\nContent-Type: image/png\r\nContent-ID: <same@example>\r\n"
        f"Content-Transfer-Encoding: base64\r\n\r\n{red}\r\n"
        "--B\r\nContent-Type: image/png\r\nContent-ID: <same@example>\r\n"
        f"Content-Transfer-Encoding: base64\r\n\r\n{blue}\r\n"
    )
    html = '<html><body><img src="cid:same@example"></body></html>'

    doc = _convert_stream(
        _archive(html, parts), options=HTMLBackendOptions(fetch_images=True)
    )

    assert doc.pictures[0].get_image(doc).getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize(
    ("root_location", "expected"),
    [
        (r"C:\safe\page.html", r"C:\safe\images\a.png"),
        ("C:/safe/page.html", r"C:\safe\images\a.png"),
    ],
)
def test_windows_locations_are_joined_without_urljoin(
    root_location: str, expected: str
):
    first = HTMLDocumentBackend._join_mhtml_location(root_location, "images/a.png")
    equivalent = HTMLDocumentBackend._join_mhtml_location(
        root_location, "./images/a.png"
    )

    assert PureWindowsPath(first) == PureWindowsPath(expected)
    assert PureWindowsPath(equivalent) == PureWindowsPath(expected)


@pytest.mark.parametrize(
    "root_location",
    [r"C:\outside\page.html", "C:/outside/page.html", "file:///C:/outside/page.html"],
)
def test_windows_root_outside_source_directory_never_reaches_shared_loader(
    root_location: str,
):
    options = HTMLBackendOptions(
        fetch_images=True,
        enable_local_fetch=True,
        source_uri=PureWindowsPath(r"C:\safe\archive.mhtml"),
    )
    data = _archive(
        '<html><body><img src="missing.png"></body></html>',
        root_location=root_location,
    )

    with patch(
        "docling.backend.html_backend.ImageResourceLoader.create_image_ref"
    ) as shared_loader:
        doc = _convert_stream(data, options=options)

    shared_loader.assert_not_called()
    assert doc.pictures[0].image is None


@pytest.mark.parametrize(
    "root_location",
    [r"C:\safe\page.html", "C:/safe/page.html", "file:///C:/safe/page.html"],
)
def test_equivalent_windows_archive_locations_resolve(root_location: str):
    png = base64.b64encode(_red_png()).decode()
    parts = (
        "--B\r\nContent-Type: image/png\r\nContent-Location: images/a.png\r\n"
        f"Content-Transfer-Encoding: base64\r\n\r\n{png}\r\n"
    )
    options = HTMLBackendOptions(
        fetch_images=True,
        source_uri=PureWindowsPath(r"C:\safe\archive.mhtml"),
    )
    html = '<html><body><img src="./images/a.png"></body></html>'

    doc = _convert_stream(
        _archive(html, parts, root_location=root_location), options=options
    )

    assert doc.pictures[0].get_image(doc).getpixel((0, 0)) == (255, 0, 0)


def test_missing_and_unsupported_images_remain_placeholders():
    parts = (
        "--B\r\nContent-Type: image/svg+xml\r\n"
        "Content-ID: <vector@example>\r\n\r\n"
        '<svg xmlns="http://www.w3.org/2000/svg"></svg>\r\n'
        "--B\r\nContent-Type: image/png\r\n"
        "Content-ID: <empty@example>\r\n"
        "Content-Transfer-Encoding: base64\r\n\r\n\r\n"
    )
    html = (
        '<html><body><img src="cid:missing@example">'
        '<img src="cid:vector@example">'
        '<img src="cid:empty@example"></body></html>'
    )

    with pytest.warns(UserWarning, match="embedded MHTML image"):
        doc = _convert_stream(
            _archive(html, parts), options=HTMLBackendOptions(fetch_images=True)
        )

    assert len(doc.pictures) == 3
    assert all(picture.image is None for picture in doc.pictures)


def test_archive_image_size_limit_is_enforced():
    png = base64.b64encode(_red_png()).decode()
    parts = (
        "--B\r\nContent-Type: image/png\r\nContent-ID: <large@example>\r\n"
        f"Content-Transfer-Encoding: base64\r\n\r\n{png}\r\n"
    )
    html = '<html><body><img src="cid:large@example"></body></html>'

    with pytest.warns(UserWarning, match="exceeds size limit"):
        doc = _convert_stream(
            _archive(html, parts),
            options=HTMLBackendOptions(
                fetch_images=True, max_image_data_base64_bytes=8
            ),
        )

    assert doc.pictures[0].image is None


def test_missing_remote_image_respects_default_fetch_permission():
    html = '<html><body><img src="https://example.com/missing.png"></body></html>'

    with (
        patch(
            "docling.backend.utils.image_resource_loader.requests.Session.get"
        ) as mocked_get,
        pytest.warns(UserWarning, match="Fetching remote resources"),
    ):
        doc = _convert_stream(
            _archive(html), options=HTMLBackendOptions(fetch_images=True)
        )
        mocked_get.assert_not_called()

    assert len(doc.pictures) == 1
    assert doc.pictures[0].image is None


def test_missing_remote_image_is_fetched_when_explicitly_enabled():
    html = '<html><body><img src="image.png"></body></html>'
    response = Mock()
    response.headers = {}
    response.raise_for_status = Mock()
    response.iter_content = Mock(return_value=[_red_png()])
    response.is_redirect = False
    response.is_permanent_redirect = False

    with patch(
        "docling.backend.utils.image_resource_loader.requests.Session.get",
        return_value=response,
    ) as mocked_get:
        options = HTMLBackendOptions(
            fetch_images=True,
            enable_remote_fetch=True,
            source_uri=AnyUrl("https://example.com/archive.mhtml"),
        )
        doc = _convert_stream(
            _archive(html, root_location="page.html"), options=options
        )

    mocked_get.assert_called_once()
    assert mocked_get.call_args.args[0] == "https://example.com/image.png"
    assert doc.pictures[0].image is not None
    assert str(options.source_uri) == "https://example.com/archive.mhtml"


def test_relative_root_path_uses_archive_directory_for_local_fallback(
    tmp_path: Path,
):
    (tmp_path / "fallback.png").write_bytes(_red_png())
    archive_path = tmp_path / "archive.mhtml"
    archive_path.write_bytes(
        _archive(
            '<html><body><img src="fallback.png"></body></html>',
            root_location="page.html",
        )
    )
    options = HTMLBackendOptions(fetch_images=True, enable_local_fetch=True)
    converter = DocumentConverter(
        allowed_formats=[InputFormat.MHTML],
        format_options={InputFormat.MHTML: HTMLFormatOption(backend_options=options)},
    )

    doc = converter.convert(archive_path).document

    image = doc.pictures[0].get_image(doc)
    assert image is not None
    assert image.size == (1, 1)
    assert options.source_uri is None


def test_relative_root_stream_uses_non_filesystem_base():
    options = HTMLBackendOptions(fetch_images=True, enable_local_fetch=True)
    data = _archive(
        '<html><body><img src="fallback.png"></body></html>',
        root_location="page.html",
    )

    with patch(
        "docling.backend.html_backend.ImageResourceLoader.create_image_ref"
    ) as shared_loader:
        doc = _convert_stream(data, options=options)

    shared_loader.assert_not_called()
    assert doc.pictures[0].image is None
    assert options.source_uri is None


def test_data_uri_image_uses_shared_html_loader():
    encoded = base64.b64encode(_red_png()).decode()
    html = f'<html><body><img src="data:image/png;base64,{encoded}"></body></html>'
    doc = _convert_stream(_archive(html), options=HTMLBackendOptions(fetch_images=True))

    assert doc.pictures[0].image is not None
    assert doc.pictures[0].get_image(doc).size == (1, 1)


@pytest.mark.parametrize(
    "data",
    [
        b"MIME-Version: 1.0\r\nContent-Type: text/plain\r\n\r\nplain text\r\n",
        b"not a mime message",
        b'MIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary="B"\r\n\r\n--B--\r\n',
        _archive(""),
    ],
)
def test_unusable_input_fails_cleanly(data: bytes):
    converter = DocumentConverter(allowed_formats=[InputFormat.MHTML])
    result = converter.convert(
        DocumentStream(name="invalid.mhtml", stream=BytesIO(data)),
        raises_on_error=False,
    )

    assert result.status == ConversionStatus.FAILURE
    assert result.errors


@pytest.mark.parametrize(
    "related_options, extra_parts",
    [
        ('; start="<missing@example>"', ""),
        (
            '; start="<not-html@example>"',
            "--B\r\nContent-Type: text/plain\r\n"
            "Content-ID: <not-html@example>\r\n\r\nnot html\r\n",
        ),
    ],
)
def test_invalid_related_start_fails_cleanly(related_options: str, extra_parts: str):
    data = _archive(
        "<p>Fallback must not be selected</p>", extra_parts, related_options
    )
    converter = DocumentConverter(allowed_formats=[InputFormat.MHTML])
    result = converter.convert(
        DocumentStream(name="invalid.mhtml", stream=BytesIO(data)),
        raises_on_error=False,
    )

    assert result.status == ConversionStatus.FAILURE
    assert result.errors


def test_browser_rendering_is_rejected_for_mhtml():
    converter = DocumentConverter(
        allowed_formats=[InputFormat.MHTML],
        format_options={
            InputFormat.MHTML: HTMLFormatOption(
                backend_options=HTMLBackendOptions(render_page=True)
            )
        },
    )
    result = converter.convert(
        DocumentStream(
            name="render.mhtml",
            stream=BytesIO(_archive("<p>Render</p>")),
        ),
        raises_on_error=False,
    )

    assert result.status == ConversionStatus.FAILURE
    assert any("not supported for MHTML" in err.error_message for err in result.errors)


@pytest.mark.parametrize("name", ["original.mhtml", "original.mht"])
def test_path_and_stream_preserve_original_origin(tmp_path: Path, name: str):
    data = _archive("<html><body><p>Origin</p></body></html>")
    path = tmp_path / name
    path.write_bytes(data)
    converter = DocumentConverter(allowed_formats=[InputFormat.MHTML])

    path_doc = converter.convert(path).document
    stream_doc = converter.convert(
        DocumentStream(name=name, stream=BytesIO(data))
    ).document

    for doc in (path_doc, stream_doc):
        assert doc.name == Path(name).stem
        assert doc.origin is not None
        assert doc.origin.filename == name
        assert doc.origin.mimetype == "application/x-mimearchive"


def test_backend_contract():
    assert HTMLDocumentBackend.supported_formats() == {
        InputFormat.HTML,
        InputFormat.MHTML,
    }
    assert not HTMLDocumentBackend.supports_pagination()
