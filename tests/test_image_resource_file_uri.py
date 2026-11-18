# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Check file URI resolution on native filesystems."""

import sys
from io import BytesIO
from pathlib import Path

import pytest

from docling.backend.utils.image_resource_loader import ImageResourceLoader
from docling.datamodel.backend_options import HTMLBackendOptions, MarkdownBackendOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import (
    DocumentConverter,
    HTMLFormatOption,
    MarkdownFormatOption,
)

pytestmark = pytest.mark.cross_platform


def test_resolve_relative_path_accepts_file_uri_base(tmp_path):
    loader = ImageResourceLoader()
    base_path = (tmp_path / "doc.md").as_uri()

    resolved = loader.resolve_relative_path("figure.png", base_path)

    assert Path(resolved) == (tmp_path / "figure.png").resolve()


@pytest.mark.parametrize("authority", ["localhost", "LOCALHOST", "LoCaLhOsT"])
def test_resolve_relative_path_accepts_localhost_file_uri(tmp_path, authority):
    loader = ImageResourceLoader()
    base_path = (
        (tmp_path / "doc.md").as_uri().replace("file:///", f"file://{authority}/", 1)
    )

    resolved = loader.resolve_relative_path("figure.png", base_path)

    assert Path(resolved) == (tmp_path / "figure.png").resolve()


@pytest.mark.skipif(
    sys.platform != "win32", reason="UNC filesystem paths require Windows"
)
def test_resolve_relative_path_accepts_unc_file_uri():
    loader = ImageResourceLoader()

    resolved = loader.resolve_relative_path(
        "figure.png", "file://server/share/docs/doc.md"
    )

    assert Path(resolved) == Path("//server/share/docs/figure.png")


def test_resolve_relative_path_keeps_file_uri_confinement(tmp_path):
    loader = ImageResourceLoader()
    base_path = (tmp_path / "doc.md").as_uri()

    with pytest.raises(ValueError, match="Path traversal blocked"):
        loader.resolve_relative_path("../outside.png", base_path)


@pytest.mark.parametrize(
    "base", ["file://", "file://localhost", "file://LOCALHOST", "file://server"]
)
def test_resolve_relative_path_rejects_empty_file_uri(base):
    loader = ImageResourceLoader()

    with pytest.raises(ValueError, match="empty path"):
        loader.resolve_relative_path("figure.png", base)


def test_file_uri_decodes_real_directory(tmp_path):
    directory = tmp_path / "space 中文 %23"
    directory.mkdir()
    image_path = directory / "figure.png"
    image_path.write_bytes(b"target")
    resolved = ImageResourceLoader().resolve_relative_path(
        "figure.png", (directory / "doc.md").as_uri()
    )
    assert Path(resolved).read_bytes() == b"target"


@pytest.mark.skipif(sys.platform == "win32", reason="Windows supports UNC authorities")
def test_non_local_authority_is_rejected():
    with pytest.raises(ValueError, match="Windows UNC"):
        ImageResourceLoader().resolve_relative_path(
            "figure.png", "file://server/share/docs/doc.md"
        )


def test_remote_base_keeps_url_resolution():
    assert (
        ImageResourceLoader().resolve_relative_path(
            "figure.png", "https://example.com/docs/article.html"
        )
        == "https://example.com/docs/figure.png"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Windows supports UNC authorities")
@pytest.mark.parametrize("format", [InputFormat.HTML, InputFormat.MD])
def test_backend_keeps_content_after_rejected_authority(format, monkeypatch):
    calls = []

    def record_read(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Rejected authority must not read an image")

    monkeypatch.setattr(ImageResourceLoader, "load_image_data", record_read)
    source = "file://server/share/article.html"
    if format == InputFormat.HTML:
        option = HTMLFormatOption(
            backend_options=HTMLBackendOptions(
                source_uri=source, fetch_images=True, enable_local_fetch=True
            )
        )
        content = '<p>Before</p><img src="figure.png"><p>After</p>'
        name = "article.html"
    else:
        option = MarkdownFormatOption(
            backend_options=MarkdownBackendOptions(
                source_uri=source, fetch_images=True, enable_local_fetch=True
            )
        )
        content = "Before\n\n![Caption](figure.png)\n\nAfter\n"
        name = "article.md"
    converter = DocumentConverter(
        allowed_formats=[format], format_options={format: option}
    )
    with pytest.warns(UserWarning, match="Windows UNC"):
        doc = converter.convert(
            DocumentStream(name=name, stream=BytesIO(content.encode()))
        ).document
    markdown = doc.export_to_markdown()
    assert markdown.index("Before") < markdown.index("After")
    assert len(doc.pictures) == 1 and doc.pictures[0].image is None
    assert calls == []
