# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Exercise local JATS figures through the stream conversion boundary."""

import builtins
import sys
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument, ImageRefMode, PictureItem, TextItem
from PIL import Image

from docling.backend.utils.image_resource_loader import ImageResourceLoader
from docling.datamodel.backend_options import JatsBackendOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import DocumentConverter, XMLJatsFormatOption

pytestmark = pytest.mark.cross_platform


def convert_figure(graphic: str, options: JatsBackendOptions) -> DoclingDocument:
    xml = f"""<!DOCTYPE article PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving and Interchange DTD v1.2 20190208//EN" "JATS-archivearticle1.dtd">
<article xmlns:xlink="http://www.w3.org/1999/xlink">
<front><article-meta><title-group><article-title>Figure test</article-title>
</title-group></article-meta></front>
<body><p>Before figure.</p><fig><label>Fig. 1</label>
<caption><p>Green figure.</p></caption>{graphic}</fig>
<p>After figure.</p></body></article>"""
    converter = DocumentConverter(
        allowed_formats=[InputFormat.XML_JATS],
        format_options={
            InputFormat.XML_JATS: XMLJatsFormatOption(backend_options=options)
        },
    )
    return converter.convert(
        DocumentStream(name="article.nxml", stream=BytesIO(xml.encode()))
    ).document


def assert_structure(doc: DoclingDocument) -> PictureItem:
    title = doc.body.children[0].resolve(doc)
    assert isinstance(title, TextItem) and title.text == "Figure test"
    before, picture, after = [ref.resolve(doc) for ref in title.children]
    assert isinstance(before, TextItem) and before.text == "Before figure."
    assert isinstance(picture, PictureItem)
    assert isinstance(after, TextItem) and after.text == "After figure."
    assert before.parent == picture.parent == after.parent == title.get_ref()
    assert len(picture.captions) == 1
    caption = picture.captions[0].resolve(doc)
    assert isinstance(caption, TextItem) and caption.text == "Fig. 1 Green figure."
    markdown = doc.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
    assert markdown.index(before.text) < markdown.index(caption.text)
    assert markdown.index(caption.text) < markdown.index(after.text)
    return picture


def source_path(tmp_path: Path, kind: str) -> Path | str:
    path = tmp_path / "article.nxml"
    if kind == "path":
        return path
    uri = path.as_uri()
    return (
        uri.replace("file:///", "file://LoCaLhOsT/", 1) if kind == "localhost" else uri
    )


@pytest.mark.parametrize("source_kind", ["path", "file", "localhost"])
@pytest.mark.parametrize(
    "graphic",
    [
        '<graphic xlink:href="figure.png"/>',
        '<graphic xlink:href="figure"/>',
        '<alternatives><graphic xlink:href="missing.png"/>'
        '<graphic xlink:href="figure.png"/></alternatives>',
    ],
)
def test_stream_loads_local_figure(tmp_path: Path, source_kind: str, graphic: str):
    directory = tmp_path / "space 中文"
    directory.mkdir()
    Image.new("RGB", (7, 5), (0, 255, 0)).save(directory / "figure.png")
    doc = convert_figure(
        graphic,
        JatsBackendOptions(
            fetch_images=True,
            enable_local_fetch=True,
            source_uri=source_path(directory, source_kind),
        ),
    )
    picture = assert_structure(doc)
    image = picture.get_image(doc)
    assert image is not None and image.size == (7, 5)
    assert image.getpixel((0, 0)) == (0, 255, 0)
    assert "data:image/png;base64," in doc.export_to_markdown(
        image_mode=ImageRefMode.EMBEDDED
    )


@pytest.mark.parametrize("source_kind", ["path", "file"])
@pytest.mark.parametrize("damage", ["unrecognized", "truncated"])
@pytest.mark.parametrize("fallback", [False, True])
def test_damaged_figure_keeps_content(
    tmp_path: Path, source_kind: str, damage: str, fallback: bool
):
    buffer = BytesIO()
    Image.new("RGB", (32, 32), (0, 255, 0)).save(buffer, format="BMP")
    (tmp_path / "broken.bmp").write_bytes(
        buffer.getvalue()[:100] if damage == "truncated" else b"not an image"
    )
    Image.new("RGB", (7, 5), (0, 255, 0)).save(tmp_path / "figure.png")
    graphic = '<graphic xlink:href="broken.bmp"/>'
    if fallback:
        graphic = (
            f'<alternatives>{graphic}<graphic xlink:href="figure.png"/></alternatives>'
        )
    with pytest.warns(UserWarning, match="Could not process an image"):
        doc = convert_figure(
            graphic,
            JatsBackendOptions(
                fetch_images=True,
                enable_local_fetch=True,
                source_uri=source_path(tmp_path, source_kind),
            ),
        )
    picture = assert_structure(doc)
    if fallback:
        image = picture.get_image(doc)
        assert image is not None and image.getpixel((0, 0)) == (0, 255, 0)
    else:
        assert picture.image is None


@pytest.mark.parametrize("source_kind", ["path", "file"])
@pytest.mark.parametrize("failure", ["missing", "absolute", "traversal", "svg"])
def test_local_figure_guards(tmp_path: Path, source_kind: str, failure: str, caplog):
    directory = tmp_path / "article"
    directory.mkdir()
    Image.new("RGB", (7, 5)).save(tmp_path / "outside.png")
    Image.new("RGB", (7, 5)).save(directory / "fallback.png")
    href = {
        "missing": "missing.png",
        "absolute": (tmp_path / "outside.png").as_posix(),
        "traversal": "../outside.png",
        "svg": "figure.svg",
    }[failure]
    graphic = f'<graphic xlink:href="{href}"/>'
    if failure in {"absolute", "traversal"}:
        graphic = f'<alternatives>{graphic}<graphic xlink:href="fallback.png"/></alternatives>'
    options = JatsBackendOptions(
        fetch_images=True,
        enable_local_fetch=True,
        source_uri=source_path(directory, source_kind),
    )
    if failure == "svg":
        doc = convert_figure(graphic, options)
        assert "Skipping unsupported JATS SVG" in caplog.text
    else:
        message = {
            "missing": "no matching local file",
            "absolute": "Absolute paths are not allowed",
            "traversal": "Path traversal blocked",
        }[failure]
        with pytest.warns(UserWarning, match=message):
            doc = convert_figure(graphic, options)
    picture = assert_structure(doc)
    assert (picture.image is not None) == (failure == "absolute")


@pytest.mark.parametrize(
    "source",
    [
        "file://",
        "file://localhost",
        "file://LOCALHOST",
        "file:///",
        pytest.param(
            "file:///C:/",
            marks=pytest.mark.skipif(
                sys.platform != "win32", reason="Windows drive root"
            ),
        ),
        "file://server/share/",
    ],
)
def test_root_file_uri_is_not_a_document(source: str, monkeypatch):
    reads = []

    def record_read(*args, **kwargs):
        reads.append(args)
        raise AssertionError("Invalid base must not reach image reading")

    monkeypatch.setattr(ImageResourceLoader, "load_image_data", record_read)
    with pytest.warns(UserWarning, match="source document|Windows UNC"):
        doc = convert_figure(
            '<graphic xlink:href="figure.png"/>',
            JatsBackendOptions(
                fetch_images=True, enable_local_fetch=True, source_uri=source
            ),
        )
    assert assert_structure(doc).image is None
    assert reads == []


def test_disabled_images_do_not_resolve_or_probe(monkeypatch):
    calls = []

    def record_call(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Disabled image loading must not resolve or probe")

    monkeypatch.setattr(ImageResourceLoader, "normalize_local_base_path", record_call)
    monkeypatch.setattr(ImageResourceLoader, "load_image_data", record_call)
    monkeypatch.setattr(Path, "is_file", record_call)
    doc = convert_figure(
        '<graphic xlink:href="figure.png"/>', JatsBackendOptions(source_uri="file://")
    )
    assert assert_structure(doc).image is None
    assert calls == []


def test_local_permission_prevents_image_bytes(tmp_path: Path, monkeypatch):
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (7, 5)).save(image_path)
    reads = []
    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and Path(file) == image_path:
            reads.append(file)
            raise AssertionError("Local permission must prevent image byte reads")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    with pytest.warns(UserWarning, match="enable_local_fetch"):
        doc = convert_figure(
            '<graphic xlink:href="figure.png"/>',
            JatsBackendOptions(
                fetch_images=True, source_uri=(tmp_path / "article.nxml").as_uri()
            ),
        )
    assert assert_structure(doc).image is None
    assert reads == []


@pytest.mark.parametrize(
    "source_kind", ["path", "file", "remote", "ftp", "other", "none"]
)
def test_jats_never_fetches_remote_images(
    tmp_path: Path, source_kind: str, monkeypatch
):
    calls = []

    def record_fetch(*args, **kwargs):
        calls.append(args)
        raise AssertionError("JATS must not fetch remote images")

    monkeypatch.setattr(ImageResourceLoader, "fetch_remote", record_fetch)
    sources = {
        "path": tmp_path / "article.nxml",
        "file": (tmp_path / "article.nxml").as_uri(),
        "remote": "https://example.com/article.nxml",
        "ftp": "ftp://example.com/article.nxml",
        "other": "custom://example.com/article.nxml",
        "none": None,
    }
    doc = convert_figure(
        '<graphic xlink:href="https://example.com/figure.png"/>'
        '<graphic xlink:href="figure.png"/>'
        if source_kind not in {"path", "file"}
        else '<graphic xlink:href="https://example.com/figure.png"/>',
        JatsBackendOptions(
            fetch_images=True, enable_remote_fetch=True, source_uri=sources[source_kind]
        ),
    )
    assert assert_structure(doc).image is None
    assert calls == []


@pytest.mark.skipif(sys.platform == "win32", reason="Windows supports UNC authorities")
def test_jats_rejects_non_local_authority(monkeypatch):
    calls = []

    def record_read(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Unsupported authority must not read images")

    monkeypatch.setattr(ImageResourceLoader, "load_image_data", record_read)
    with pytest.warns(UserWarning, match="Windows UNC"):
        doc = convert_figure(
            '<graphic xlink:href="figure.png"/>',
            JatsBackendOptions(
                fetch_images=True,
                enable_local_fetch=True,
                source_uri="file://server/share/article.nxml",
            ),
        )
    assert assert_structure(doc).image is None
    assert calls == []
