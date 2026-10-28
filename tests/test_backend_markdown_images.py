# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument, ImageRefMode
from PIL import Image

from docling.datamodel.backend_options import MarkdownBackendOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.document_converter import DocumentConverter, MarkdownFormatOption
from tests.test_backend_markdown import _convert_markdown, _png_data_uri

pytestmark = pytest.mark.cross_platform


@pytest.mark.parametrize(
    "html_block",
    ["", "<h5>HTML heading</h5>", "<table><tr><td>HTML cell</td></tr></table>"],
    ids=["plain", "heading", "table"],
)
@pytest.mark.parametrize("fetch_images", [False, True])
@pytest.mark.parametrize("caption", ["", "First picture"])
def test_mixed_html_keeps_markdown_pictures(
    html_block: str, fetch_images: bool, caption: str
) -> None:
    title = f' "{caption}"' if caption else ""
    markdown = (
        f"# Title\n\n{html_block}\n\nBefore picture.\n\n"
        f"![alt]({_png_data_uri(7, 5)}{title})\n\nBetween pictures.\n\n"
        f'![alt]({_png_data_uri(3, 2)} "Second picture")\n\nAfter picture.\n'
    )
    doc = _convert_markdown(markdown, MarkdownBackendOptions(fetch_images=fetch_images))

    assert len(doc.pictures) == 2
    assert [picture.caption_text(doc) for picture in doc.pictures] == [
        caption,
        "Second picture",
    ]
    for picture, expected_size in zip(doc.pictures, [(7, 5), (3, 2)], strict=True):
        assert picture.parent is not None
        assert picture.get_ref() in picture.parent.resolve(doc).children
        image = picture.get_image(doc)
        if fetch_images:
            assert image is not None
            assert image.size == expected_size
            assert image.getpixel((0, 0)) == (255, 0, 0)
        else:
            assert picture.image is None
            assert image is None

    ordered = [item.self_ref for item, _ in doc.iterate_items()]
    before = next(text for text in doc.texts if text.text == "Before picture.")
    between = next(text for text in doc.texts if text.text == "Between pictures.")
    after = next(text for text in doc.texts if text.text == "After picture.")
    assert (
        ordered.index(before.self_ref)
        < ordered.index(doc.pictures[0].self_ref)
        < ordered.index(between.self_ref)
        < ordered.index(doc.pictures[1].self_ref)
        < ordered.index(after.self_ref)
    )
    restored = DoclingDocument.model_validate(doc.export_to_dict())
    assert len(restored.pictures) == 2
    assert [picture.image is not None for picture in restored.pictures] == [
        fetch_images,
        fetch_images,
    ]
    exported = doc.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
    assert exported.count("data:image/png;base64,") == (2 if fetch_images else 0)
    assert "After picture." in exported
    assert "application/x-docling-md-picture" not in doc.model_dump_json()
    assert "<img" not in doc.export_to_html()


@pytest.mark.parametrize("html_block", ["", "<h5>HTML heading</h5>"])
def test_mixed_html_keeps_size_rejected_image_placeholder(html_block: str) -> None:
    markdown = f'{html_block}\n\n![alt]({_png_data_uri(7, 5)} "caption")\n'
    with pytest.warns(UserWarning, match="exceeds size limit"):
        doc = _convert_markdown(
            markdown,
            MarkdownBackendOptions(fetch_images=True, max_image_data_base64_bytes=8),
        )
    assert len(doc.pictures) == 1
    assert doc.pictures[0].image is None
    assert doc.pictures[0].caption_text(doc) == "caption"


def test_image_over_pixel_limit_is_skipped() -> None:
    markdown = (
        f"Before.\n\n![big]({_png_data_uri(8, 8)})\n\n"
        f"![small]({_png_data_uri(7, 5)})\n\nAfter.\n"
    )
    with pytest.warns(UserWarning, match="exceeds the limit of 63 pixels"):
        doc = _convert_markdown(
            markdown, MarkdownBackendOptions(fetch_images=True, max_image_pixels=63)
        )

    assert len(doc.pictures) == 2
    assert doc.pictures[0].image is None
    assert doc.pictures[1].get_image(doc).size == (7, 5)
    assert [text.text for text in doc.texts] == ["Before.", "big", "small", "After."]


@pytest.mark.parametrize("fetch_images", [False, True])
def test_original_html_images_keep_the_normal_loader(fetch_images: bool) -> None:
    markdown = (
        f'<div><img src="{_png_data_uri(4, 3)}"></div>\n\n'
        f'![alt]({_png_data_uri(7, 5)} "Markdown picture")\n'
    )
    doc = _convert_markdown(markdown, MarkdownBackendOptions(fetch_images=fetch_images))
    assert len(doc.pictures) == 2
    assert doc.pictures[1].caption_text(doc) == "Markdown picture"
    if fetch_images:
        assert [picture.get_image(doc).size for picture in doc.pictures] == [
            (4, 3),
            (7, 5),
        ]
    else:
        assert all(picture.image is None for picture in doc.pictures)


@pytest.mark.parametrize("html_block", ["", "<h5>HTML heading</h5>"])
@pytest.mark.parametrize(
    "resource", ["missing", "local-denied", "remote-denied", "bad-data"]
)
def test_rejected_resources_stay_placeholders(
    tmp_path: Path, html_block: str, resource: str
) -> None:
    source = "missing.png"
    local_fetch = True
    if resource == "local-denied":
        Image.new("RGB", (7, 5), color="red").save(tmp_path / "local.png")
        source = "local.png"
        local_fetch = False
    elif resource == "remote-denied":
        source = "https://example.com/picture.png"
    elif resource == "bad-data":
        source = "data:image/png;base64,aW52YWxpZA=="
    with pytest.warns(UserWarning, match="Could not process an image"):
        doc = _convert_markdown(
            f'{html_block}\n\n![alt]({source} "caption")\n\nAfter picture.\n',
            MarkdownBackendOptions(
                fetch_images=True,
                enable_local_fetch=local_fetch,
                source_uri=tmp_path / "source.md",
            ),
        )
    assert len(doc.pictures) == 1
    assert doc.pictures[0].image is None
    assert doc.pictures[0].caption_text(doc) == "caption"
    assert "After picture." in doc.export_to_markdown()


def test_converter_keeps_an_image_inside_a_markdown_table_cell() -> None:
    markdown = (
        "<h5>HTML heading</h5>\n\n| Picture |\n|---|\n"
        f'| ![alt]({_png_data_uri(7, 5)} "cell picture") |\n'
    )
    converter = DocumentConverter(
        allowed_formats=[InputFormat.MD],
        format_options={
            InputFormat.MD: MarkdownFormatOption(
                backend_options=MarkdownBackendOptions(fetch_images=True)
            )
        },
    )
    result = converter.convert_string(markdown, InputFormat.MD)
    assert result.status == ConversionStatus.SUCCESS
    doc = result.document
    assert len(doc.tables) == 1
    assert len(doc.pictures) == 1
    picture = doc.pictures[0]
    assert picture.parent is not None
    assert picture.get_ref() in picture.parent.resolve(doc).children
    assert picture.caption_text(doc) == "cell picture"
    assert "cell picture" in doc.tables[0].export_to_markdown(doc)
    assert "data:image/png;base64," in doc.export_to_markdown(
        image_mode=ImageRefMode.EMBEDDED
    )
