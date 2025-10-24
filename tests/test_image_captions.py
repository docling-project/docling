"""Tests for image caption functionality in markdown files."""

import pytest
from io import StringIO
from pathlib import Path

from docling_core.types.doc import DocItemLabel
from docling.datamodel.document import InputDocument, PictureItem, TextItem
from docling.datamodel.base_models import InputFormat
from docling.backend.md_backend import MarkdownDocumentBackend


import tempfile
import os


def create_backend_from_content(md_content: str) -> MarkdownDocumentBackend:
    """Helper function to create a markdown backend from string content."""
    # Create a temporary file with the markdown content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp_file:
        tmp_file.write(md_content)
        tmp_path = tmp_file.name
    
    try:
        # Convert to Path object
        md_path = Path(tmp_path)
        
        in_doc = InputDocument(
            path_or_stream=md_path,
            format=InputFormat.MD,
            backend=MarkdownDocumentBackend,
        )
        backend = MarkdownDocumentBackend(in_doc=in_doc, path_or_stream=md_path)
        return backend
    finally:
        # Clean up the temporary file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass  # File already deleted


def test_inline_image_captions():
    """Test that inline image captions are detected and processed correctly."""
    md_content = '''# Test Document

![Test Image](./test.png) <span data-class="image-caption">This is an inline caption</span>

Some other text.

![Another Image](./another.png) <span data-class="image-caption">Another inline caption</span>
'''
    
    backend = create_backend_from_content(md_content)
    doc = backend.convert()
    
    # Find all pictures
    pictures = []
    for item, _ in doc.iterate_items():
        if isinstance(item, PictureItem):
            pictures.append(item)
    
    # Verify we have 2 pictures
    assert len(pictures) == 2, f"Expected 2 pictures, found {len(pictures)}"
    
    # Verify both pictures have captions
    first_caption = pictures[0].captions[0].resolve(doc)
    assert first_caption.text == "This is an inline caption"
    assert first_caption.label == DocItemLabel.CAPTION
    
    second_caption = pictures[1].captions[0].resolve(doc)
    assert second_caption.text == "Another inline caption"
    assert second_caption.label == DocItemLabel.CAPTION


def test_block_level_image_captions():
    """Test that block-level image captions (next line or with whitespace) are detected."""
    md_content = '''# Test Document

![Image 1](./image1.png)
<span data-class="image-caption">Next line caption</span>

![Image 2](./image2.png)

<span data-class="image-caption">Caption after whitespace</span>
'''
    
    backend = create_backend_from_content(md_content)
    doc = backend.convert()
    
    # Find all pictures
    pictures = []
    for item, _ in doc.iterate_items():
        if isinstance(item, PictureItem):
            pictures.append(item)
    
    # Verify we have 2 pictures
    assert len(pictures) == 2, f"Expected 2 pictures, found {len(pictures)}"
    
    # Verify both pictures have captions
    first_caption = pictures[0].captions[0].resolve(doc)
    assert first_caption.text == "Next line caption"
    
    second_caption = pictures[1].captions[0].resolve(doc)
    assert second_caption.text == "Caption after whitespace"


def test_mixed_caption_styles():
    """Test document with mixed inline and block-level captions."""
    md_content = '''# Test Document

## Inline
![Inline](./inline.png) <span data-class="image-caption">Inline caption</span>

## Next line
![Next line](./nextline.png)
<span data-class="image-caption">Next line caption</span>

## With whitespace
![Whitespace](./whitespace.png)

<span data-class="image-caption">Caption after whitespace</span>
'''
    
    backend = create_backend_from_content(md_content)
    doc = backend.convert()
    
    # Find all pictures
    pictures = []
    for item, _ in doc.iterate_items():
        if isinstance(item, PictureItem):
            pictures.append(item)
    
    # Verify we have 3 pictures
    assert len(pictures) == 3, f"Expected 3 pictures, found {len(pictures)}"
    
    # Verify all pictures have captions
    expected_captions = ["Inline caption", "Next line caption", "Caption after whitespace"]
    for i, picture in enumerate(pictures):
        assert picture.captions, f"Picture {i} should have captions"
        caption = picture.captions[0].resolve(doc)
        assert caption.text == expected_captions[i], f"Picture {i}: Expected '{expected_captions[i]}', got '{caption.text}'"


def test_images_without_captions():
    """Test that images without captions work as before."""
    md_content = '''# Test Document

![Image 1](./image1.png)

Some text.

![Image 2](./image2.png)

More text.
'''
    
    backend = create_backend_from_content(md_content)
    doc = backend.convert()
    
    # Find all pictures
    pictures = []
    for item, _ in doc.iterate_items():
        if isinstance(item, PictureItem):
            pictures.append(item)
    
    # Verify we have 2 pictures
    assert len(pictures) == 2, f"Expected 2 pictures, found {len(pictures)}"
    
    # Verify pictures don't have captions
    for i, picture in enumerate(pictures):
        assert not picture.captions, f"Picture {i} should not have captions"


def test_images_with_title_attribute():
    """Test that title attribute captions still work (existing behavior)."""
    md_content = '''# Test Document

![Image with title](./image.png "This is a title attribute")
'''
    
    backend = create_backend_from_content(md_content)
    doc = backend.convert()
    
    # Find all pictures
    pictures = []
    for item, _ in doc.iterate_items():
        if isinstance(item, PictureItem):
            pictures.append(item)
    
    # Verify we have 1 picture
    assert len(pictures) == 1, f"Expected 1 picture, found {len(pictures)}"
    
    # Verify picture has caption from title attribute
    picture = pictures[0]
    assert picture.captions, "Picture should have captions"
    caption = picture.captions[0].resolve(doc)
    assert caption.text == "This is a title attribute", f"Expected title attribute caption, got '{caption.text}'"


def test_caption_class_variations():
    """Test that both data-class and class attributes work for captions."""
    md_content = '''# Test Document

![Image 1](./image1.png) <span data-class="image-caption">Data class caption</span>

![Image 2](./image2.png) <span class="image-caption">Class caption</span>
'''
    
    backend = create_backend_from_content(md_content)
    doc = backend.convert()
    
    # Find all pictures
    pictures = []
    for item, _ in doc.iterate_items():
        if isinstance(item, PictureItem):
            pictures.append(item)
    
    # Verify we have 2 pictures
    assert len(pictures) == 2, f"Expected 2 pictures, found {len(pictures)}"
    
    # Verify both pictures have captions
    first_caption = pictures[0].captions[0].resolve(doc)
    assert first_caption.text == "Data class caption"
    
    second_caption = pictures[1].captions[0].resolve(doc)
    assert second_caption.text == "Class caption"


def test_caption_priority_title_vs_html():
    """Test that title attribute takes priority over HTML span captions."""
    md_content = '''# Test Document

![Image](./image.png "Title attribute") <span data-class="image-caption">HTML caption</span>
'''
    
    backend = create_backend_from_content(md_content)
    doc = backend.convert()
    
    # Find all pictures
    pictures = []
    for item, _ in doc.iterate_items():
        if isinstance(item, PictureItem):
            pictures.append(item)
    
    # Verify we have 1 picture
    assert len(pictures) == 1, f"Expected 1 picture, found {len(pictures)}"
    
    # Verify picture uses title attribute, not HTML caption
    picture = pictures[0]
    assert picture.captions, "Picture should have captions"
    caption = picture.captions[0].resolve(doc)
    assert caption.text == "Title attribute", f"Expected title attribute to take priority, got '{caption.text}'"


def test_multiple_images_single_paragraph():
    """Test handling of multiple images in the same paragraph with captions."""
    md_content = '''# Test Document

![Image 1](./image1.png) <span data-class="image-caption">Caption 1</span> and ![Image 2](./image2.png) <span data-class="image-caption">Caption 2</span>
'''
    
    backend = create_backend_from_content(md_content)
    doc = backend.convert()
    
    # Find all pictures
    pictures = []
    for item, _ in doc.iterate_items():
        if isinstance(item, PictureItem):
            pictures.append(item)
    
    # Verify we have 2 pictures
    assert len(pictures) == 2, f"Expected 2 pictures, found {len(pictures)}"
    
    # Verify both pictures have their respective captions
    first_caption = pictures[0].captions[0].resolve(doc)
    assert first_caption.text == "Caption 1"
    
    second_caption = pictures[1].captions[0].resolve(doc)
    assert second_caption.text == "Caption 2"


@pytest.mark.parametrize("test_file", [
    "image_caption_inline.md",
    "image_caption_block.md", 
    "image_no_caption.md"
])
def test_caption_test_files(test_file):
    """Test the caption functionality using the test markdown files."""
    test_path = Path("tests/data/md/image_captions") / test_file
    
    if not test_path.exists():
        pytest.skip(f"Test file {test_file} not found")
    
    in_doc = InputDocument(
        path_or_stream=test_path,
        format=InputFormat.MD,
        backend=MarkdownDocumentBackend,
    )
    backend = MarkdownDocumentBackend(in_doc=in_doc, path_or_stream=test_path)
    assert backend.is_valid()
    
    doc = backend.convert()
    assert doc is not None
    
    # Find all pictures
    pictures = []
    for item, _ in doc.iterate_items():
        if isinstance(item, PictureItem):
            pictures.append(item)
    
    # Basic validation that conversion worked
    assert len(pictures) > 0, f"Expected some pictures in {test_file}"
    
    # Verify caption behavior based on file type
    if "no_caption" in test_file:
        # Most images should not have captions (except the title attribute one)
        caption_count = sum(1 for pic in pictures if pic.captions)
        assert caption_count <= 1, f"Expected at most 1 captioned image in {test_file}"
    else:
        # All images should have captions
        for i, picture in enumerate(pictures):
            assert picture.captions, f"Picture {i} in {test_file} should have captions"
            caption = picture.captions[0].resolve(doc)
            assert isinstance(caption, TextItem), f"Caption {i} should be TextItem"
            assert caption.label == DocItemLabel.CAPTION, f"Caption {i} should have CAPTION label"
            assert caption.text.strip(), f"Caption {i} should have non-empty text"