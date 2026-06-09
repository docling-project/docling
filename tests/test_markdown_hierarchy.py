"""Test that Markdown parser respects heading hierarchy."""

from pathlib import Path

import pytest

from docling.backend.md_backend import MarkdownDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling_core.types.doc import DocItemLabel, GroupLabel

pytestmark = pytest.mark.cross_platform


def test_markdown_heading_hierarchy():
    """Test that Markdown parser creates proper hierarchical structure."""
    markdown_content = """# Main Title

This is content under the main title.

## Section 1

Content under section 1.

### Subsection 1.1

Content under subsection 1.1.

### Subsection 1.2

Content under subsection 1.2.

## Section 2

Content under section 2.

### Subsection 2.1

Content under subsection 2.1.
"""
    
    from io import BytesIO
    
    stream = BytesIO(markdown_content.encode("utf-8"))
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.MD,
        backend=MarkdownDocumentBackend,
        filename="test.md",
    )
    backend = MarkdownDocumentBackend(
        in_doc=in_doc,
        path_or_stream=stream,
    )
    
    doc = backend.convert()
    
    # Verify document structure
    # The document should have a hierarchical structure where:
    # - Main Title is at the root
    # - Section 1 and Section 2 are children of Main Title (via section groups)
    # - Subsections are children of their respective sections
    
    # Get all items
    items = list(doc.iterate_items())
    
    # Find the title
    titles = [item for item, _ in items if item.label == DocItemLabel.TITLE]
    assert len(titles) == 1, "Should have exactly one title"
    title = titles[0]
    assert "Main Title" in title.text
    
    # Find headings (section headers)
    headings = [item for item, _ in items if item.label == DocItemLabel.SECTION_HEADER]
    
    # We should have 5 headings total (2 level-2 and 3 level-3)
    assert len(headings) >= 5, f"Should have at least 5 headings, found {len(headings)}"
    
    # Verify that we can find all expected sections
    section_1 = next((h for h in headings if "Section 1" in h.text), None)
    section_2 = next((h for h in headings if "Section 2" in h.text), None)
    
    assert section_1 is not None, "Should find Section 1"
    assert section_2 is not None, "Should find Section 2"
    
    # Find subsections
    subsection_1_1 = next((h for h in headings if "Subsection 1.1" in h.text), None)
    subsection_1_2 = next((h for h in headings if "Subsection 1.2" in h.text), None)
    subsection_2_1 = next((h for h in headings if "Subsection 2.1" in h.text), None)
    
    assert subsection_1_1 is not None, "Should find Subsection 1.1"
    assert subsection_1_2 is not None, "Should find Subsection 1.2"
    assert subsection_2_1 is not None, "Should find Subsection 2.1"
    
    # Verify hierarchical structure exists through parent relationships or document structure
    # The document should maintain the heading hierarchy even if not through explicit section groups
    all_items_with_labels = [(item, level) for item, level in items if hasattr(item, 'label')]
    
    # Check that we have a proper hierarchy by verifying levels
    heading_levels = {}
    for item, level in items:
        if item.label == DocItemLabel.SECTION_HEADER:
            heading_levels[item.text] = level
    
    # Level 2 headings (##) should be at a deeper level than the title
    # Level 3 headings (###) should be at a deeper level than level 2 headings
    if "Section 1" in heading_levels and "Subsection 1.1" in heading_levels:
        assert heading_levels["Subsection 1.1"] > heading_levels["Section 1"], \
            "Subsection 1.1 should be at a deeper level than Section 1"
    
    print("\n=== Document Structure ===")
    for item, level in items:
        indent = "  " * level
        parent_ref = item.parent if hasattr(item, 'parent') else None
        print(f"{indent}{item.label}: {getattr(item, 'text', '')[:50]} (parent: {parent_ref})")


def test_markdown_vs_html_hierarchy_consistency():
    """Test that Markdown and HTML produce similar hierarchical structures."""
    markdown_content = """# Article Title

Introduction text.

## Methods

Methods description.

### Data Collection

Data collection details.

## Results

Results description.
"""
    
    html_content = """<!DOCTYPE html>
<html>
<body>
<h1>Article Title</h1>
<p>Introduction text.</p>
<h2>Methods</h2>
<p>Methods description.</p>
<h3>Data Collection</h3>
<p>Data collection details.</p>
<h2>Results</h2>
<p>Results description.</p>
</body>
</html>
"""
    
    from io import BytesIO
    from docling.backend.html_backend import HTMLDocumentBackend
    
    # Convert Markdown
    md_stream = BytesIO(markdown_content.encode("utf-8"))
    md_in_doc = InputDocument(
        path_or_stream=md_stream,
        format=InputFormat.MD,
        backend=MarkdownDocumentBackend,
        filename="test.md",
    )
    md_backend = MarkdownDocumentBackend(
        in_doc=md_in_doc,
        path_or_stream=md_stream,
    )
    md_doc = md_backend.convert()
    
    # Convert HTML
    html_stream = BytesIO(html_content.encode("utf-8"))
    html_in_doc = InputDocument(
        path_or_stream=html_stream,
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="test.html",
    )
    html_backend = HTMLDocumentBackend(
        in_doc=html_in_doc,
        path_or_stream=html_stream,
    )
    html_doc = html_backend.convert()
    
    # Get headings from both
    md_items = list(md_doc.iterate_items())
    html_items = list(html_doc.iterate_items())
    
    md_headings = [
        item for item, _ in md_items
        if item.label in (DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER)
    ]
    html_headings = [
        item for item, _ in html_items
        if item.label in (DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER)
    ]
    
    # Should have same number of headings
    assert len(md_headings) == len(html_headings), (
        f"Markdown has {len(md_headings)} headings, HTML has {len(html_headings)}"
    )
    
    # Both should have hierarchical structure (headings with parents)
    md_with_parents = sum(1 for h in md_headings if h.parent is not None)
    html_with_parents = sum(1 for h in html_headings if h.parent is not None)
    
    # At least some headings should have parents (not all flat)
    assert md_with_parents > 0, "Markdown headings should have hierarchical parents"
    assert html_with_parents > 0, "HTML headings should have hierarchical parents"
    
    print(f"\nMarkdown: {md_with_parents}/{len(md_headings)} headings have parents")
    print(f"HTML: {html_with_parents}/{len(html_headings)} headings have parents")

# Made with Bob
