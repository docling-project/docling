import sys
from pathlib import Path
import re

# Add the root directory to the system path
sys.path.insert(0, str(Path(__file__).parent.parent))

from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc.document import ContentLayer
from docling.datamodel.document import InputDocument, DoclingDocument


def test_is_hidden_element():
    """Test the is_hidden_element method directly."""
    # Create a minimal instance of HTMLDocumentBackend
    in_path = Path("tests/data/html/wiki_duck.html")
    in_doc = InputDocument(
        path_or_stream=in_path,
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
    )
    backend = HTMLDocumentBackend(
        in_doc=in_doc,
        path_or_stream=in_path,
    )

    # Test with different types of hidden elements
    from bs4 import BeautifulSoup, Tag

    # Hidden by class
    tag = BeautifulSoup('<div class="hidden">Test</div>', "html.parser").div
    assert backend.is_hidden_element(tag) == True

    # Hidden by d-none class (Bootstrap)
    tag = BeautifulSoup('<div class="d-none">Test</div>', "html.parser").div
    assert backend.is_hidden_element(tag) == True

    # Hidden by style
    tag = BeautifulSoup('<div style="display:none">Test</div>', "html.parser").div
    assert backend.is_hidden_element(tag) == True

    # Hidden by attribute
    tag = BeautifulSoup("<div hidden>Test</div>", "html.parser").div
    assert backend.is_hidden_element(tag) == True

    # Not hidden
    tag = BeautifulSoup("<div>Test</div>", "html.parser").div
    assert backend.is_hidden_element(tag) == False

    print("All is_hidden_element tests passed!")


def test_panel_title_extraction():
    """Test the handle_panel_title method directly."""
    # Create a minimal instance of HTMLDocumentBackend
    in_path = Path("tests/data/html/wiki_duck.html")
    in_doc = InputDocument(
        path_or_stream=in_path,
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
    )
    backend = HTMLDocumentBackend(
        in_doc=in_doc,
        path_or_stream=in_path,
    )

    # Initialize necessary attributes
    backend.content_layer = ContentLayer.BODY

    # Create a mock document
    doc = DoclingDocument(name="test")

    # Create a BeautifulSoup tag for a panel title
    from bs4 import BeautifulSoup

    html = """
    <div class="panel-title">
        <a class="collapsed" role="button">How can I get a digitally signed bank statement?</a>
    </div>
    """

    panel_title = BeautifulSoup(html, "html.parser").div

    # Set the parent level
    backend.level = 0
    backend.parents = {0: None}

    # Call the method
    backend.handle_panel_title(panel_title, doc)

    # Check if something was added to the document
    assert len(doc.body.children) == 1

    # Export to markdown to check the content
    markdown_content = doc.export_to_markdown()
    assert "How can I get a digitally signed bank statement?" in markdown_content

    print("Panel title extraction test passed!")


if __name__ == "__main__":
    test_is_hidden_element()
    test_panel_title_extraction()
    print("All tests passed successfully!")
