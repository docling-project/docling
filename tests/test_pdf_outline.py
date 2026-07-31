from docling.utils.pdf_outline import extract_outline_from_docling_parse


class _MockTocNode:
    """Duck-typed stand-in for docling_parse's PdfTableOfContents node.

    extract_outline_from_docling_parse only accesses .children, .text, and
    .orig on each node, so a lightweight mock is sufficient and avoids a
    dependency on constructing a real PDF with an outline.
    """

    def __init__(self, text="", children=None):
        self.text = text
        self.orig = text
        self.children = children or []


class _MockPdfDocument:
    """Duck-typed stand-in for docling_parse's PdfDocument, exposing only
    the one method extract_outline_from_docling_parse calls."""

    def __init__(self, toc_root):
        self._toc_root = toc_root

    def get_table_of_contents(self):
        return self._toc_root


def _build_chain(depth: int) -> _MockTocNode:
    """Build a linear chain of nested nodes depth levels deep:
    root -> child -> child -> ... (depth - 1 named children below root)."""
    root = _MockTocNode("level_0")
    current = root
    for i in range(1, depth):
        child = _MockTocNode(f"level_{i}")
        current.children = [child]
        current = child
    return root


def test_outline_no_toc_returns_empty_list():
    class _NoTocDoc:
        def get_table_of_contents(self):
            return None

    assert extract_outline_from_docling_parse(_NoTocDoc()) == []


def test_outline_flat_structure():
    root = _MockTocNode(
        "root",
        children=[_MockTocNode("First"), _MockTocNode("Second"), _MockTocNode("Third")],
    )
    items = extract_outline_from_docling_parse(_MockPdfDocument(root))
    assert [(item.title, item.level) for item in items] == [
        ("First", 0),
        ("Second", 0),
        ("Third", 0),
    ]


def test_outline_nested_structure_preserves_order_and_levels():
    root = _MockTocNode(
        "root",
        children=[
            _MockTocNode(
                "Chapter 1",
                children=[_MockTocNode("1.1"), _MockTocNode("1.2")],
            ),
            _MockTocNode("Chapter 2"),
        ],
    )
    items = extract_outline_from_docling_parse(_MockPdfDocument(root))
    assert [(item.title, item.level) for item in items] == [
        ("Chapter 1", 0),
        ("1.1", 1),
        ("1.2", 1),
        ("Chapter 2", 0),
    ]


def test_outline_blank_and_whitespace_titles_are_excluded():
    root = _MockTocNode(
        "root",
        children=[
            _MockTocNode(""),
            _MockTocNode("   "),
            _MockTocNode("  Real Title  "),
        ],
    )
    items = extract_outline_from_docling_parse(_MockPdfDocument(root))
    assert [(item.title, item.level) for item in items] == [("Real Title", 0)]


def test_outline_deep_chain_does_not_raise_recursion_error():
    """Regression test: a naive recursive walk over the outline tree raises
    RecursionError once the tree is deeper than Python's call-stack limit
    (default 1000). Large real-world documents can legitimately have this
    many nested heading levels. Use a depth well past the default limit to
    make sure this is actually exercised regardless of interpreter
    settings."""
    depth = 5000
    root = _build_chain(depth)

    items = extract_outline_from_docling_parse(_MockPdfDocument(root))

    assert len(items) == depth - 1
    assert items[0].title == "level_1"
    assert items[0].level == 0
    assert items[-1].title == f"level_{depth - 1}"
    assert items[-1].level == depth - 2
