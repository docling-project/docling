# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

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


class _MockPdfiumPage:
    """Stand-in for a pypdfium2 page: extract_outline_from_pdfium only reads the
    height and closes it."""

    def __init__(self, height: float):
        self._height = height
        self.closed = False

    def get_height(self) -> float:
        return self._height

    def close(self) -> None:
        self.closed = True


class _MockPdfiumDocument:
    """Stand-in for a pypdfium2 PdfDocument holding an outline and page heights."""

    def __init__(self, toc, page_height: float = 800.0):
        self._toc = toc
        self._page_height = page_height
        self.pages_opened: list[int] = []
        self.max_depth_used: int | None = None

    def get_toc(self, max_depth: int = 15):
        # Mirrors pypdfium2: entries below the depth bound are dropped silently.
        self.max_depth_used = max_depth
        return iter([bm for bm in self._toc if bm.level < max_depth])

    def __getitem__(self, index: int) -> _MockPdfiumPage:
        self.pages_opened.append(index)
        return _MockPdfiumPage(self._page_height)


class _Pypdfium4Bookmark:
    """pypdfium2 4.30's ``PdfOutlineItem``: a namedtuple carrying the destination
    fields directly, with no ``get_title``/``get_dest`` methods."""

    def __init__(self, title, level, page_index, view_mode, view_pos):
        self.title = title
        self.level = level
        self.page_index = page_index
        self.view_mode = view_mode
        self.view_pos = view_pos


class _Pypdfium5Dest:
    def __init__(self, index, view):
        self._index = index
        self._view = view

    def get_index(self):
        return self._index

    def get_view(self):
        return self._view


class _Pypdfium5Bookmark:
    """pypdfium2 5's bookmark object, which reaches the destination through
    accessor methods."""

    def __init__(self, title, level, dest):
        self._title = title
        self.level = level
        self._dest = dest

    def get_title(self):
        return self._title

    def get_dest(self):
        return self._dest


def _xyz_mode() -> int:
    import pypdfium2.raw as pdfium_c

    return pdfium_c.PDFDEST_VIEW_XYZ


def test_outline_pdfium_reads_the_pypdfium2_4x_namedtuple():
    """Regression test: on pypdfium2 4.x the bookmarks have no ``get_title()``,
    so reaching for it lost the whole outline on a supported version."""
    from docling.utils.pdf_outline import extract_outline_from_pdfium

    doc = _MockPdfiumDocument(
        [
            _Pypdfium4Bookmark("Contents", 0, 2, _xyz_mode(), [0.0, 736.0, 0.0]),
            _Pypdfium4Bookmark("Chapter 1", 1, 9, _xyz_mode(), [0.0, 700.0, 0.0]),
        ]
    )

    items = extract_outline_from_pdfium(doc)

    assert [(i.title, i.level, i.page_no, i.y_top) for i in items] == [
        ("Contents", 0, 3, 64.0),
        ("Chapter 1", 1, 10, 100.0),
    ]


def test_outline_pdfium_reads_the_pypdfium2_5x_bookmark_objects():
    """The 5.x shape must keep working: both majors are supported."""
    from docling.utils.pdf_outline import extract_outline_from_pdfium

    doc = _MockPdfiumDocument(
        [
            _Pypdfium5Bookmark(
                "Contents", 0, _Pypdfium5Dest(2, (_xyz_mode(), [0.0, 736.0, 0.0]))
            ),
        ]
    )

    items = extract_outline_from_pdfium(doc)

    assert [(i.title, i.level, i.page_no, i.y_top) for i in items] == [
        ("Contents", 0, 3, 64.0)
    ]


def test_outline_pdfium_entry_without_a_target_page():
    """An entry whose destination carries no page keeps ``page_no`` unset, and no
    page is opened to look up a height."""
    from docling.utils.pdf_outline import extract_outline_from_pdfium

    doc = _MockPdfiumDocument(
        [_Pypdfium4Bookmark("Dangling", 0, None, _xyz_mode(), [0.0, 736.0, 0.0])]
    )

    items = extract_outline_from_pdfium(doc)

    assert [(i.title, i.page_no, i.y_top) for i in items] == [("Dangling", None, None)]
    assert doc.pages_opened == []


def test_outline_pdfium_view_mode_without_a_vertical_position():
    """FIT carries no top coordinate, so the page survives but ``y_top`` does not."""
    import pypdfium2.raw as pdfium_c

    from docling.utils.pdf_outline import extract_outline_from_pdfium

    doc = _MockPdfiumDocument(
        [_Pypdfium4Bookmark("Fitted", 0, 4, pdfium_c.PDFDEST_VIEW_FIT, [])]
    )

    items = extract_outline_from_pdfium(doc)

    assert [(i.title, i.page_no, i.y_top) for i in items] == [("Fitted", 5, None)]
    assert doc.pages_opened == []


def test_outline_pdfium_blank_titles_are_excluded():
    from docling.utils.pdf_outline import extract_outline_from_pdfium

    doc = _MockPdfiumDocument(
        [
            _Pypdfium4Bookmark("   ", 0, 1, _xyz_mode(), [0.0, 736.0, 0.0]),
            _Pypdfium4Bookmark("  Real  ", 0, 1, _xyz_mode(), [0.0, 736.0, 0.0]),
        ]
    )

    items = extract_outline_from_pdfium(doc)

    assert [(i.title, i.page_no) for i in items] == [("Real", 2)]


def test_outline_pdfium_reuses_one_page_height_per_page():
    """The height lookup is cached: several bookmarks on a page open it once."""
    from docling.utils.pdf_outline import extract_outline_from_pdfium

    doc = _MockPdfiumDocument(
        [
            _Pypdfium4Bookmark("A", 0, 7, _xyz_mode(), [0.0, 700.0, 0.0]),
            _Pypdfium4Bookmark("B", 1, 7, _xyz_mode(), [0.0, 600.0, 0.0]),
        ]
    )

    items = extract_outline_from_pdfium(doc)

    assert [i.y_top for i in items] == [100.0, 200.0]
    assert doc.pages_opened == [7]


def _deep_outline_pdf(depth: int) -> bytes:
    """A one-page PDF whose outline is a single chain `depth` levels deep.

    Written out by hand rather than with a PDF library: nothing that can author an
    outline is a dependency of this project, and this test has to run everywhere.
    """
    catalog, pages, page, outlines = 1, 2, 3, 4
    first_item = outlines + 1
    objects: dict[int, str] = {
        catalog: f"<< /Type /Catalog /Pages {pages} 0 R /Outlines {outlines} 0 R >>",
        pages: f"<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>",
        page: f"<< /Type /Page /Parent {pages} 0 R /MediaBox [0 0 612 792] >>",
        outlines: (
            f"<< /Type /Outlines /First {first_item} 0 R "
            f"/Last {first_item} 0 R /Count {depth} >>"
        ),
    }
    for i in range(depth):
        num = first_item + i
        parent = outlines if i == 0 else num - 1
        entry = (
            f"<< /Title (level {i}) /Parent {parent} 0 R "
            f"/Dest [{page} 0 R /XYZ 0 {700 - i} 0]"
        )
        if i + 1 < depth:
            entry += f" /First {num + 1} 0 R /Last {num + 1} 0 R /Count 1"
        objects[num] = entry + " >>"

    out = bytearray(b"%PDF-1.7\n")
    offsets: dict[int, int] = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n{objects[num]}\nendobj\n".encode()
    xref_offset = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    for num in range(1, size):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {size} /Root {catalog} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()
    return bytes(out)


def test_outline_pdfium_reads_past_the_default_depth_limit():
    """Regression test: pypdfium2's `get_toc()` stops at 15 levels unless told
    otherwise, so a deeper outline came back silently truncated -- unlike the
    docling-parse walker, which is unbounded."""
    from io import BytesIO

    import pypdfium2 as pdfium

    from docling.utils.pdf_outline import extract_outline_from_pdfium

    depth = 20
    pdoc = pdfium.PdfDocument(BytesIO(_deep_outline_pdf(depth)))

    items = extract_outline_from_pdfium(pdoc)

    assert [item.title for item in items] == [f"level {i}" for i in range(depth)]
    assert [item.level for item in items] == list(range(depth))
    assert all(item.page_no == 1 for item in items)
