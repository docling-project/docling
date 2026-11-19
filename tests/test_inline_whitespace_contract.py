# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Backend-side checks for the InlineGroup whitespace contract.

Inline runs carry their own significant whitespace and the docling-core serializers
concatenate them faithfully, with no separator of their own. The per-format boundary cases
live next to each backend's own tests; this module holds the cases that need to build a
source document on the fly.
"""

import re
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument, GroupLabel
from docx import Document

from docling.backend.asciidoc_backend import AsciiDocBackend
from docling.backend.html_backend import HTMLDocumentBackend
from docling.backend.msword_backend import MsWordDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument, TextItem
from docling.utils._inline_runs import trim_run_edges

from .test_backend_jats import convert_jats_body


def test_docx_paragraph_runs_keep_their_own_whitespace(tmp_path):
    """A DOCX run's boundary whitespace stays in the run.

    In DOCX the space at a formatting boundary very often sits *inside* the formatted run.
    Runs are emitted verbatim, with only the paragraph edges trimmed, and the serializer
    hoists edge whitespace out of the emphasis markers (``**bold** .`` and ``H 2 O`` must
    not appear).
    """
    docx_path = tmp_path / "runs.docx"
    document = Document()
    paragraph = document.add_paragraph()
    paragraph.add_run("   Normal ")  # leading paragraph padding is trimmed
    bold = paragraph.add_run("bold ")  # the boundary space lives inside the bold run
    bold.bold = True
    paragraph.add_run("tail.   ")  # trailing paragraph padding is trimmed

    no_space = document.add_paragraph()
    no_space.add_run("H")
    sub = no_space.add_run("2")
    sub.font.subscript = True
    no_space.add_run("O")

    document.save(docx_path)

    in_doc = InputDocument(
        path_or_stream=docx_path,
        format=InputFormat.DOCX,
        backend=MsWordDocumentBackend,
    )
    doc = MsWordDocumentBackend(in_doc=in_doc, path_or_stream=docx_path).convert()

    runs = [item.text for item, _ in doc.iterate_items() if isinstance(item, TextItem)]
    assert runs == ["Normal ", "bold ", "tail.", "H", "2", "O"]

    md = doc.export_to_markdown()
    assert "Normal **bold** tail." in md
    assert "H2O" in md


def test_docx_rich_table_cell_keeps_its_inline_group(tmp_path):
    """Runs in a rich cell stay one inline flow, not blocks with their own separator."""
    docx_path = tmp_path / "cell.docx"
    document = Document()
    paragraph = document.add_table(rows=1, cols=2).cell(0, 0).paragraphs[0]
    paragraph.add_run("Water, ")
    paragraph.add_run("bold").bold = True
    paragraph.add_run(", H")
    paragraph.add_run("2").font.subscript = True
    paragraph.add_run("O.")
    document.save(docx_path)

    in_doc = InputDocument(
        path_or_stream=docx_path,
        format=InputFormat.DOCX,
        backend=MsWordDocumentBackend,
    )
    doc = MsWordDocumentBackend(in_doc=in_doc, path_or_stream=docx_path).convert()

    assert all(
        item.parent.resolve(doc).label == GroupLabel.INLINE for item in doc.texts
    )
    assert "| Water, **bold**, H2O." in doc.export_to_markdown()


def test_docx_inline_equations_keep_the_boundary_space():
    """Text after an inline equation keeps its leading space.

    ``_add_inline_equations_to_parent`` builds its own InlineGroup, so the fragment after
    the last equation must keep its leading space (not ``$A= \\pi r^{2}$is the area formula``).
    """
    source = Path("tests/data/docx/sources/equations.docx")
    in_doc = InputDocument(
        path_or_stream=source,
        format=InputFormat.DOCX,
        backend=MsWordDocumentBackend,
    )
    doc = MsWordDocumentBackend(in_doc=in_doc, path_or_stream=source).convert()
    md = doc.export_to_markdown()

    # list items (the helper's own InlineGroup) ...
    assert (
        "- First item with inline equation: $A= \\pi r^{2}$ is the area formula." in md
    )
    assert (
        "- Second item with equations: $E=mc^{2}$ and $F=ma$ are physics formulas."
        in md
    )
    assert "- The formula $a^{2}+b^{2}=c^{2}$ is the Pythagorean theorem." in md
    # ... and a normal paragraph through the same helper, where the boundary is a period.
    assert (
        "This is a word document and this is an inline equation: $A= \\pi r^{2}$." in md
    )

    # No equation is glued to the following word anywhere in the document.
    assert not re.search(r"\}\$[A-Za-z]", md), md


def _convert_html(tmp_path, body: str):
    source = tmp_path / "t.html"
    source.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")
    in_doc = InputDocument(
        path_or_stream=source,
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
    )
    return HTMLDocumentBackend(in_doc=in_doc, path_or_stream=source).convert()


@pytest.mark.parametrize(
    ("body", "expected_md", "expected_txt"),
    [
        # Adjacent code spans: the boundary belongs between the two `` ` `` delimiters, not
        # inside either of them (not ``a`` b``).
        ("<p><code>a</code> <code>b</code></p>", "`a` `b`", "a b"),
        ("<p><code>a</code> <code>b</code> <code>c</code></p>", "`a` `b` `c`", "a b c"),
        # No source whitespace means no boundary to preserve.
        ("<p>x<code>a</code><code>b</code>y</p>", "x`ab`y", "xaby"),
        # One side plain: the space goes on the plain run.
        ("<p><code>a</code> <b>b</b></p>", "`a` **b**", "a b"),
        ("<p><b>a</b> <code>b</code></p>", "**a** `b`", "a b"),
        ("<p>see <code>a</code> then</p>", "see `a` then", "see a then"),
        # The original contract cases stay correct.
        (
            "<p>Water is H<sub>2</sub>O here</p>",
            "Water is H2O here",
            "Water is H2O here",
        ),
        (
            "<p>x<sup>2</sup> + y<sup>3</sup> uses sup</p>",
            "x2 + y3 uses sup",
            "x2 + y3 uses sup",
        ),
    ],
)
def test_html_boundary_stays_outside_code_delimiters(
    tmp_path, body, expected_md, expected_txt
):
    doc = _convert_html(tmp_path, body)
    assert doc.export_to_markdown() == expected_md
    assert doc.export_to_text() == expected_txt


def test_html_pre_keeps_interior_whitespace(tmp_path):
    """`<pre>` is preformatted: only the block edges are trimmed, never each fragment.

    The one-code-item-per-fragment shape of `<pre>` with inline markup is a separate
    pre-existing issue and is not addressed here.
    """
    doc = _convert_html(tmp_path, '<pre>See <a href="x">link</a> first</pre>')
    assert doc.export_to_text() == "See link first"

    plain = _convert_html(tmp_path, "<pre>\ndef f():\n    return 1\n</pre>")
    assert "def f():\n    return 1" in plain.export_to_markdown()


def test_jats_inline_runs_rejoin_to_the_source_text():
    """Runs concatenate back to the paragraph text: no invented and no lost separator."""
    cases = {
        "The species <italic>Homo sapiens</italic> is <bold>common</bold>.": (
            "The species Homo sapiens is common."
        ),
        # no whitespace at the boundary in the source, so none in the runs
        "Water is H<sub>2</sub>O here.": "Water is H2O here.",
        "Note<sup>1</sup> follows.": "Note1 follows.",
        # block edges are trimmed exactly once; interior whitespace is verbatim, since
        # XML content is not whitespace-collapsed the way HTML normal flow is
        "  padded  <italic>x</italic>  ": "padded  x",
    }
    for paragraph, expected in cases.items():
        doc = convert_jats_body(f"<sec><title>T</title><p>{paragraph}</p></sec>")
        runs = [
            t.text for t in doc.texts if t.label == DocItemLabel.TEXT and t.text != "T"
        ]
        assert "".join(runs) == expected, paragraph


def test_asciidoc_code_run_edge_whitespace_stays_outside_the_code():
    doc = DoclingDocument(name="code-edges")
    AsciiDocBackend._flush_text_data(
        doc=doc,
        text_data=["Install with `pip install x` first, x`` a ``y."],
        parent=None,
    )

    assert [item.text for item in doc.texts] == [
        "Install with ",
        "pip install x",
        " first, x",
        " ",
        "a",
        " ",
        "y.",
    ]
    assert doc.export_to_markdown() == "Install with `pip install x` first, x `a` y."


def test_asciidoc_code_run_keeps_the_source_whitespace_outside_the_code():
    doc = DoclingDocument(name="code-edges")
    AsciiDocBackend._flush_text_data(
        doc=doc, text_data=["x``  a  ``y and w`` ``z."], parent=None
    )

    assert [(item.text, item.label) for item in doc.texts] == [
        ("x", DocItemLabel.TEXT),
        ("  ", DocItemLabel.TEXT),
        ("a", DocItemLabel.CODE),
        ("  ", DocItemLabel.TEXT),
        ("y and w", DocItemLabel.TEXT),
        (" ", DocItemLabel.TEXT),
        ("z.", DocItemLabel.TEXT),
    ]


@pytest.mark.parametrize(
    "source, expected_runs, expected_markdown",
    [
        (
            "a https://x.y[ pad ]b end.",
            ["a ", " pad ", "b end."],
            "a  [pad](https://x.y/) b end.",
        ),
        (
            "a https://x.y[pad ^]b end.",
            ["a ", "pad ", "b end."],
            "a [pad](https://x.y/) b end.",
        ),
        # `^` is the new-window marker only as the final character
        (
            "a https://x.y[pad^ ]b end.",
            ["a ", "pad^ ", "b end."],
            "a [pad^](https://x.y/) b end.",
        ),
        # blank brackets show the target and keep their whitespace
        (
            "a https://x.y[ ]b end.",
            ["a ", "https://x.y", " ", "b end."],
            "a [https://x.y](https://x.y/) b end.",
        ),
        # only the block's outer edge is trimmed
        (
            "https://x.y[ pad ] end.",
            ["pad ", " end."],
            "[pad](https://x.y/)  end.",
        ),
        (
            "a https://x.y[ *Bold* ]b end.",
            ["a ", " ", "Bold", " ", "b end."],
            "a  [**Bold**](https://x.y/) b end.",
        ),
    ],
)
def test_asciidoc_link_text_keeps_its_edge_whitespace(
    source: str, expected_runs: list[str], expected_markdown: str
):
    doc = DoclingDocument(name="link-edges")
    AsciiDocBackend._flush_text_data(doc=doc, text_data=[source], parent=None)

    assert [item.text for item in doc.texts] == expected_runs
    assert doc.export_to_markdown() == expected_markdown


@dataclass(frozen=True)
class _Run:
    text: str
    fixed: bool = False


def _trim(runs: list[_Run]) -> list[_Run]:
    return trim_run_edges(
        runs,
        get_text=lambda run: run.text,
        with_text=lambda run, text: replace(run, text=text),
        is_trimmable=lambda run: not run.fixed,
    )


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        # Only the outer edges are trimmed; the boundary whitespace between runs stays.
        (["  a ", " b  "], ["a ", " b"]),
        (["  a  "], ["a"]),
        # Blank edge runs are dropped, a blank interior run is a boundary and stays.
        (["  ", "a", " ", "b", "\n"], ["a", " ", "b"]),
        (["", "a", "", "b"], ["a", "b"]),
        (["  ", " "], []),
        ([], []),
    ],
)
def test_trim_run_edges_keeps_the_boundary_between_runs(
    texts: list[str], expected: list[str]
):
    runs = [_Run(text) for text in texts]

    assert [run.text for run in _trim(runs)] == expected
    assert [run.text for run in runs] == texts


def test_trim_run_edges_stops_at_a_run_that_is_not_trimmable():
    runs = [_Run("  "), _Run("x", fixed=True), _Run(" a "), _Run("y", fixed=True)]

    assert _trim(runs) == [_Run("x", fixed=True), _Run(" a "), _Run("y", fixed=True)]
