from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel, GroupLabel

from docling.backend.latex_backend import LatexDocumentBackend
from docling.datamodel.backend_options import LatexBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument, InputDocument
from docling.document_converter import DocumentConverter

from ..test_data_gen_flag import GEN_TEST_DATA
from ..verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA
LATEX_DATA_DIR = Path("./tests/data/latex/")


def test_latex_basic_conversion():
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\section{Introduction}
    Hello World.
    \\end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    assert len(doc.texts) > 0
    # Check structure
    headers = [t for t in doc.texts if t.label == DocItemLabel.SECTION_HEADER]
    paragraphs = [t for t in doc.texts if t.label != DocItemLabel.SECTION_HEADER]

    assert len(headers) == 1
    assert headers[0].text == "Introduction"
    assert "Hello World" in paragraphs[0].text


def test_latex_preamble_filter():
    latex_content = b"""
    \\documentclass{article}
    \\usepackage{test}
    \\title{Ignored Title}
    \\begin{document}
    Real Content
    \\end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    # Preamble metadata (\title, \author, \date) is now extracted
    # following pandoc's approach. Only package commands should be filtered.

    full_text = doc.export_to_markdown()
    assert "Real Content" in full_text
    assert "Ignored Title" in full_text
    assert "usepackage" not in full_text


def test_latex_escaped_chars():
    # Test correct handling of escaped chars to ensure text isn't split
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    value is 23\\% which is high.
    Costs \\$100.
    \\end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    text_items = [
        t.text
        for t in doc.texts
        if t.label == DocItemLabel.TEXT or t.label == DocItemLabel.PARAGRAPH
    ]
    full_text = " ".join(text_items)

    # "23%" should be together, not "23" and "%" split
    assert "23%" in full_text or "23\\%" in full_text
    # Should not have loose "%" newline
    assert "which is high" in full_text
    assert "$100" in full_text or "\\$100" in full_text


def test_latex_is_valid():
    """Test is_valid method"""
    # Valid document
    latex_content = b"\\documentclass{article}\\begin{document}Content\\end{document}"
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    assert backend.is_valid() is True

    # Empty document
    empty_content = b"   "
    in_doc_empty = InputDocument(
        path_or_stream=BytesIO(empty_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="empty.tex",
    )
    backend_empty = LatexDocumentBackend(
        in_doc=in_doc_empty, path_or_stream=BytesIO(empty_content)
    )
    assert backend_empty.is_valid() is False


def test_latex_supports_pagination():
    """Test supports_pagination class method"""
    assert LatexDocumentBackend.supports_pagination() is False


def test_latex_supported_formats():
    """Test supported_formats class method"""
    formats = LatexDocumentBackend.supported_formats()
    assert InputFormat.LATEX in formats


def test_latex_file_path_loading(tmp_path):
    """Test loading LaTeX from file path instead of BytesIO"""
    latex_file = tmp_path / "test.tex"
    latex_file.write_text(
        r"""
    \documentclass{article}
    \begin{document}
    File content here.
    \end{document}
    """
    )

    in_doc = InputDocument(
        path_or_stream=latex_file,
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=latex_file)
    doc = backend.convert()

    md = doc.export_to_markdown()
    assert "File content here" in md


def test_latex_no_document_env():
    """Test LaTeX without document environment processes all nodes"""
    latex_content = b"""
    \\section{Direct Section}
    Some direct content without document environment.
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    md = doc.export_to_markdown()
    assert "Direct Section" in md or "direct content" in md


def get_latex_converter():
    """Create a DocumentConverter for LaTeX files."""
    converter = DocumentConverter(allowed_formats=[InputFormat.LATEX])
    return converter


def test_e2e_latex_conversions(latex_paths):
    """E2E test for LaTeX conversions with ground-truth comparison."""
    if not latex_paths:
        pytest.skip("No LaTeX test files found")

    converter = get_latex_converter()

    for latex_path in latex_paths:
        if latex_path.parent.resolve() == LATEX_DATA_DIR.resolve():
            gt_name = latex_path.name
        else:
            gt_name = f"{latex_path.parent.name}_{latex_path.name}"

        gt_path = LATEX_DATA_DIR.parent / "groundtruth" / "docling_v2" / gt_name

        conv_result: ConversionResult = converter.convert(latex_path)
        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown()
        assert verify_export(pred_md, str(gt_path) + ".md", generate=GENERATE), (
            f"Markdown export mismatch for {latex_path}"
        )

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", generate=GENERATE), (
            f"Indented text export mismatch for {latex_path}"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE), (
            f"Document JSON mismatch for {latex_path}"
        )


def test_latex_document_with_leading_comments():
    """Test that documents starting with comment lines don't cause regex errors"""
    latex_content = b"""% This is a leading comment
% Another comment line
\\documentclass{article}
\\begin{document}
\\section{Test Section}
This is test content.
\\end{document}
"""
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    # Should parse successfully without regex errors
    assert len(doc.texts) > 0
    md = doc.export_to_markdown()
    assert "Test Section" in md
    assert "test content" in md


def test_latex_filecontents_ignored():
    """Test filecontents environment is ignored"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{filecontents}{sample.bib}
    @article{test, author={A}, title={B}}
    \\end{filecontents}
    \\begin{document}
    Actual content.
    \\end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    md = doc.export_to_markdown()
    assert "Actual content" in md
    # filecontents should not appear in output
    assert "@article" not in md


def test_latex_convert_error_fallback():
    """Test convert() returns an empty doc (not an exception) when _do_parse_and_process errors."""
    latex_content = b"\\documentclass{article}\\begin{document}Hello\\end{document}"
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    options = LatexBackendOptions(parse_timeout=0.05)
    backend = LatexDocumentBackend(
        in_doc=in_doc, path_or_stream=BytesIO(latex_content), options=options
    )

    def _raise(doc):
        raise RuntimeError("Simulated parse failure")

    backend._do_parse_and_process = _raise  # type: ignore[method-assign]
    doc = backend.convert()
    assert doc is not None


def test_latex_input_cycle_detection(tmp_path):
    """Test that circular \\input doesn't stack overflow"""
    # Create two files that reference each other
    file_a = tmp_path / "a.tex"
    file_b = tmp_path / "b.tex"

    file_a.write_text(
        "\\documentclass{article}\\begin{document}A content\\input{b}\\end{document}"
    )
    file_b.write_text("B content\\input{a}")

    in_doc = InputDocument(
        path_or_stream=file_a,
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="a.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=file_a)
    # Should not crash / stack overflow
    doc = backend.convert()
    md = doc.export_to_markdown()
    assert "A content" in md


def test_latex_author_date():
    """Test \\author and \\date text is preserved"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\title{My Paper}
    \\author{Jane Doe}
    \\date{January 2025}
    Some content.
    \\end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    md = doc.export_to_markdown()
    assert "Jane Doe" in md
    assert "January 2025" in md
