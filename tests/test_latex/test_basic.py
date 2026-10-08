# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel, GroupLabel

from docling.backend.latex_backend import LatexDocumentBackend
from docling.datamodel.backend_options import LatexBackendOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument, InputDocument
from docling.document_converter import DocumentConverter, LatexFormatOption

from ..test_data_gen_flag import GEN_TEST_DATA
from ..verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA
LATEX_DATA_DIR = Path("./tests/data/latex/sources/")


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

        gt_path = LATEX_DATA_DIR.parent / "groundtruth" / gt_name

        conv_result: ConversionResult = converter.convert(latex_path)
        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown(compact_tables=True)
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


@pytest.mark.parametrize(
    ("encoding", "inputenc"), [("utf-8", "utf8"), ("latin-1", "latin1")]
)
def test_latex_input_file_in_main_file_encoding(tmp_path, encoding, inputenc):
    """An \\input file in the same encoding as the main file is kept."""
    main_file = tmp_path / "main.tex"
    main_file.write_text(
        "\\documentclass{article}\n"
        f"\\usepackage[{inputenc}]{{inputenc}}\n"
        "\\begin{document}\n"
        "Résumé of the thesis.\n\n"
        "\\input{chapter}\n"
        "\\end{document}\n",
        encoding=encoding,
    )
    (tmp_path / "chapter.tex").write_text(
        "\\section{Méthode}\nThe chapter body.\n", encoding=encoding
    )

    conv_result = get_latex_converter().convert(main_file)
    md = conv_result.document.export_to_markdown()

    assert conv_result.status == ConversionStatus.SUCCESS
    assert conv_result.errors == []
    assert "Résumé of the thesis." in md
    assert "Méthode" in md
    assert "The chapter body." in md


def _write_paper_with_includes(directory: Path) -> Path:
    """Write a main.tex that pulls in a sibling .tex file and a figure."""
    from PIL import Image as PILImage

    (directory / "intro.tex").write_text("\\section{Introduction}\nIncluded text.\n")
    PILImage.new("RGB", (20, 10), "blue").save(directory / "fig.png")
    main_file = directory / "main.tex"
    main_file.write_text(
        "\\documentclass{article}\n\\begin{document}\nMain text.\n"
        "\\input{intro}\n\\includegraphics{fig.png}\n\\end{document}\n"
    )
    return main_file


@pytest.mark.parametrize("relative", [False, True])
def test_latex_str_source_resolves_included_files(tmp_path, monkeypatch, relative):
    """A str source resolves \\input and \\includegraphics like a Path source."""
    main_file = _write_paper_with_includes(tmp_path)
    if relative:
        monkeypatch.chdir(tmp_path)
        source = "main.tex"
    else:
        source = str(main_file)

    doc = get_latex_converter().convert(source).document

    assert "Included text." in doc.export_to_markdown()
    assert len(doc.pictures) == 1
    assert doc.pictures[0].image is not None


def test_latex_stream_resolves_included_files_with_source_uri(tmp_path):
    """A stream resolves relative files against source_uri when it is given."""
    main_file = _write_paper_with_includes(tmp_path)
    stream = BytesIO(main_file.read_bytes())
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="main.tex",
    )

    doc = LatexDocumentBackend(
        in_doc=in_doc,
        path_or_stream=stream,
        options=LatexBackendOptions(source_uri=main_file),
    ).convert()

    assert "Included text." in doc.export_to_markdown()
    assert doc.pictures[0].image is not None


def test_latex_format_option_source_uri_only_for_local_str():
    """Only a local str source gets a source_uri; a URL has no local directory."""
    option = LatexFormatOption()

    local = option.backend_options_for_input("paper/main.tex")
    assert local is not None
    assert local.source_uri == Path("paper/main.tex")
    assert option.backend_options_for_input("https://example.com/main.tex") is None
    assert option.backend_options_for_input(Path("paper/main.tex")) is None

    custom = LatexFormatOption(backend_options=LatexBackendOptions(parse_timeout=5))
    options = custom.backend_options_for_input("paper/main.tex")
    assert options is not None
    assert options.parse_timeout == 5
    assert options.source_uri == Path("paper/main.tex")


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
