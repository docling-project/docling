# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import re
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.transforms.serializer.latex import LaTeXParams
from docling_core.types.doc import ContentLayer, DocItemLabel, GroupLabel

from docling.backend.latex_backend import LatexDocumentBackend
from docling.datamodel.backend_options import LatexBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument, InputDocument
from docling.document_converter import DocumentConverter

from ..test_data_gen_flag import GEN_TEST_DATA
from ..verify_utils import verify_document, verify_export
from .latex_serializer import SourceLaTeXDocSerializer, SourceLaTeXInlineSerializer

GENERATE = GEN_TEST_DATA
LATEX_DATA_DIR = Path("./tests/data/latex/sources/")


def test_latex_unknown_macro_fallback():
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\unknownmacro{Known Content}
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
    assert "Known Content" in md


def test_latex_preserves_source_commands():
    latex_content = rb"""\documentclass{article}
\usepackage[utf8]{inputenc}
\newenvironment{custom}[1]{\textbf{#1}}{\par}
\newcommand{\myterm}[1]{first #1
	second}
\def\legacy{old}
\begin{document}
Body text.
\end{document}
"""
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    expected = [
        r"\documentclass{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\newenvironment{custom}[1]{\textbf{#1}}{\par}",
        "\\newcommand{\\myterm}[1]{first #1\n\tsecond}",
        r"\def\legacy{old}",
    ]
    source_commands = [
        item for item in doc.texts if item.content_layer == ContentLayer.FURNITURE
    ]
    assert [item.label for item in source_commands] == [DocItemLabel.TEXT] * len(
        expected
    )
    assert [item.text for item in source_commands] == expected
    assert [item.orig for item in source_commands] == expected
    assert all(command in doc.export_to_doclang() for command in expected)
    restored = DoclingDocument.model_validate_json(doc.model_dump_json())
    assert [
        item.text
        for item in restored.texts
        if item.content_layer == ContentLayer.FURNITURE
    ] == expected
    assert "Body text." in doc.export_to_markdown()
    assert all(command not in doc.export_to_markdown() for command in expected)

    parsed_latex = (
        SourceLaTeXDocSerializer(
            doc=doc,
            params=LaTeXParams(
                layers={ContentLayer.BODY, ContentLayer.FURNITURE},
                document_class="",
                packages=[],
                include_annotations=False,
                enable_chart_tables=False,
            ),
            inline_serializer=SourceLaTeXInlineSerializer(),
        )
        .serialize()
        .text
    )
    preamble, body = parsed_latex.split(r"\begin{document}", maxsplit=1)
    assert [preamble.index(command) for command in expected] == sorted(
        preamble.index(command) for command in expected
    )
    assert all(command not in body for command in expected)
    assert r"\textbackslash{}newcommand" not in parsed_latex
    assert "Body text." in body


def test_latex_preserves_complete_newtheorem_declarations():
    declarations = [
        r"\newtheorem{theorem}{Theorem}",
        r"\newtheorem{corollary}[theorem]{Corollary}",
        r"\newtheorem{lemma}{Lemma}[section]",
        r"\newtheorem*{remark}{Remark}",
    ]
    source = (
        "\\documentclass{article}\n"
        + "\n".join(declarations)
        + "\n\\begin{document}\nBody text.\n\\end{document}"
    ).encode("utf-8")
    stream = BytesIO(source)
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    doc = LatexDocumentBackend(in_doc=in_doc, path_or_stream=stream).convert()

    source_commands = [
        item.text for item in doc.texts if item.content_layer == ContentLayer.FURNITURE
    ]
    assert source_commands[1:] == declarations
    assert all(command in doc.export_to_doclang() for command in declarations)

    parsed = (
        SourceLaTeXDocSerializer(
            doc=doc,
            params=LaTeXParams(
                layers={ContentLayer.BODY, ContentLayer.FURNITURE},
                document_class="",
                packages=[],
                include_annotations=False,
                enable_chart_tables=False,
            ),
            inline_serializer=SourceLaTeXInlineSerializer(),
        )
        .serialize()
        .text
    )
    preamble, body = parsed.split(r"\begin{document}", maxsplit=1)
    assert all(command in preamble for command in declarations)
    assert all(command not in body for command in declarations)
    assert "Body text." in body


def test_latex_label():
    """Test label macro parsing"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\section{Introduction}
    \\label{sec:intro}
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
    backend.convert()

    # Labels are stored internally
    assert "sec:intro" in backend.labels


def test_latex_title_macro():
    """Test title macro inside document"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\title{Document Title}
    \\maketitle
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

    title_items = [t for t in doc.texts if t.label == DocItemLabel.TITLE]
    assert len(title_items) >= 1


def test_latex_title_with_optional_short_title():
    latex_content = (
        rb"\documentclass{article}\begin{document}"
        rb"\title[Short $x$]{Long $y$}\maketitle"
        rb"\end{document}"
    )
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()
    titles = [item for item in doc.texts if item.label == DocItemLabel.TITLE]
    assert len(titles) == 1
    assert titles[0].children
    assert "<formula>y</formula>" in doc.export_to_doclang()
    assert "Short" not in doc.export_to_markdown()


def test_latex_heading_levels():
    """Test different heading levels"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\part{Part One}
    \\chapter{Chapter One}
    \\section{Section One}
    \\subsection{Subsection One}
    \\subsubsection{Subsubsection One}
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

    headers = [t for t in doc.texts if t.label == DocItemLabel.SECTION_HEADER]
    assert len(headers) >= 3


def test_latex_section_heading_and_title_keep_inline_formulas():
    source = rb"""\documentclass{article}
\title[Short $x$]{From $A_1$ to the $B^2$ case}
\begin{document}
\maketitle
\section[Short $k$]{Dimension $2$ and rank $r$}
\subsection{The $q$ case}
\end{document}"""
    stream = BytesIO(source)
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    doc = LatexDocumentBackend(in_doc=in_doc, path_or_stream=stream).convert()

    titles = [item for item in doc.texts if item.label == DocItemLabel.TITLE]
    headings = [item for item in doc.texts if item.label == DocItemLabel.SECTION_HEADER]
    assert len(titles) == 1
    assert len(headings) == 2
    assert titles[0].children
    assert all(item.children for item in headings)
    assert [item.level for item in headings] == [1, 2]

    dclg = doc.export_to_doclang()
    assert "<formula>A_1</formula>" in dclg
    assert "<formula>B^2</formula>" in dclg
    assert "<formula>2</formula>" in dclg
    assert "<formula>r</formula>" in dclg
    assert "<formula>q</formula>" in dclg
    assert "Short" not in dclg

    parsed = (
        SourceLaTeXDocSerializer(
            doc=doc,
            params=LaTeXParams(
                layers={ContentLayer.BODY, ContentLayer.FURNITURE},
                document_class="",
                packages=[],
                include_annotations=False,
                enable_chart_tables=False,
            ),
            inline_serializer=SourceLaTeXInlineSerializer(),
        )
        .serialize()
        .text
    )
    assert r"From $A_1$ to the $B^2$ case" in parsed
    assert r"\section{Dimension $2$ and rank $r$}" in parsed
    assert r"\subsection{The $q$ case}" in parsed
    assert r"\$" not in parsed


def test_latex_text_formatting():
    """Test text formatting macros"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    This is \\textbf{bold} and \\textit{italic} and \\emph{emphasized}.
    Also \\texttt{monospace} and \\underline{underlined}.
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
    assert "bold" in md
    assert "italic" in md
    assert "emphasized" in md


def test_latex_inline_macro_around_paragraph_break():
    """Inline macros next to a paragraph break keep their spacing and paragraph"""
    latex_content = b"""\\documentclass{article}
\\begin{document}
A \\textit{it} text. B \\textbf{bf} more.

C \\emph{em}x and \\textit{it}.
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

    assert [t.text for t in doc.texts if t.content_layer == ContentLayer.BODY] == [
        "A it text. B bf more.",
        "C emx and it.",
    ]


def test_latex_marginpar():
    """Test marginpar macro is handled without error"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    Main text with marginpar.
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
    assert doc is not None


def test_latex_newline_macro():
    """Test handling of \\\\ newline macro"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    Line one\\\\
    Line two
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

    # Should not crash
    assert doc is not None


def test_latex_tilde_macro():
    """Test ~ (non-breaking space) macro handling"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    Dr.~Smith arrived.
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
    assert "Smith" in md


def test_latex_custom_macro_with_backslash():
    """Test that custom macros containing backslashes don't cause regex errors"""
    latex_content = b"""\\documentclass{article}
\\newcommand{\\myterm}{special term}
\\newcommand{\\myvalue}{42}
\\begin{document}
This is \\myterm and the value is \\myvalue.
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
    # The macro expansion should work
    assert "special term" in md and "42" in md


def extract_macro_name_old(raw_string):
    """The OLD buggy implementation"""
    # This was the original broken code
    macro_name_raw = raw_string.strip("{} ")
    macro_name = macro_name_raw.lstrip("\\")
    return macro_name


def extract_macro_name_new(raw_string):
    """The NEW fixed implementation"""
    # This is the fixed code
    macro_name = raw_string.strip("{} \n\t\\")
    if macro_name.startswith("\\"):
        macro_name = macro_name[1:]
    return macro_name


def test_macro_extraction():
    """Test various formats of macro names"""
    print("\n" + "=" * 80)
    print("TESTING MACRO NAME EXTRACTION LOGIC")
    print("=" * 80)

    # Test cases: (input, expected_output)
    test_cases = [
        (r"{\myterm}", "myterm"),
        (r"\myterm", "myterm"),
        (r"{ \myterm }", "myterm"),
        (r"{  \myvalue  }", "myvalue"),
        (r"{\important}", "important"),
        (r"{ \test }", "test"),
        (r"{\alpha}", "alpha"),
    ]

    print("\n" + "-" * 80)
    print("OLD (BUGGY) IMPLEMENTATION:")
    print("-" * 80)
    old_passed = 0
    for input_str, expected in test_cases:
        result = extract_macro_name_old(input_str)
        status = "✓" if result == expected else "✗"
        print(
            f"{status} Input: {input_str!r:20} → Result: {result!r:15} (expected: {expected!r})"
        )
        if result == expected:
            old_passed += 1

    print("\n" + "-" * 80)
    print("NEW (FIXED) IMPLEMENTATION:")
    print("-" * 80)
    new_passed = 0
    for input_str, expected in test_cases:
        result = extract_macro_name_new(input_str)
        status = "✓" if result == expected else "✗"
        print(
            f"{status} Input: {input_str!r:20} → Result: {result!r:15} (expected: {expected!r})"
        )
        if result == expected:
            new_passed += 1

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"OLD implementation: {old_passed}/{len(test_cases)} tests passed")
    print(f"NEW implementation: {new_passed}/{len(test_cases)} tests passed")

    if new_passed == len(test_cases):
        print("\n✓✓ SUCCESS: New implementation fixes the bug!")
    else:
        print("\n✗✗ FAILURE: New implementation still has issues")

    assert new_passed == len(test_cases), (
        f"New implementation failed: {new_passed}/{len(test_cases)} tests passed"
    )


def test_edge_cases():
    """Test edge cases and special characters"""
    print("\n" + "=" * 80)
    print("TESTING EDGE CASES")
    print("=" * 80)

    edge_cases = [
        # Format: (input, expected, description)
        (r"{\cmd}", "cmd", "Simple macro"),
        (r"{\\cmd}", "cmd", "Double backslash"),
        (r"{ \cmd }", "cmd", "Spaces around macro"),
        (r"{\   cmd   }", "cmd", "Extra spaces"),
        (r"{\my_macro}", "my_macro", "Underscore in name"),
        (r"{\MyMacro}", "MyMacro", "CamelCase"),
        (r"{\MACRO}", "MACRO", "Uppercase"),
    ]

    all_passed = True
    for input_str, expected, description in edge_cases:
        result = extract_macro_name_new(input_str)
        status = "✓" if result == expected else "✗"
        print(f"{status} {description:25} | Input: {input_str!r:20} → {result!r}")
        if result != expected:
            print(f"   Expected: {expected!r}")
            all_passed = False

    if all_passed:
        print("\n✓✓ All edge cases passed!")
    else:
        print("\n✗✗ Some edge cases failed")

    assert all_passed, "Some edge cases failed"


def test_debug_macro_extraction():
    """Debug test to see what's actually being extracted"""
    from io import BytesIO

    from docling.backend.latex_backend import LatexDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import InputDocument

    latex_content = b"""\\documentclass{article}
\\newcommand{\\myterm}{special term}
\\newcommand{\\myvalue}{42}
\\begin{document}
This is \\myterm and the value is \\myvalue.
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

    print(f"\n{'=' * 80}")
    print("DEBUG INFO:")
    print(f"{'=' * 80}")
    print(f"Custom macros extracted: {backend._custom_macros}")
    print(f"Number of text items: {len(doc.texts)}")
    print("\nText items:")
    for i, text_item in enumerate(doc.texts):
        print(f"  {i}: {text_item.label} = {text_item.text!r}")

    md = doc.export_to_markdown()
    print("\nMarkdown output:")
    print(md)
    print(f"{'=' * 80}\n")

    # Check if macros were registered
    assert "myterm" in backend._custom_macros, "myterm not in _custom_macros!"
    assert backend._custom_macros["myterm"] == "special term"
    assert "special term" in md, f"'special term' not in output: {md!r}"
    assert "42" in md, f"'42' not in output: {md!r}"


def test_latex_href_macro():
    """Test \\href{url}{display} emits a markdown-style link."""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    Visit \href{https://example.com}{Example Site} for more.
    \end{document}
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
    assert "Example Site" in md
    assert "https://example.com" in md


def test_latex_textcolor_macro():
    """Test \\textcolor{color}{text} extracts the text content and ignores the color."""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    This is \textcolor{red}{important} text.
    Also \colorbox{yellow}{highlighted} here.
    \end{document}
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
    assert "important" in md
    assert "highlighted" in md
    # Color names should not leak into output
    assert "red" not in md
    assert "yellow" not in md


def test_latex_custom_macro_parameters():
    """Test custom macros with parameters expand arguments before formatting is unwrapped."""
    latex_content = rb"""
    \documentclass{article}
    \newcommand{\highlight}[1]{\textcolor{white}{\textbf{#1}}}
    \newcommand{\metric}[2]{#1{\scriptsize$\_{#2}$}}
    \begin{document}
    \highlight{Result}
    \metric{Accuracy}{test}
    \end{document}
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
    assert "Result" in md
    assert "Accuracy" in md
    assert "test" in md
    assert "#1" not in md
    assert "#2" not in md
    assert "\\textcolor" not in md
    assert "\\scriptsize" not in md


def test_latex_legacy_font_switches():
    """Test legacy font/size switches (\\bf, \\it, \\tt, \\large, \\tiny) are silently ignored."""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    {\bf bold text} and {\it italic text}.
    {\tt monospace} and {\large big} and {\tiny small}.
    Normal content here.
    \end{document}
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
    # Content inside the groups should still appear
    assert "bold text" in md
    assert "italic text" in md
    assert "Normal content here" in md


def test_latex_accent_macro():
    """Test accent macros (\\'{e}, \\`{a}) are converted to Unicode characters."""
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    \section{G\"odel}
    caf\'{e} and na\"{i}ve, Espa\~na, J{\"u}rgen, {\c{C}}a\u{g}lar, \`a la.

    {\o}, \ss{}, \aa, \v{s}, \'{\i}, http://x.org/\~{}user.
    \end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    texts = [t.text for t in doc.texts if t.content_layer == ContentLayer.BODY]
    assert texts == [
        "Gödel",
        "café and naïve, España, Jürgen, Çağlar, à la.",
        "ø, ß, å, š, í, http://x.org/~user.",
    ]


def test_latex_accent_on_custom_macro():
    """An accent keeps a document macro in its argument: \\'{\\vowel} -> é."""
    latex_content = rb"""
    \documentclass{article}
    \newcommand{\vowel}{e}
    \newcommand{\dotless}{\i}
    \newcommand{\letter}{e}
    \newcommand{\chained}{\letter}
    \newcommand{\cycle}{\cycle}
    \begin{document}
    \section{caf\'{\chained}}
    caf\'{\vowel} followed by text, caf\'\vowel, caf\'{\chained}, \'{\dotless}.

    x\'{\cycle}y.
    \end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    doc = backend.convert()

    texts = [t.text for t in doc.texts if t.content_layer == ContentLayer.BODY]
    # A macro that never resolves stops expanding and keeps its text, unaccented.
    assert texts == [
        "café",
        "café followed by text, café, café, í.",
        "x\\cycley.",
    ]


def test_latex_renewcommand():
    """Test \\renewcommand and \\providecommand macros are expanded"""
    latex_content = rb"""
    \documentclass{article}
    \newcommand{\foo}{original}
    \renewcommand{\foo}{replaced}
    \providecommand{\bar}{provided}
    \begin{document}
    Value is \foo{} and \bar{}.
    \end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    backend.convert()

    # renewcommand should have overwritten the original
    assert "foo" in backend._custom_macros
    assert backend._custom_macros["foo"] == "replaced"
    assert "bar" in backend._custom_macros
    assert backend._custom_macros["bar"] == "provided"


def test_vspace_argument_does_not_leak():
    """Verify \\vspace{-1mm} does not produce '-1mm' as a text item."""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    Before text.
    \\vspace{-1mm}
    After text.
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

    all_text = " ".join(t.text for t in doc.texts)
    assert "-1mm" not in all_text, f"vspace argument leaked into text: {all_text!r}"
    assert "Before text" in all_text
    assert "After text" in all_text


def test_hspace_argument_does_not_leak():
    """Verify \\hspace{0.2cm} does not produce '0.2cm' as a text item."""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    Left text.\\hspace{0.2cm}Right text.
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

    all_text = " ".join(t.text for t in doc.texts)
    assert "0.2cm" not in all_text, f"hspace argument leaked into text: {all_text!r}"
    assert "Left text" in all_text
    assert "Right text" in all_text
