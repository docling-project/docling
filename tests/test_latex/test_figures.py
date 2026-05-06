from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel, GroupLabel
from docling_core.types.doc.document import PictureMeta

from docling.backend.latex_backend import LatexDocumentBackend
from docling.datamodel.backend_options import LatexBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument, InputDocument
from docling.document_converter import DocumentConverter

from ..test_data_gen_flag import GEN_TEST_DATA
from ..verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA
LATEX_DATA_DIR = Path("./tests/data/latex/")


def test_latex_caption():
    """Test caption macro parsing via includegraphics"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\includegraphics{test.png}
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

    # includegraphics creates a caption with the image path
    caption_items = [t for t in doc.texts if t.label == DocItemLabel.CAPTION]
    assert len(caption_items) >= 1
    assert "test.png" in caption_items[0].text


def test_latex_includegraphics():
    """Test includegraphics with actual image file"""
    import tempfile
    from pathlib import Path

    from PIL import Image as PILImage

    # Create a temporary directory and test image
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tex_file = tmpdir_path / "test.tex"
        img_file = tmpdir_path / "test_image.png"

        # Create a simple test image with known DPI
        test_img = PILImage.new("RGB", (100, 50), color="red")
        test_img.save(img_file, dpi=(96, 96))

        latex_content = b"""
        \\documentclass{article}
        \\begin{document}
        \\includegraphics{test_image.png}
        \\end{document}
        """

        # Write LaTeX content to file
        tex_file.write_bytes(latex_content)

        in_doc = InputDocument(
            path_or_stream=tex_file,
            format=InputFormat.LATEX,
            backend=LatexDocumentBackend,
            filename="test.tex",
        )
        backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=tex_file)
        doc = backend.convert()

        # Verify picture was created
        assert len(doc.pictures) >= 1
        picture = doc.pictures[0]

        # Verify image was embedded (not None)
        assert picture.image is not None

        # Verify caption was created
        assert len(picture.captions) >= 1
        assert "test_image.png" in picture.captions[0].resolve(doc).text


def test_latex_includegraphics_missing_image():
    """Test includegraphics gracefully handles missing images"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\includegraphics{nonexistent_image.png}
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

    # Picture should still be created with caption
    assert len(doc.pictures) >= 1
    picture = doc.pictures[0]

    # Image should be None (couldn't load)
    assert picture.image is None

    # Caption should still exist
    assert len(picture.captions) >= 1
    assert "nonexistent_image.png" in picture.captions[0].resolve(doc).text


def test_latex_figure_environment():
    """Test figure environment parsing"""
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{figure}
    \\includegraphics{test.png}
    \\caption{Test figure}
    \\end{figure}
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

    assert len(doc.pictures) >= 1
    captions = [t for t in doc.texts if t.label == DocItemLabel.CAPTION]
    assert len(captions) >= 1


def test_latex_figure_with_caption():
    """Test that figure environment properly groups caption and image"""
    latex_content = b"""\\documentclass{article}
\\begin{document}
\\begin{figure}
\\includegraphics{test.png}
\\caption{This is a test figure caption}
\\label{fig:test}
\\end{figure}
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

    # Should have a figure group
    figure_groups = [g for g in doc.groups if g.name == "figure"]
    assert len(figure_groups) >= 1

    # Should have picture and caption
    assert len(doc.pictures) >= 1
    captions = [t for t in doc.texts if t.label == DocItemLabel.CAPTION]
    # includegraphics creates one caption, and \caption macro creates another
    assert len(captions) >= 1


def test_latex_tikzpicture_atomic_capture():
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    Before.
    \begin{tikzpicture}
    \draw (0,0) -- (1,1);
    \node at (0.5,0.5) {A};
    \end{tikzpicture}
    After.
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

    tikz_pictures = [
        p
        for p in doc.pictures
        if p.meta and isinstance(p.meta, PictureMeta) and p.meta.code is not None
    ]
    assert len(tikz_pictures) >= 1
    assert any(
        "\\begin{tikzpicture}" in p.meta.code.text
        for p in tikz_pictures
        if p.meta and p.meta.code
    )
    assert any(
        "\\end{tikzpicture}" in p.meta.code.text
        for p in tikz_pictures
        if p.meta and p.meta.code
    )
    non_code_text = " ".join(t.text for t in doc.texts)

    assert "Before" in non_code_text
    assert "After" in non_code_text


def test_latex_tikzpicture_nested_atomic_capture():
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    \begin{tikzpicture}
    \draw (0,0) -- (1,1);
    \begin{tikzpicture}
    \draw (2,2) -- (3,3);
    \end{tikzpicture}
    \end{tikzpicture}
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

    tikz_pictures = [
        p
        for p in doc.pictures
        if p.meta and isinstance(p.meta, PictureMeta) and p.meta.code is not None
    ]
    assert len(tikz_pictures) >= 1
    captured = "\n".join(
        p.meta.code.text for p in tikz_pictures if p.meta and p.meta.code
    )

    assert captured.count("\\begin{tikzpicture}") >= 2
    assert captured.count("\\end{tikzpicture}") >= 2


def test_latex_tikzpicture_inside_figure_does_not_nest_figure_group():
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    \begin{figure}
    \begin{tikzpicture}
    \draw (0,0) -- (1,1);
    \end{tikzpicture}
    \end{figure}
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

    figure_groups = [g for g in doc.groups if g.name == "figure"]
    assert len(figure_groups) == 1


def test_latex_tikzpicture_end_with_whitespace_is_atomic():
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    \begin{tikzpicture}
    \draw (0,0) -- (1,1);
    \end { tikzpicture }
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

    tikz_pictures = [
        p
        for p in doc.pictures
        if p.meta and isinstance(p.meta, PictureMeta) and p.meta.code is not None
    ]
    assert len(tikz_pictures) >= 1
    assert any(
        "\\end { tikzpicture }" in picture.meta.code.text
        for picture in tikz_pictures
        if picture.meta and picture.meta.code
    )


def test_latex_tikzpicture_unclosed_warns_and_falls_back(caplog):
    latex_content = rb"""
    \documentclass{article}
    \begin{document}
    \begin{tikzpicture}
    \draw (0,0) -- (1,1);
    \end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    with caplog.at_level(
        "WARNING", logger="docling.backend.latex.handlers.environments"
    ):
        backend = LatexDocumentBackend(
            in_doc=in_doc, path_or_stream=BytesIO(latex_content)
        )
        doc = backend.convert()

    assert any(
        "tikzpicture extraction failed" in record.message for record in caplog.records
    )
    assert not any(
        p.meta and isinstance(p.meta, PictureMeta) and p.meta.code is not None
        for p in doc.pictures
    )
    md = doc.export_to_markdown()
    assert "(0,0)" in md
    assert "(1,1)" in md


def test_latex_tikzpicture_fallback_preserves_text_label(monkeypatch):
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    test
    \\end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))

    class DummyNode:
        def __init__(self):
            self.nodelist = []

    captured = {"label": None}

    def fake_extract(_node):
        return None

    def fake_process_nodes(nodes, doc, parent, formatting, text_label):
        captured["label"] = text_label

    monkeypatch.setattr(backend, "_extract_tikzpicture_atomic", fake_extract)
    monkeypatch.setattr(backend, "_process_nodes", fake_process_nodes)

    backend._process_tikzpicture(
        DummyNode(),
        DoclingDocument(name="dummy"),
        text_label=DocItemLabel.LIST_ITEM,
    )

    assert captured["label"] == DocItemLabel.LIST_ITEM


def test_latex_tikz_validate_depth_guard_and_arg_recursion():
    latex_content = b"""
    \\documentclass{article}
    \\begin{document}
    test
    \\end{document}
    """
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))

    assert backend._validate_tikz_nodelist([], depth=51) is False

    class DummyArg:
        def __init__(self, nodelist):
            self.nodelist = nodelist

    class DummyArgData:
        def __init__(self, argnlist):
            self.argnlist = argnlist

    class DummyNode:
        def __init__(self, nodeargd=None, nodelist=None):
            self.nodeargd = nodeargd
            self.nodelist = nodelist

    node = DummyNode(nodeargd=DummyArgData([DummyArg([])]), nodelist=[])
    assert backend._validate_tikz_nodelist([node], depth=0) is True
