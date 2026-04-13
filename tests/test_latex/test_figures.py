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
