from io import BytesIO
from pathlib import Path

from docling.backend.latex_backend import LatexDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.document_converter import DocumentConverter

LATEX_DATA_DIR = Path("./tests/data/latex/")


def make_backend(
    latex_content: bytes, filename: str = "test.tex", **backend_kwargs
) -> LatexDocumentBackend:
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename=filename,
    )
    return LatexDocumentBackend(
        in_doc=in_doc,
        path_or_stream=BytesIO(latex_content),
        **backend_kwargs,
    )


def make_backend_from_path(
    path: Path, filename: str | None = None
) -> LatexDocumentBackend:
    in_doc = InputDocument(
        path_or_stream=path,
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename=filename or path.name,
    )
    return LatexDocumentBackend(in_doc=in_doc, path_or_stream=path)


def get_latex_converter() -> DocumentConverter:
    return DocumentConverter(allowed_formats=[InputFormat.LATEX])
