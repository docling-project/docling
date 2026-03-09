import warnings
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Union

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.backend_options import PdfBackendOptions

if TYPE_CHECKING:
    from docling.datamodel.document import InputDocument


class DoclingParseV2DocumentBackend(DoclingParseDocumentBackend):
    def __init__(
        self,
        in_doc: "InputDocument",
        path_or_stream: Union[BytesIO, Path],
        options: PdfBackendOptions = PdfBackendOptions(),
    ):
        warnings.warn(
            "DoclingParseV2DocumentBackend was removed in docling 2.74.0 and will raise an "
            "error in a future release. Use DoclingParseDocumentBackend instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(in_doc, path_or_stream, options)
