import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Union

from docling_core.types.doc.common.origin import DocumentOrigin
from docling_core.types.doc.document import DoclingDocument
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_DCLX_MIMETYPE = "application/zip"


class DocLangArchiveBackend(DeclarativeDocumentBackend):
    @override
    def __init__(
        self, in_doc: InputDocument, path_or_stream: Union[BytesIO, Path]
    ) -> None:
        super().__init__(in_doc, path_or_stream)
        self._temp_dir: Path | None = None
        self._doc_or_err = self._get_doc_or_err()

    @override
    def is_valid(self) -> bool:
        return isinstance(self._doc_or_err, DoclingDocument)

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.DCLX}

    def _get_doc_or_err(self) -> Union[DoclingDocument, Exception]:
        try:
            if isinstance(self.path_or_stream, Path):
                doc = DoclingDocument.load_from_doclang_archive(self.path_or_stream)
            elif isinstance(self.path_or_stream, BytesIO):
                self._temp_dir = Path(tempfile.mkdtemp(prefix="docling_dclx_"))
                archive_path = self._temp_dir / (self.file.name or "document.dclx")
                archive_path.write_bytes(self.path_or_stream.getvalue())
                artifacts_dir = self._temp_dir / "artifacts"
                doc = DoclingDocument.load_from_doclang_archive(
                    archive_path,
                    artifacts_dir=artifacts_dir,
                )
            else:
                raise RuntimeError(f"Unexpected: {type(self.path_or_stream)=}")

            doc.origin = DocumentOrigin(
                filename=self.file.name or "file",
                mimetype=_DCLX_MIMETYPE,
                binary_hash=self.document_hash,
            )
            return doc
        except Exception as e:
            return e

    @override
    def convert(self) -> DoclingDocument:
        if isinstance(self._doc_or_err, DoclingDocument):
            return self._doc_or_err

        raise self._doc_or_err

    @override
    def unload(self) -> None:
        if self._temp_dir is not None and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None
        super().unload()
