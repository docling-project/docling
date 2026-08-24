from io import BytesIO
from pathlib import Path
from typing import Union

from docling_core.transforms.deserializer.doclang import DocLangDocDeserializer
from docling_core.types.doc import DoclingDocument, DocumentOrigin
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


class DocLangDocumentBackend(DeclarativeDocumentBackend):
    @override
    def __init__(
        self, in_doc: InputDocument, path_or_stream: Union[BytesIO, Path]
    ) -> None:
        super().__init__(in_doc, path_or_stream)
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
        return {InputFormat.XML_DOCLANG}

    def _get_doc_or_err(self) -> Union[DoclingDocument, Exception]:
        try:
            if isinstance(self.path_or_stream, Path):
                text = self.path_or_stream.read_text(encoding="utf-8")
            elif isinstance(self.path_or_stream, BytesIO):
                text = self.path_or_stream.getvalue().decode("utf-8")
            else:
                raise RuntimeError(f"Unexpected: {type(self.path_or_stream)=}")

            doc = DocLangDocDeserializer().deserialize_str(text)
            doc.origin = DocumentOrigin(
                filename=self.file.name or "file",
                mimetype="application/xml",
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
