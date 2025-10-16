from io import BytesIO
from pathlib import Path
from typing import Union

from docling_core.types.doc import DoclingDocument
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


class DoclingJSONBackend(DeclarativeDocumentBackend):
    @override
    def __init__(
        self, in_doc: InputDocument, path_or_stream: Union[BytesIO, Path]
    ) -> None:
        super().__init__(in_doc, path_or_stream)

        # given we need to store any actual conversion exception for raising it from
        # convert(), this captures the successful result or the actual error in a
        # mutually exclusive way:
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
        return {InputFormat.JSON_DOCLING}

    def _get_doc_or_err(self) -> Union[DoclingDocument, Exception]:
        try:
            json_data: Union[str, bytes]
            if isinstance(self.path_or_stream, Path):
                with open(self.path_or_stream, encoding="utf-8") as f:
                    json_data = f.read()
            elif isinstance(self.path_or_stream, BytesIO):
                json_data = self.path_or_stream.getvalue()
            else:
                raise RuntimeError(f"Unexpected: {type(self.path_or_stream)=}")
            return DoclingDocument.model_validate_json(json_data=json_data)
        except Exception as e:
            return e

    @override
    def convert(self) -> DoclingDocument:
        if isinstance(self._doc_or_err, DoclingDocument):
            return self._doc_or_err
        else:
            raise self._doc_or_err
