import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from docling_core.transforms.deserializer.doclang import DocLangDocDeserializer
from docling_core.types.doc import DoclingDocument, DocumentOrigin
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.backend_options import HwpBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


class HwpDocumentBackend(DeclarativeDocumentBackend):
    """Experimental backend for Korean HWP/HWPX (Hangul Word Processor) documents.

    The document is lowered to DocLang XML by the external ``hangulang`` package
    (a wrapper around the ``rhwp`` parser core) and then deserialized into a
    ``DoclingDocument`` through Docling's DocLang deserializer.

    This backend is experimental and depends on the early-stage (0.x)
    ``hangulang`` / ``rhwp`` projects; its output may change or be removed in
    future versions. The dependency is optional: install it with
    ``pip install docling[format-hwp]``.
    """

    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
        options: Optional[HwpBackendOptions] = None,
    ) -> None:
        hwp_options = options or HwpBackendOptions()
        super().__init__(in_doc, path_or_stream, hwp_options)
        self.hwp_options = hwp_options
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
        return {InputFormat.HWP}

    @override
    def convert(self) -> DoclingDocument:
        if isinstance(self._doc_or_err, DoclingDocument):
            return self._doc_or_err

        raise self._doc_or_err

    def _get_doc_or_err(self) -> Union[DoclingDocument, Exception]:
        try:
            data = self._read_bytes()
            doclang_xml = self._convert_to_doclang(data)
            doc = DocLangDocDeserializer().deserialize_str(doclang_xml)
            doc.origin = DocumentOrigin(
                filename=self.file.name or "file",
                mimetype="application/octet-stream",
                binary_hash=self.document_hash,
            )
            return doc
        except Exception as e:
            return e

    def _read_bytes(self) -> bytes:
        if isinstance(self.path_or_stream, Path):
            return self.path_or_stream.read_bytes()
        if isinstance(self.path_or_stream, BytesIO):
            return self.path_or_stream.getvalue()
        raise RuntimeError(f"Unexpected HWP input type: {type(self.path_or_stream)!r}")

    def _convert_to_doclang(self, data: bytes) -> str:
        try:
            from hangulang import convert_to_doclang
        except ImportError as exc:
            raise RuntimeError(
                "The 'hangulang' package is required for HWP/HWPX support. "
                "Install it with `pip install docling[format-hwp]`."
            ) from exc

        return convert_to_doclang(
            data, include_locations=self.hwp_options.include_locations
        )
