# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from docling_core.types.doc import DoclingDocument
from typing_extensions import override

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.backend_options import BackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.utils.oxr_utils import parse_oxr_result

_log = logging.getLogger(__name__)


class OxrJSONBackend(DeclarativeDocumentBackend):
    """Backend for a saved OXR parse response.

    The response is converted by :func:`docling.utils.oxr_utils.parse_oxr_result`
    without a recognition pipeline. ``backend_options.enable_local_fetch``
    controls whether picture files named by the OXR host are read; they are
    skipped by default.
    """

    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
        options: Optional[BackendOptions] = None,
    ) -> None:
        super().__init__(in_doc, path_or_stream, options)

        # Store the converted document or the conversion error in a mutually
        # exclusive way, so convert() can raise the original failure.
        self._doc_or_err = self._get_doc_or_err()
        if isinstance(self._doc_or_err, Exception):
            # The converter reports a generic backend failure, so keep the
            # underlying reason in the log.
            _log.warning("%s: %s", self.file.name, self._doc_or_err)

    @override
    def is_valid(self) -> bool:
        return isinstance(self._doc_or_err, DoclingDocument)

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        # The declarative document already holds every page.
        return False

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.OXR}

    def _get_doc_or_err(self) -> Union[DoclingDocument, Exception]:
        try:
            if isinstance(self.path_or_stream, Path):
                content = self.path_or_stream.read_bytes()
                base_dir: Path | None = self.path_or_stream.parent
            elif isinstance(self.path_or_stream, BytesIO):
                content = self.path_or_stream.getvalue()
                base_dir = None
            else:
                raise RuntimeError(f"Unexpected: {type(self.path_or_stream)=}")
            return parse_oxr_result(
                content,
                load_local_images=self.options.enable_local_fetch,
                base_dir=base_dir,
            )
        except Exception as e:
            return e

    @override
    def convert(self) -> DoclingDocument:
        if isinstance(self._doc_or_err, DoclingDocument):
            return self._doc_or_err
        raise self._doc_or_err
