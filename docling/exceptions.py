# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from docling.utils.ocr_language import OcrLanguageResolver, OcrLanguageSupport


class BaseError(RuntimeError):
    pass


class ConversionError(BaseError):
    pass


class DocumentLoadError(ConversionError):
    """A backend could not parse the input bytes into a document.

    Raised in a backend's load path to signal bad input, as distinct from
    internal defects (missing dependency, bug). Subclasses ``RuntimeError`` via
    ``BaseError``, so existing ``except RuntimeError`` callers keep working.
    """


class OperationNotAllowed(BaseError):
    pass


class SecurityError(BaseError):
    pass


class AcceleratorDeviceNotAvailableError(BaseError):
    """Raised when an explicitly requested accelerator device is not available."""


class OcrLanguageNotSupportedError(BaseError):
    """Raised when an OCR engine has no model for a requested language.

    Docling never silently substitutes a different recognizer: when the
    canonicalized request cannot be served, the engine says so and names what it
    does support.
    """

    def __init__(
        self,
        engine: str,
        language: str,
        supported: "OcrLanguageSupport | None" = None,
        detail: str | None = None,
    ):
        self.engine = engine
        self.language = language
        self.detail = detail
        self.supported = supported if supported is not None else OcrLanguageSupport()
        message = f"{engine} has no model for the OCR language {language!r}."
        if detail:
            message = f"{message} {detail}"
        if self.supported.bcp47:
            message = f"{message} Supported: {', '.join(self.supported.bcp47)}."
        if self.supported.native:
            # Rendered the way they have to be written back, prefix and all.
            codes = ", ".join(
                f"{OcrLanguageResolver._NATIVE_PREFIX}{code}"
                for code in self.supported.native
            )
            message = f"{message} Engine codes: {codes}."
        super().__init__(message)
