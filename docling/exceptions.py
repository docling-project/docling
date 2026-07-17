from typing import List, Tuple


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


class DoclingError(Exception):
    """Base exception for all Docling-related errors."""

    pass


class DoclingModelDownloadError(DoclingError):
    """Raised when download a model fails (e.g. connection issues, invalid token)."""

    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception


class DoclingMultiModelDownloadError(DoclingError):
    """Raised when one or more models fail during batch download"""

    def __init__(self, failures: List[Tuple[str, Exception]]):
        self.failures = failures
        msgs = [f"{model_name}: {exc}" for model_name, exc in failures]
        super().__init__(
            f"Failed to download {len(failures)} model(s):\n" + "\n".join(msgs)
        )
