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
