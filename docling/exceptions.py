class BaseError(RuntimeError):
    pass


class ConversionError(BaseError):
    pass


class OperationNotAllowed(BaseError):
    pass


class SecurityError(BaseError):
    pass


class AcceleratorDeviceNotAvailableError(BaseError):
    """Raised when an explicitly requested accelerator device is not available."""

