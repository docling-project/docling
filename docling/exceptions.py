class BaseError(RuntimeError):
    pass


class ConversionError(Exception):
    """Custom exception for conversion errors."""
    pass


class HwpConversionError(ConversionError):
    """Custom exception for HWP conversion errors."""
    pass


class OperationNotAllowed(BaseError):
    pass
