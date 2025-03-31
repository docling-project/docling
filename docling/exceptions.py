class BaseError(RuntimeError):
    pass


class ConversionError(BaseError):
    pass


class OperationNotAllowed(BaseError):
    pass
