from typing import Optional


class GenosServiceException(Exception):
    def __init__(
        self,
        error_code: str,
        error_msg: Optional[str] = None,
        msg_params: Optional[dict] = None,
        *,
        stage: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> None:
        self.code = 1
        self.error_code = error_code
        self.error_msg = error_msg or "GenOS Service Exception"
        self.msg_params = msg_params or {}
        # #329: 어느 단계에서 실패했는지(stage)와 실패 성격(error_type: transient/permanent/timeout).
        self.stage = stage
        self.error_type = error_type

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(code={self.code!r}, errMsg={self.error_msg!r})"
