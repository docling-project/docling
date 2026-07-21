from typing import Optional, Any
from fastapi import Request, status
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class BaseResponse(BaseModel):
    code: int
    errMsg: str
    data: Optional[Any] = None


def make_success_response(data: Optional[Any] = None):
    return BaseResponse(code=0, errMsg='success', data=data)


def make_failure_response(
    errMsg: str = 'failure',
    *,
    error_code: Optional[Any] = None,
    stage: Optional[str] = None,
    error_kind: Optional[str] = None,
):
    # #329: strict 경로에서 어느 단계(stage)에서 어떤 성격(error_kind: transient/permanent/timeout)의
    # 실패인지 caller(Temporal activity)가 알 수 있게 envelope 에 실어 준다.
    # (필드명은 root main.py 의 기존 error_type=예외 클래스명 과의 충돌을 피해 error_kind 로 통일.)
    body: dict = dict(code=1, errMsg=errMsg, data=None)
    if error_code is not None:
        body['error_code'] = error_code
    if stage is not None:
        body['stage'] = stage
    if error_kind is not None:
        body['error_kind'] = error_kind
    return JSONResponse(body, status_code=200)


def failure_response_from_exc(exc: Exception):
    """예외에서 error_code/stage/error_kind 를 덕타이핑으로 추출해 실패 envelope 생성(#329).

    facade 별 GenosServiceException 사본(서로 다른 클래스)과 common.exception 사본을
    모두 동일하게 처리하기 위해 getattr 로 접근한다. 기존 per-route 응답과의 호환을 위해
    error_code 가 있으면 legacy 'error_msg' 미러 필드도 함께 싣는다.
    실패 성격은 예외의 error_type attribute 에서 읽되, 응답 키는 error_kind 로 통일한다.
    """
    err_msg = str(getattr(exc, 'error_msg', None) or exc)
    error_code = getattr(exc, 'error_code', None)
    body: dict = dict(code=1, errMsg=err_msg, data=None)
    if error_code is not None:
        body['error_code'] = error_code
        body['error_msg'] = err_msg  # legacy mirror (기존 caller 호환)
    stage = getattr(exc, 'stage', None)
    if stage is not None:
        body['stage'] = stage
    error_kind = getattr(exc, 'error_type', None)
    if error_kind is not None:
        body['error_kind'] = error_kind
    return JSONResponse(body, status_code=200)


async def assert_cancelled(request: Request):
    if await request.is_disconnected():
        raise HTTPException(status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST, detail='Assertion Cancelled')
