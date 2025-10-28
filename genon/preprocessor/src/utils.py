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


def make_failure_response(errMsg: str = 'failure'):
    return JSONResponse(dict(code=1, errMsg=errMsg), status_code=200)


async def assert_cancelled(request: Request):
    if await request.is_disconnected():
        raise HTTPException(status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST, detail='Assertion Cancelled')
