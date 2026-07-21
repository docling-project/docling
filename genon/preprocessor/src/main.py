import asyncio
import os
import sys
import traceback
import time

from fastapi import FastAPI, Request, Body
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from logger import Logger
from utils import make_failure_response, make_success_response, failure_response_from_exc
from config import cors_config
from common.exception import GenosServiceException
from common.settings import settings
from util.minio_resource import download_resource_files

sys.path.append(os.path.dirname(__file__) + '/util')

logger = Logger.getLogger(__name__)

app: FastAPI = FastAPI()
cors_config(app)


@app.exception_handler(GenosServiceException)
async def mlops_exception_handler(request, exc: GenosServiceException):
    logger.error(f"[GenosServiceException]: {exc.error_msg}")
    body = {'code': exc.error_code, 'errMsg': exc.error_msg, 'data': None, 'error_code': exc.error_code}
    # #329: strict 경로의 stage/error_kind 를 envelope 에 실어 준다(있을 때만).
    if getattr(exc, 'stage', None) is not None:
        body['stage'] = exc.stage
    if getattr(exc, 'error_type', None) is not None:
        body['error_kind'] = exc.error_type
    return JSONResponse(body, status_code=200)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    logger.error(f'[RequestValidationError]: {exc.errors()}')
    return make_failure_response(str(exc))


@app.exception_handler(Exception)
async def exception_handler(request, exc: Exception):
    logger.error(f'[Exception]: {exc}')
    return make_failure_response(str(exc))


@app.get('/healthcheck')
async def healthcheck() -> object:
    return {'status': 'ok'}


if settings.PREPROCESSOR_ID:
    download_resource_files(
        bucket_name='preprocessor',
        resource_id=settings.PREPROCESSOR_ID,
        path='/app/resource',
    )

# 이 파일 마운트
from preprocessor import DocumentProcessor

processor = DocumentProcessor()


def _request_deadline_seconds(params: dict):
    """#329: params.request_deadline(초, >0)이면 요청 전체 hard deadline 으로 쓴다.

    LLM 호출 단위 timeout 은 facade 내부(llm_cache.remaining_timeout)에서 이미 적용되며,
    이 값은 그 위에 씌우는 요청 전체 상한(비-LLM 행잉 방어)이다. 미설정이면 None(무제한).
    """
    try:
        secs = float(params.get('request_deadline'))
    except (TypeError, ValueError):
        return None
    return secs if secs > 0 else None


async def _run_with_deadline(request, file_path, params):
    """processor 호출을 요청 deadline 으로 감싼다. 초과 시 timeout 성격의 실패로 매핑."""
    rd = _request_deadline_seconds(params)
    if rd is None:
        return await processor(request, file_path, **params)
    return await asyncio.wait_for(processor(request, file_path, **params), timeout=rd)


@app.post('/run')
async def run(
        request: Request,
        file_path: str = Body(..., embed=True),
        params: dict = Body(default_factory=dict)
):
    pt = time.time()
    try:
        logger.info(f'Start: "{file_path}"')
        data = await _run_with_deadline(request, file_path, params)
        logger.info(f'Success: "{file_path}"')
    except asyncio.TimeoutError:
        logger.error(f'Error(timeout): "{file_path}"')
        return make_failure_response('request deadline exceeded', error_code=1,
                                     stage='request', error_kind='timeout')
    except GenosServiceException as e:
        logger.error(f'Error: "{file_path}"\n{traceback.format_exc()}\n')
        return failure_response_from_exc(e)
    except Exception as e:
        logger.error(f'Error: "{file_path}"\n{traceback.format_exc()}\n')
        return failure_response_from_exc(e)
    finally:
        logger.info(f'End: "{file_path}" ({time.time() - pt:.2f} seconds)')
    return make_success_response(data=data)


@app.post('/parser')
async def parse(
        request: Request,
        file_path: str = Body(..., embed=True),
        params: dict = Body(default_factory=dict)
):
    if not getattr(processor, 'IS_PARSER', False):
        return JSONResponse(
            {'code': 1,
             'errMsg': '현재 설치된 전처리기는 /parser API를 지원하지 않습니다.',
             'data': None,
             'error_code': 1,
             'error_msg': '현재 설치된 전처리기는 /parser API를 지원하지 않습니다.'},
            status_code=200)
    pt = time.time()
    try:
        logger.info(f'[parser] Start: "{file_path}"')
        data = await _run_with_deadline(request, file_path, params)
        logger.info(f'[parser] Success: "{file_path}"')
    except asyncio.TimeoutError:
        logger.error(f'[parser] Error(timeout): "{file_path}"')
        return make_failure_response('request deadline exceeded', error_code=1,
                                     stage='request', error_kind='timeout')
    except GenosServiceException as e:
        logger.error(f'[parser] Error: "{file_path}"\n{traceback.format_exc()}\n')
        return failure_response_from_exc(e)
    except Exception as e:
        logger.error(f'[parser] Error: "{file_path}"\n{traceback.format_exc()}\n')
        return failure_response_from_exc(e)
    finally:
        logger.info(f'[parser] End: "{file_path}" ({time.time() - pt:.2f} seconds)')
    return make_success_response(data=data)


@app.post('/chunker')
async def chunker(
        request: Request,
        file_path: str = Body(default='', embed=True),
        params: dict = Body(default_factory=dict)
):
    if not getattr(processor, 'IS_CHUNKER', False):
        return JSONResponse(
            {'code': 1,
             'errMsg': '현재 설치된 전처리기는 /chunker API를 지원하지 않습니다.',
             'data': None,
             'error_code': 1,
             'error_msg': '현재 설치된 전처리기는 /chunker API를 지원하지 않습니다.'},
            status_code=200)
    pt = time.time()
    try:
        logger.info('[chunker] Start')
        # 앞단계(파싱) 결과 docling JSON 은 params["document"] 로 인라인 전달된다.
        data = await _run_with_deadline(request, file_path, params)
        logger.info('[chunker] Success')
    except asyncio.TimeoutError:
        logger.error('[chunker] Error(timeout)')
        return make_failure_response('request deadline exceeded', error_code=1,
                                     stage='request', error_kind='timeout')
    except GenosServiceException as e:
        logger.error(f'[chunker] Error\n{traceback.format_exc()}\n')
        return failure_response_from_exc(e)
    except Exception as e:
        logger.error(f'[chunker] Error\n{traceback.format_exc()}\n')
        return failure_response_from_exc(e)
    finally:
        logger.info(f'[chunker] End ({time.time() - pt:.2f} seconds)')
    return make_success_response(data=data)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('main:app', host='0.0.0.0', port=7084, reload=True)
