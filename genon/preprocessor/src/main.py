import os
import sys
import traceback
import time

from fastapi import FastAPI, Request, Body
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from logger import Logger
from utils import make_failure_response, make_success_response
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
    return JSONResponse({'code': exc.error_code, 'errMsg': exc.error_msg, 'data': None, 'error_code': exc.error_code},
                        status_code=200)


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


@app.post('/run')
async def run(
        request: Request,
        file_path: str = Body(..., embed=True),
        params: dict = Body(default_factory=dict)
):
    pt = time.time()
    try:
        logger.info(f'Start: "{file_path}"')
        data = await processor(request, file_path, **params)
        logger.info(f'Success: "{file_path}"')
    except GenosServiceException as e:
        logger.error(f'Error: "{file_path}"\n{traceback.format_exc()}\n')
        return JSONResponse(
            {'code': 1, 'errMsg': e.error_msg, 'data': None, 'error_code': e.error_code,
             'error_msg': e.error_msg},
            status_code=200)
    except Exception as e:
        logger.error(f'Error: "{file_path}"\n{traceback.format_exc()}\n')
        return make_failure_response(str(e))
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
        data = await processor(request, file_path, **params)
        logger.info(f'[parser] Success: "{file_path}"')
    except GenosServiceException as e:
        logger.error(f'[parser] Error: "{file_path}"\n{traceback.format_exc()}\n')
        return JSONResponse(
            {'code': 1, 'errMsg': e.error_msg, 'data': None,
             'error_code': e.error_code, 'error_msg': e.error_msg},
            status_code=200)
    except Exception as e:
        logger.error(f'[parser] Error: "{file_path}"\n{traceback.format_exc()}\n')
        return make_failure_response(str(e))
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
        data = await processor(request, file_path, **params)
        logger.info('[chunker] Success')
    except GenosServiceException as e:
        logger.error(f'[chunker] Error\n{traceback.format_exc()}\n')
        return JSONResponse(
            {'code': 1, 'errMsg': e.error_msg, 'data': None,
             'error_code': e.error_code, 'error_msg': e.error_msg},
            status_code=200)
    except Exception as e:
        logger.error(f'[chunker] Error\n{traceback.format_exc()}\n')
        return make_failure_response(str(e))
    finally:
        logger.info(f'[chunker] End ({time.time() - pt:.2f} seconds)')
    return make_success_response(data=data)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('main:app', host='0.0.0.0', port=7084, reload=True)
