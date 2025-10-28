from fastapi import FastAPI, Request, Body
from fastapi.exceptions import RequestValidationError

import traceback
import time
from logger import Logger
from utils import make_failure_response, make_success_response
from config import cors_config
from common.exception import GenosServiceException
from fastapi.responses import JSONResponse

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


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('main:app', host='0.0.0.0', port=7084, reload=True)
