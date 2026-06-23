import os
import sys
import traceback
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# Put preprocessor src ahead of /app/src to avoid collisions like common.settings.
for module_path in (BASE_DIR / 'genon' / 'preprocessor' / 'src',):
    if module_path.is_dir():
        module_path_str = str(module_path)
        while module_path_str in sys.path:
            sys.path.remove(module_path_str)
        sys.path.insert(0, module_path_str)

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


@app.get('/health')
async def health() -> object:
    return {'status': 'ok'}


if settings.PREPROCESSOR_ID:
    download_resource_files(
        bucket_name='preprocessor',
        resource_id=settings.PREPROCESSOR_ID,
        path='/app/resource',
    )

from genon.preprocessor.facade.attachment_processor import DocumentProcessor as AttachmentDocumentProcessor
from genon.preprocessor.facade.intelligent_processor import DocumentProcessor as IntelligentDocumentProcessor
from genon.preprocessor.facade.convert_processor import DocumentProcessor as ConvertDocumentProcessor

from genon.preprocessor.facade.parser_processor import DocumentProcessor as ParserDocumentProcessor
from genon.preprocessor.facade.chunking_processor import DocumentProcessor as ChunkingDocumentProcessor

# config 는 resource/ 로 고정한다. (무인자 생성 시 facade 기본 해석기가 resource_dev/ 를
# 우선하므로, resource_dev 유무와 무관하게 항상 출고용 resource/ 를 읽도록 config_path 를 명시.)
# resource_dev 로 테스트하려면 아래 "resource" 를 "resource_dev" 로만 바꾸면 된다.
RESOURCE_DIR = BASE_DIR / "genon" / "preprocessor" / "resource"


def _cfg(name: str) -> str:
    return str(RESOURCE_DIR / f"{name}_processor_config.yaml")


# 프로세서는 모듈 로딩 시 1회만 생성해 재사용한다(요청마다 재생성하면 config/토크나이저/
# 파이프라인 초기화 비용이 반복됨). 각 프로세서는 resource/<name>_processor_config.yaml 을 로드한다.
attachment_processor = AttachmentDocumentProcessor(config_path=_cfg("attachment"))    # 첨부용
intelligent_processor = IntelligentDocumentProcessor(config_path=_cfg("intelligent"))  # 적재용(지능형)
convert_processor = ConvertDocumentProcessor(config_path=_cfg("convert"))             # 변환용
parser_processor = ParserDocumentProcessor(config_path=_cfg("parser"))               # 파싱 전용(/parser)
chunking_processor = ChunkingDocumentProcessor(config_path=_cfg("chunking"))         # 청킹 전용(/chunker)


async def _run(tag, processor, request, file_path, params, marker=None):
    """엔드포인트 공통 실행 래퍼: 마커 가드 + 로깅 + 예외 처리 + 응답 포맷."""
    if marker and not getattr(processor, marker, False):
        msg = f'현재 설치된 전처리기는 /{tag} API를 지원하지 않습니다.'
        return JSONResponse(
            {'code': 1, 'errMsg': msg, 'data': None, 'error_code': 1, 'error_msg': msg},
            status_code=200)
    pt = time.time()
    try:
        logger.info(f'[{tag}] Start: "{file_path}"')
        data = await processor(request, file_path, **params)
        logger.info(f'[{tag}] Success: "{file_path}"')
        return make_success_response(data=data)
    except GenosServiceException as e:
        logger.error(f'[{tag}] Error: "{file_path}"\n{traceback.format_exc()}\n')
        return JSONResponse(
            {'code': 1, 'errMsg': e.error_msg, 'data': None,
             'error_code': e.error_code, 'error_msg': e.error_msg},
            status_code=200)
    except Exception as e:
        logger.error(f'[{tag}] Error: "{file_path}"\n{traceback.format_exc()}\n')
        return make_failure_response(str(e))
    finally:
        logger.info(f'[{tag}] End: "{file_path}" ({time.time() - pt:.2f} seconds)')


# ── 적재 프로세서: 프로세서별 별도 엔드포인트 ──────────────────────────────
# /preprocess 는 하위호환을 위해 intelligent 의 별칭으로 유지한다.

@app.post('/preprocess')
async def preprocess(
        request: Request,
        file_path: str = Body(..., embed=True),
        params: dict = Body(default_factory=dict)
):
    return await _run('preprocess', intelligent_processor, request, file_path, params)


@app.post('/preprocess/attachment')
async def preprocess_attachment(
        request: Request,
        file_path: str = Body(..., embed=True),
        params: dict = Body(default_factory=dict)
):
    return await _run('preprocess/attachment', attachment_processor, request, file_path, params)


@app.post('/preprocess/intelligent')
async def preprocess_intelligent(
        request: Request,
        file_path: str = Body(..., embed=True),
        params: dict = Body(default_factory=dict)
):
    return await _run('preprocess/intelligent', intelligent_processor, request, file_path, params)


@app.post('/preprocess/convert')
async def preprocess_convert(
        request: Request,
        file_path: str = Body(..., embed=True),
        params: dict = Body(default_factory=dict)
):
    return await _run('preprocess/convert', convert_processor, request, file_path, params)


@app.post('/parser')
async def parse(
        request: Request,
        file_path: str = Body(..., embed=True),
        params: dict = Body(default_factory=dict)
):
    return await _run('parser', parser_processor, request, file_path, params, marker='IS_PARSER')


@app.post('/chunker')
async def chunker(
        request: Request,
        file_path: str = Body(default='', embed=True),
        params: dict = Body(default_factory=dict)
):
    # 앞단계(파싱) 결과 docling JSON 은 params["document"] 로 인라인 전달된다.
    return await _run('chunker', chunking_processor, request, file_path, params, marker='IS_CHUNKER')


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('main:app', host='0.0.0.0', port=7084, reload=True)
