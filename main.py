import os
import sys
import json
import shutil
import asyncio
import tempfile
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

from fastapi import FastAPI, Request, Body, UploadFile, File, Form
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from logger import Logger
from utils import make_success_response
from config import cors_config
from common.exception import GenosServiceException
from common.settings import settings
from util.minio_resource import download_resource_files

sys.path.append(os.path.dirname(__file__) + '/util')

logger = Logger.getLogger(__name__)

app: FastAPI = FastAPI()
cors_config(app)


# ── 에러 응답 ────────────────────────────────────────────────────────────
# 에러 분류 코드 — 일반 예외(Python 빌트인 등)를 카테고리로 매핑한다.
# GenosServiceException 은 facade 가 부여한 error_code 를 그대로 보존한다.
ERROR_CODE_INPUT = 'INPUT_ERROR'        # 잘못된 입력/파일 (FileNotFound, Value, Key, Type ...)
ERROR_CODE_TIMEOUT = 'TIMEOUT_ERROR'    # 타임아웃
ERROR_CODE_INTERNAL = 'INTERNAL_ERROR'  # 그 외 내부 오류

_INPUT_EXC = (FileNotFoundError, IsADirectoryError, NotADirectoryError,
              PermissionError, ValueError, KeyError, TypeError, IndexError)
_TRACEBACK_TAIL_LINES = 8  # 응답에 포함할 traceback 마지막 N 줄 (운영 디버깅용 요약)


def _classify_error(exc: Exception) -> str:
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return ERROR_CODE_TIMEOUT
    if isinstance(exc, _INPUT_EXC):
        return ERROR_CODE_INPUT
    return ERROR_CODE_INTERNAL


def _error_response(tag: str, file_path: str, exc: Exception, error_code=None, stage=None) -> JSONResponse:
    """모든 에러 경로의 응답 형태를 통일한다.

    code 는 항상 1(실패 플래그), 기존 키(errMsg/error_code/error_msg/data)는 유지하고
    컨텍스트(error_type/tag/file_path) 와 traceback 요약을 추가로 담는다.
    """
    etype = type(exc).__name__
    raw_msg = getattr(exc, 'error_msg', None) or str(exc) or etype
    if error_code is None:
        error_code = _classify_error(exc)
    # errMsg: 사람이 보는 메시지에 컨텍스트(엔드포인트·예외타입) 보강
    err_msg = f'[{tag}] {etype}: {raw_msg}'
    tb = traceback.format_exc()
    tb_tail = (''.join(tb.splitlines(keepends=True)[-_TRACEBACK_TAIL_LINES:])
               if tb and not tb.startswith('NoneType: None') else '')
    body = {
        'code': 1,
        'errMsg': err_msg,
        'error_msg': err_msg,
        'error_code': error_code,
        'error_type': etype,      # 예외 클래스명(기존 의미 보존)
        'tag': tag,               # 실패한 엔드포인트/단계
        'file_path': file_path,   # 대상 파일
        'data': None,
        'traceback': tb_tail,     # traceback 마지막 N 줄 요약
    }
    # #329: facade 가 부여한 실패 단계(stage)와 성격(error_kind: transient/permanent/timeout)을
    # caller(Temporal activity)가 알 수 있게 노출(있을 때만). 기존 error_type(클래스명)은 보존하고
    # 스펙의 error_type 값은 명명 충돌을 피해 error_kind 로 싣는다.
    fac_stage = getattr(exc, 'stage', None) or stage
    if fac_stage is not None:
        body['stage'] = fac_stage
    fac_kind = getattr(exc, 'error_type', None)
    if fac_kind is None and isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        fac_kind = 'timeout'   # facade 값이 없는 요청-레벨/일반 timeout 도 성격을 명시
    if fac_kind is not None:
        body['error_kind'] = fac_kind
    return JSONResponse(body, status_code=200)


@app.exception_handler(GenosServiceException)
async def mlops_exception_handler(request, exc: GenosServiceException):
    logger.error(f"[GenosServiceException]: {exc.error_msg}")
    return _error_response('app', '', exc, error_code=exc.error_code)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    logger.error(f'[RequestValidationError]: {exc.errors()}')
    return _error_response('app', '', exc, error_code=ERROR_CODE_INPUT)


@app.exception_handler(Exception)
async def exception_handler(request, exc: Exception):
    logger.error(f'[Exception]: {exc}')
    return _error_response('app', '', exc)


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


def _request_deadline_seconds(params: dict):
    """#329: params.request_deadline(초, >0)이면 요청 전체 hard deadline 으로 쓴다.

    LLM 호출 단위 timeout 은 facade 내부(llm_cache.remaining_timeout, CacheContext.deadline)에서
    이미 적용되며, 이 값은 그 위에 씌우는 요청 전체 상한(비-LLM 행잉 방어)이다. 미설정이면 None(무제한).
    """
    try:
        secs = float(params.get('request_deadline'))
    except (TypeError, ValueError):
        return None
    return secs if secs > 0 else None


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
        # #329: 요청 전체 deadline(params.request_deadline) 이 있으면 행잉 대신 timeout 응답.
        rd = _request_deadline_seconds(params)
        if rd is None:
            data = await processor(request, file_path, **params)
        else:
            data = await asyncio.wait_for(processor(request, file_path, **params), timeout=rd)
        logger.info(f'[{tag}] Success: "{file_path}"')
        return make_success_response(data=data)
    except asyncio.TimeoutError as e:
        logger.error(f'[{tag}] Error(timeout): "{file_path}" (request_deadline exceeded)')
        return _error_response(tag, file_path, e, error_code=ERROR_CODE_TIMEOUT, stage='request')
    except GenosServiceException as e:
        logger.error(f'[{tag}] Error: "{file_path}"\n{traceback.format_exc()}\n')
        return _error_response(tag, file_path, e, error_code=e.error_code)  # facade 코드 보존
    except Exception as e:
        logger.error(f'[{tag}] Error: "{file_path}"\n{traceback.format_exc()}\n')
        return _error_response(tag, file_path, e)  # 타입 기반 자동 분류
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


# 코드서빙 게이트웨이({base}/api/gateway/code_serving/{id}/{route})는 route 를 단일 세그먼트로만
# 포워딩하므로, 슬래시가 포함된 중첩 경로(/preprocess/xxx)는 게이트웨이로 호출되지 않는다.
# 따라서 /parser·/chunker 처럼 평탄(단일 세그먼트) 경로(/preprocess_xxx)로 노출한다.

@app.post('/preprocess_attachment')
async def preprocess_attachment(
        request: Request,
        file_path: str = Body(..., embed=True),
        params: dict = Body(default_factory=dict)
):
    return await _run('preprocess_attachment', attachment_processor, request, file_path, params)


@app.post('/preprocess_intelligent')
async def preprocess_intelligent(
        request: Request,
        file_path: str = Body(..., embed=True),
        params: dict = Body(default_factory=dict)
):
    return await _run('preprocess_intelligent', intelligent_processor, request, file_path, params)


@app.post('/preprocess_convert')
async def preprocess_convert(
        request: Request,
        file_path: str = Body(..., embed=True),
        params: dict = Body(default_factory=dict)
):
    return await _run('preprocess_convert', convert_processor, request, file_path, params)


@app.post('/parser')
async def parse(
        request: Request,
        file_path: str = Body(..., embed=True),
        params: dict = Body(default_factory=dict)
):
    return await _run('parser', parser_processor, request, file_path, params, marker='IS_PARSER')


# /parser 의 multipart 변형: 클라이언트 로컬 파일을 업로드받아 파싱한다.
# 기존 /parser(JSON file_path)는 그대로 두고, 업로드 바이트를 임시 파일로 저장한 뒤
# 그 경로를 동일한 parser_processor 에 넘겨 파서 내부 로직을 그대로 재사용한다.
# parser 는 확장자로 형식을 판단하므로 업로드 원본 파일명의 확장자를 보존한다.
@app.post('/parser_upload')
async def parse_upload(
        request: Request,
        file: UploadFile = File(...),
        params: str = Form('{}'),
):
    # params 는 multipart 폼 특성상 JSON 문자열로 받는다. 파싱 실패는 입력 오류로 처리.
    try:
        params_dict = json.loads(params) if params else {}
        if not isinstance(params_dict, dict):
            raise ValueError('params 는 JSON 객체여야 합니다.')
    except (ValueError, TypeError) as e:
        return _error_response('parser_upload', file.filename or '', e, error_code=ERROR_CODE_INPUT)

    # 원본 파일명/확장자 보존 — 확장자가 파서의 형식 라우팅 기준이다.
    safe_name = os.path.basename(file.filename or 'upload')
    if not os.path.splitext(safe_name)[1]:
        safe_name = 'upload'  # 확장자 없는 파일명은 그대로 두되 basename 만 사용
    tmp_dir = tempfile.mkdtemp(prefix='parser_upload_')
    tmp_path = os.path.join(tmp_dir, safe_name)
    try:
        contents = await file.read()
        with open(tmp_path, 'wb') as f:
            f.write(contents)
        return await _run('parser_upload', parser_processor, request, tmp_path, params_dict, marker='IS_PARSER')
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
