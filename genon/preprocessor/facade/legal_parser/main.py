from fastapi import Depends, FastAPI, Query

from commons.file_handler import load_keys_from_csv
from schemas.params import AdmRuleRequestParams, LawItemRequestParams
from schemas.schema import ParserRequest, ParserResponse, PipelineResponse
from schemas.vdb_schema import VDBResponse
from services.download_service import download_data
from services.law_service import get_parse_result
from services.service import (
    process_all_pipeline,
    process_updated_pipeline,
)
from services.vdb_service import process_law_vectorization

app = FastAPI()

@app.get("/")
async def read_root():
    return "Hello, Parser for Legal Data"

@app.get("/test/parser")
async def test_parser(
    i_start:int = Query(default=0),
    i_end:int = Query(default=5)
) -> ParserResponse:
    """
    특정 범위의 법령 ID를 테스트로 파싱하는 엔드포인트.

    Args:
        i_start (int): 시작 인덱스.
        i_end (int): 끝 인덱스.

    Returns:
        ParserResponse: 지정된 범위의 법령 ID에 대한 파싱 결과.
    """
    law_ids_dict = load_keys_from_csv()

    request = ParserRequest()
    request.law_ids = law_ids_dict.law_ids[i_start:i_end]
    request.admrule_ids = law_ids_dict.admrule_ids[i_start:i_end]

    return await get_parse_result(request)

@app.get("/test/vdb")
async def test_vdb(
) -> VDBResponse:
    """
    법령 데이터를 벡터화하는 테스트 엔드포인트.
    
    파일명 순으로 1~5번째 파일로 테스트
    vdb_info_{timestamp}.json이 생성되지 않음. 

    Returns:
        VDBResponse: 벡터화 처리 결과.
    """
    return await process_law_vectorization(is_test=True)

@app.get("/parse/all")
async def run_parser() -> PipelineResponse:
    """
    전체 법령 데이터를 대상으로 파싱 파이프라인을 실행하는 엔드포인트.

    Returns:
        PipelineResponse: 전체 파싱 파이프라인 결과.
    """
    request = load_keys_from_csv()
    return await process_all_pipeline(request)

@app.get("/parse/updated")
async def run_updator() -> PipelineResponse:
    """
    개정된 법령을 대상으로 파싱 파이프라인을 실행하는 엔드포인트.

    Returns:
        PipelineResponse: 업데이트된 데이터에 대한 파이프라인 결과.
    """
    return await process_updated_pipeline()
    
### OPEN API 법령 정보 원본 (JSON) 다운로드
@app.get("/download/law")
async def download_law(query: LawItemRequestParams = Depends()):
    """
    법령 데이터를 JSON 형식으로 다운로드하는 엔드포인트.
    constants.py에 DIR_PATH_LAW_INPUT으로 정의된 디렉토리 경로에 파일이 저장

    Args:
        query (LawItemRequestParams): 법령 데이터 다운로드를 위한 쿼리 매개변수.
    """
    await download_data(query)
    

@app.get("/download/admrule")
async def download_admrule(query: AdmRuleRequestParams = Depends()):
    """
    행정규칙 데이터를 JSON 형식으로 다운로드하는 엔드포인트.
    constants.py에 DIR_PATH_LAW_INPUT으로 정의된 디렉토리 경로에 파일이 저장

    Args:
        query (AdmRuleRequestParams): 행정규칙 데이터 다운로드를 위한 쿼리 매개변수.
    """
    await download_data(query)
