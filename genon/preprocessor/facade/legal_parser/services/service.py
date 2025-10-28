from commons.file_handler import export_mapping_json
from commons.loggers import ErrorLogger, MainLogger
from schemas.schema import (
    ParserRequest,
    ParserResponse,
    PipelineResponse,
)
from schemas.vdb_schema import VDBResponse
from services.law_service import get_amend_result, get_parse_result
from services.vdb_service import process_law_vectorization

# 로거 인스턴스 초기화
error_logger = ErrorLogger.instance()
main_logger = MainLogger.instance()

async def process_all_pipeline(request: ParserRequest) -> PipelineResponse:
    """
    전체 데이터를 대상으로 파싱 및 벡터화 파이프라인을 실행하는 함수.

    Args:
        request (ParserRequest): 파싱 요청 객체.

    Returns:
        PipelineResponse: 파싱 및 벡터화 결과를 포함한 응답 객체.
    """
    vdb_response = VDBResponse()
    mappings = []

    try:
        # 전체 파싱 시작
        main_logger.info("[process_all_pipeline] 전체 파싱 시작")
        parser_response: ParserResponse = await get_parse_result(request)

        # 파싱된 데이터가 없을 경우 처리 중단
        if parser_response.success_count == 0:
            main_logger.warning("[process_all_pipeline] 파싱된 데이터가 없습니다. 벡터화 중단.")
            return parser_response

        # 파싱 완료 후 벡터화 시작
        main_logger.info("[process_all_pipeline] 파싱 완료. VDB 업로드 시작")
        vdb_response = await process_law_vectorization()

        # 벡터화 결과를 JSON으로 저장
        mappings_dump = [m.model_dump() for m in vdb_response.mappings]
        export_mapping_json(mappings_dump)

        main_logger.info("[process_all_pipeline] VDB 업로드 완료. 파이프라인 종료")

    except Exception as e:
        # 파이프라인 실행 중 에러 처리
        error_logger.vdb_error("[run_vdb_vectorization_pipeline] 전체 파이프라인 실패", e)

    return PipelineResponse(parser=parser_response, vdb=vdb_response, mappings=mappings)

async def process_updated_pipeline() -> PipelineResponse:
    """
    개정된 데이터를 대상으로 파싱 및 벡터화 파이프라인을 실행하는 함수.

    Returns:
        PipelineResponse: 파싱 및 벡터화 결과를 포함한 응답 객체.
    """
    vdb_response = VDBResponse()

    try:
        # 개정 법령 파싱 시작
        main_logger.info("[process_updated_pipeline] 개정 법령 파싱 시작")
        parser_response: ParserResponse = await get_amend_result()

        # 파싱된 데이터가 없을 경우 처리 중단
        if parser_response.success_count == 0:
            main_logger.warning("[process_updated_pipeline] 파싱된 데이터가 없습니다. 벡터화 중단.")
            return PipelineResponse(parser=parser_response)
            
        # 파싱 완료 후 벡터화 시작
        main_logger.info("[process_updated_pipeline] 파싱 완료. VDB 업로드 시작")
        vdb_response = await process_law_vectorization()

        # 벡터화 결과를 JSON으로 저장
        mappings_dump = [m.model_dump() for m in vdb_response.mappings]
        export_mapping_json(mappings_dump)

        main_logger.info("[process_updated_pipeline] VDB 업로드 완료. 파이프라인 종료")

    except Exception as e:
        # 파이프라인 실행 중 에러 처리
        error_logger.vdb_error("[process_updated_pipeline] 개정 법령 처리 파이프라인 실패", e)

    return PipelineResponse(parser=parser_response, vdb=vdb_response)