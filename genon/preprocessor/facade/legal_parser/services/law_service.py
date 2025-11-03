import asyncio
import time

from api.law_client import get_api_amend, get_api_response
from commons.file_handler import export_json
from commons.loggers import ErrorLogger, MainLogger
from parsers.addendum import parse_addendum_info
from parsers.admrule import parse_admrule_info
from parsers.admrule_article import parse_admrule_article_info
from parsers.appendix import parse_appendix_info
from parsers.law import parse_law_info
from parsers.law_article import parse_law_article_info
from parsers.law_system import parse_law_relationships
from parsers.mapper import processor_mapping
from schemas.law_schema import ConnectedLaws, HierarchyLaws, RuleInfo
from schemas.params import (
    AdmRuleRequestParams,
    LawItemRequestParams,
    LawSystemRequestParams,
    UpdatedLawRequestParams,
)
from schemas.schema import (
    ParserContent,
    ParserRequest,
    ParserResponse,
    ParserResult,
)

# 로거 인스턴스 초기화
error_logger = ErrorLogger.instance()
main_logger = MainLogger.instance()

async def get_parsed_law_system(id: str) -> tuple[list[HierarchyLaws], list[ConnectedLaws], dict[str, list[str]]]:
    """
    법령 체계도 데이터를 가져와 상하위 법령 및 관련 법령 정보를 처리하는 함수.

    Args:
        id (str): 법령 ID.

    Returns:
        tuple: 상하위 법령, 관련 법령, 상하위법 및 관련 법령 ID 리스트.
    """
    main_logger.info("[get_law_system] 상하위법 및 관련법령 데이터 처리 시작")
    system_response: dict = await get_api_response(LawSystemRequestParams(MST=id))
    law_system = system_response.get("법령체계도")
    return parse_law_relationships(law_system)


async def get_parsed_law(id: str, hierarchy_laws: list[HierarchyLaws], connected_laws: list[ConnectedLaws]) -> ParserResult:
    """
    법령 데이터를 파싱하여 결과를 반환하는 함수.

    Args:
        id (str): 법령 ID.
        hierarchy_laws (list): 상하위 법령 데이터.
        connected_laws (list): 관련 법령 데이터.

    Returns:
        ParserResult: 파싱 결과 데이터.
    """
    # 상하위 법령 및 관련 법령 데이터 초기화
    hierarchy_laws = hierarchy_laws if isinstance(hierarchy_laws, list) else []
    connected_laws = connected_laws if isinstance(connected_laws, list) else []

    # 법령 데이터 API 호출
    law_response: dict = await get_api_response(LawItemRequestParams(MST=id))

    # 법령 메타데이터 처리
    main_logger.info(f"[parse_law_info] 법령 메타데이터 처리: id={id}")
    law_data: dict = law_response.get("법령")
    law_result: ParserContent = parse_law_info(id, law_data, hierarchy_laws, connected_laws)

    is_admrule = False  # 행정규칙 여부 설정
    
    # 법령 정보 객체 생성
    law_info = RuleInfo(
        id, law_result.metadata.enforce_date, law_result.metadata.enact_date, law_result.metadata.is_effective
    )

    # 부칙, 별표, 조문 데이터 병렬 처리
    main_logger.info("[get_parsed_law] 법령 부칙, 별표, 조문 데이터 병렬 처리 시작")
    addendum_data = law_data.get("부칙")
    appendix_data = law_data.get("별표")
    article_data = law_data.get("조문")

    # 병렬 작업 생성
    addendum_task = asyncio.to_thread(parse_addendum_info, id, addendum_data, is_admrule)
    appendix_task = asyncio.to_thread(parse_appendix_info, law_info, appendix_data, is_admrule)
    article_task = asyncio.to_thread(parse_law_article_info, law_info, article_data)

    # 병렬 작업 실행 및 결과 수집
    addendum_list, appendix_list, article_list = await asyncio.gather(
        addendum_task, appendix_task, article_task
    )

    # 조문, 부칙, 별표 연결 처리
    main_logger.info("[processor_mapping] 법령 조문 - 부칙 - 별표 연결 처리 시작")
    article_result, addendum_result, appendix_result = processor_mapping(article_list, addendum_list, appendix_list)
    
    # 파싱 결과 생성
    parse_result = ParserResult(
        law=law_result,
        addendum=addendum_result,
        appendix=appendix_result,
        article=article_result,
    )
    return parse_result


async def get_parsed_admrule(id: str, hierarchy_laws: list[HierarchyLaws], connected_laws: list[ConnectedLaws]) -> ParserResult:
    """
    행정규칙 데이터를 파싱하여 결과를 반환하는 함수.

    Args:
        id (str): 행정규칙 ID.
        hierarchy_laws (list): 상하위 법령 데이터.
        connected_laws (list): 관련 법령 데이터.

    Returns:
        ParserResult: 파싱 결과 데이터.
    """
    # 상하위 법령 및 관련 법령 데이터 초기화
    hierarchy_laws = hierarchy_laws if isinstance(hierarchy_laws, list) else []
    connected_laws = connected_laws if isinstance(connected_laws, list) else []
        
    # 행정규칙 데이터 API 호출
    admrule_response: dict = await get_api_response(AdmRuleRequestParams(ID=id))
    admrule_data = admrule_response.get("AdmRulService")

    # 행정규칙 메타데이터 처리
    main_logger.info(f"[parse_admrule_info] 행정규칙 메타데이터 처리 시작: id={id}")
    admrule_result: ParserContent = parse_admrule_info(
        id, admrule_data, hierarchy_laws, connected_laws
    )

    admrule_info = RuleInfo(
        id,
        admrule_result.metadata.enforce_date,
        admrule_result.metadata.enact_date,
        admrule_result.metadata.is_effective,
    )

    is_admrule = True  # 행정규칙 여부 설정

    # 부칙, 별표, 조문 데이터 병렬 처리
    main_logger.info("[get_parsed_admrule] 행정규칙 부칙, 별표, 조문 데이터 병렬 처리 시작")
    addendum_data = admrule_data.get("부칙")
    appendix_data = admrule_data.get("별표", {})
    article_data = admrule_data.get("조문내용", [])

    # 병렬 작업 생성
    addendum_task = asyncio.to_thread(parse_addendum_info, id, addendum_data, is_admrule)
    appendix_task = asyncio.to_thread(parse_appendix_info, admrule_info, appendix_data, is_admrule)
    article_task = asyncio.to_thread(parse_admrule_article_info, admrule_info, article_data)

    # 병렬 작업 실행 및 결과 수집
    addendum_list, appendix_list, article_list = await asyncio.gather(
        addendum_task, appendix_task, article_task
    )
    
    # 조문, 부칙, 별표 연결 처리
    main_logger.info("[processor_mapping] 행정규칙 조문 - 부칙 - 별표 연결 처리 시작")
    article_result, addendum_result, appendix_result = processor_mapping(article_list, addendum_list, appendix_list)
   
    # 파싱 결과 생성
    parse_result = ParserResult(
        law=admrule_result,
        article=article_result,
        addendum=addendum_result,
        appendix=appendix_result,
    )
    
    main_logger.info("[get_parsed_admrule] 행정규칙 데이터 파싱 완료")
    return parse_result


async def process_with_error_handling(
        id: str,
        process_func: callable,
        hierarchy_laws: list[HierarchyLaws],
        connected_laws: list[ConnectedLaws],
        response: ParserResponse,
    ) -> bool:
    """
    파싱 작업 중 에러를 처리하는 함수.

    Args:
        id (str): 처리할 ID.
        process_func (callable): 처리 함수.
        hierarchy_laws (list): 상하위 법령 데이터.
        connected_laws (list): 관련 법령 데이터.
        response (ParserResponse): 응답 객체.

    Returns:
        bool: 성공 여부.
    """
    try:
        response.increment_total()
        await process_func(id, hierarchy_laws, connected_laws)
    except Exception as e:
        main_logger.error(f"[get_parse_result] 파싱 실패: ID={id} 자세한 내용은 로그 파일을 확인하세요.")
        error_logger.law_error(id, e)

        response.increment_fail(id)
        return False
    else:
        response.increment_success()
        return True


async def get_parse_result(request: ParserRequest) -> ParserResponse:
    """
    주어진 요청에 따라 법령 및 행정규칙 데이터를 파싱하는 함수.

    Args:
        request (ParserRequest): 파싱 요청 객체.

    Returns:
        ParserResponse: 파싱 결과 응답 객체.
    """
    law_consecutive_fail = 0
    admrule_consecutive_fail = 0

    seen_law_id = set()
    seen_admrule_id = set()

    async def process_law(id: str, hierarchy_laws, connected_laws):
        """
        개별 법령 데이터를 처리하는 내부 함수.
        """
        if id in seen_law_id:
            return
        seen_law_id.add(id)
        
        main_logger.info(f"[get_parse_result] 법령 데이터 파싱 시작: ID={id}")
        start = time.perf_counter()         
        
        parse_result = await get_parsed_law(id, hierarchy_laws, connected_laws)
        export_json(parse_result.model_dump(), id, parse_result.law.metadata.law_num, False)
    
        end = time.perf_counter()
        duration = end - start
        main_logger.info(f"[get_parse_result] 법령 파싱 완료 - ID={id}, 소요 시간: {duration:.2f}초\n")

    async def process_admrule(id: str, hierarchy_laws, connected_laws):
        """
        개별 행정규칙 데이터를 처리하는 내부 함수.
        """
        if id in seen_admrule_id:
            return
        seen_admrule_id.add(id)
        
        main_logger.info(f"[get_parse_result] 행정규칙 데이터 파싱 시작: ID={id}")
        start = time.perf_counter()

        parse_result = await get_parsed_admrule(id, hierarchy_laws, connected_laws)
        export_json(parse_result.model_dump(), id, parse_result.law.metadata.admrule_num)
    
        end = time.perf_counter()
        duration = end - start
        main_logger.info(f"[get_parse_result] 행정규칙 파싱 완료 - ID={id}, 소요 시간: {duration:.2f}초\n")
    
    law_ids = request.law_ids
    admrule_ids = request.admrule_ids

    response = ParserResponse()

    for law_id in law_ids:
        if law_consecutive_fail >= 10:
            main_logger.critical("법령 - 연속된 에러 10회 초과로 Parser 실행 중단")
            break
        if law_id not in seen_law_id:
            try:
                hierarchy_laws, connected_laws, related_law_ids = await get_parsed_law_system(law_id)
            except Exception as e:
                main_logger.error(f"[get_law_system] 법령 관계도 파싱 실패: ID={law_id} 자세한 내용은 로그 파일을 확인하세요.")
                error_logger.law_error(law_id, e)

                response.increment_fail(law_id)
                law_consecutive_fail += 1
                continue
            else:
                law_consecutive_fail = 0

            for related_id in related_law_ids.get("law", []):
                if law_consecutive_fail >= 10:
                    main_logger.critical("법령 - 연속된 에러 10회 초과로 Parser 실행 중단")
                    break
                success = await process_with_error_handling(related_id, process_law, hierarchy_laws, connected_laws, response)
                if success:
                    law_consecutive_fail = 0
                else:
                    law_consecutive_fail += 1
                
            for related_id in related_law_ids.get("admrule", []):
                if admrule_consecutive_fail >= 10:
                    main_logger.critical("행정규칙 - 연속된 에러 10회 초과로 Parser 실행 중단")
                    break
                success = await process_with_error_handling(related_id, process_admrule, hierarchy_laws, connected_laws, response)
                if success:
                    admrule_consecutive_fail = 0
                else:
                    admrule_consecutive_fail += 1

    for admrule_id in admrule_ids:
        if admrule_consecutive_fail >= 10:
            main_logger.critical("행정규칙 - 연속된 에러 10회 초과로 Parser 실행 중단")
            break
        if admrule_id not in seen_admrule_id:
            success = await process_with_error_handling(admrule_id, process_admrule, [], [], response)
            if success:
                admrule_consecutive_fail = 0
            else:
                admrule_consecutive_fail += 1

    seen_count = len(seen_law_id) + len(seen_admrule_id)
    response.seen_ids = {
        "law": seen_law_id,
        "admrule": seen_admrule_id,
    }
    response.unseen_count = response.total_count - seen_count
    response.seen_count = seen_count
    response.unseen_ids = {
        "law": set(law_ids) - seen_law_id,
        "admrule": set(admrule_ids) - seen_admrule_id,
    }

    main_logger.info(f"\n[get_parse_result] 데이터 파싱 완료: {response.model_dump()}")
    return response


async def get_amend_result() -> ParserResponse:
    """
    어제 개정된 법령 데이터를 파싱하는 함수.

    Returns:
        ParserResponse: 개정된 법령 파싱 결과.
    """
    query = UpdatedLawRequestParams()
    api_response: dict = await get_api_amend(query)

    laws = api_response.get("LawSearch", {}).get("law", [])
    total = api_response.get("LawSearch", {}).get("totalCnt", 0)

    if total:
        law_ids = [law.get("법령일련번호") for law in laws]
        main_logger.info(f"[get_amend_result]: 개정된 법령 ID - {law_ids}. 개정된 법령의 파싱을 시작합니다.")
        
        updated_law_ids = ParserRequest()
        updated_law_ids.law_ids = law_ids

        return await get_parse_result(updated_law_ids)
    else:
        main_logger.info("[get_updated_result] 개정된 법령이 존재하지 않습니다.")
        return ParserResponse()