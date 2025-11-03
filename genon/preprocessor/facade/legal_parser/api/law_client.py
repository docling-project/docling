import copy
import inspect
from typing import Awaitable, Callable, TypeVar, Union

import aiohttp
from fastapi import status

from commons.loggers import ErrorLogger, MainLogger
from schemas.params import (
    AdmBylRequestParams,
    AdmRuleRequestParams,
    APIEndpoints,
    LawItemRequestParams,
    LawSystemRequestParams,
    LicBylRequestParams,
    UpdatedLawRequestParams,
)

main_logger = MainLogger.instance()
error_logger = ErrorLogger.instance()

T = TypeVar('T')

class ClientError(Exception):
    """
    API 호출 실패 시 발생하는 사용자 정의 예외 클래스.

    Attributes:
        detail (str): 에러 메시지 상세 내용.
        status_code (int): HTTP 상태 코드 (기본값: 404).
    """
    def __init__(self, detail: str, status_code: int = status.HTTP_404_NOT_FOUND):
        self.detail = detail
        self.status_code = status_code

async def fetch_api(url: str):
    """
    주어진 URL로 API 요청을 보내고 JSON 또는 텍스트 형태의 응답을 반환합니다.

    Args:
        url (str): 호출할 API의 URL.

    Returns:
        Union[dict, str]: JSON 또는 텍스트 형태의 응답 데이터.

    Raises:
        ClientError: 예상치 못한 콘텐츠 타입 또는 응답 코드가 200이 아닌 경우.
    """
    async with aiohttp.ClientSession() as client:
        async with client.get(url) as response:
            content_type = response.headers.get("Content-Type", "")

            if response.status == 200:
                if "application/json" in content_type:
                    data = await response.json()
                    if "Law" in data.keys():
                        raise ClientError(detail="해당되는 법령/행정데이터를 찾을 수 없음 : {url}")
                else:
                    data = await response.text()
                    error_logger.law_error(f"[fetch_api] 예상치 못한 데이터 타입: {content_type} ({url})")
                    raise ClientError(detail=f"Unexpected content type: {url}: {content_type}", status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
            else:
                error_logger.law_error(f"[fetch_api] API 요청 실패: {url} (HTTP {response.status})")
                raise ClientError(detail=f"Request {url} failed with status {response.status}", status_code=status.HTTP_400_BAD_REQUEST)

            return data

async def fetch_api_amend(url: str):
    """
    개정 법령 API를 호출하고 JSON 응답을 반환합니다.

    Args:
        url (str): 호출할 API URL.

    Returns:
        dict: 응답 JSON 데이터.

    Raises:
        ClientError: 응답에 개정 법령이 없거나 응답 실패 시.
    """
    async with aiohttp.ClientSession() as client:
        async with client.get(url) as response:
            content_type = response.headers.get("Content-Type", "")
            if response.status == 200:
                if "application/json" in content_type:
                    data = await response.json()
                    if data and data["LawSearch"]["totalCnt"] == 0:
                        raise ClientError(detail="해당일자에 개정된 법령을 찾을 수 없음 : {url}")
                else:
                    data = await response.text()
                    error_logger.law_error(f"[fetch_api] 예상치 못한 응답 데이터 타입: {content_type} ({url})")
                    raise ClientError(detail=f"Unexpected content type: {url}: {content_type}", status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
            else:
                error_logger.law_error(f"[fetch_api] API 요청 실패: {url} (HTTP {response.status})")
                raise ClientError(detail=f"Request {url} failed with status {response.status}", status_code=status.HTTP_400_BAD_REQUEST)

            return data

async def get_api_response(
    query: Union[
        LawItemRequestParams,
        LawSystemRequestParams,
        LicBylRequestParams,
        AdmBylRequestParams,
        AdmRuleRequestParams
    ],
):
    """
    단일 API 요청을 수행하고 응답을 반환합니다.

    Args:
        query: API 요청 파라미터 객체.

    Returns:
        dict or str: API 응답 결과.
    """
    api_url = APIEndpoints().get_item_url(query.get_query_params())
    main_logger.debug(f"[get_api_response] API 요청 시작: {api_url}")
    response = await fetch_api(api_url)
    main_logger.debug(f"[get_api_response] API 요청 성공: {api_url}")
    return response

async def get_all_api_responses(
    query,
    api_func: Union[Callable, Awaitable[dict]],
    merge_func: Callable
):
    """
    페이징이 포함된 API를 반복 호출하여 모든 결과를 가져오고 병합합니다.

    Args:
        query: 요청 파라미터 객체 (display, page 포함).
        api_func: API 요청을 수행할 함수 (비동기 or 동기).
        merge_func: 결과를 병합할 함수.

    Returns:
        dict: 병합된 API 응답 결과.
    """
    url = APIEndpoints().get_list_url(query.get_query_params())
    is_async = inspect.iscoroutinefunction(api_func)

    # 첫 페이지 요청
    if is_async:
        first_response = await api_func(url)
    else:
        first_response = api_func(url)

    total_count = int(first_response.get("LawSearch", {}).get("totalCnt", 0))
    all_results = [first_response]

    # 필요한 총 페이지 수 계산
    total_pages = (total_count + query.display - 1) // query.display

    main_logger.debug(f"총 {total_count}개 항목, {total_pages}페이지 조회 필요")
    
    # 2페이지부터 마지막 페이지까지 요청
    for page in range(2, total_pages + 1):
        query.page = page
        url = APIEndpoints().get_list_url(query.get_query_params())
        main_logger.debug(f"{page}/{total_pages} 페이지 요청 중... / {url}")

        if is_async:
            response = await api_func(url)
        else:
            response = api_func(url)

        all_results.append(response)

    # 병합 처리
    merged_result = merge_func(all_results)
    return merged_result

def merge_amend_responses(responses) -> dict[dict[list[dict]]]:
    """
    개정 법령 API 응답들을 병합하는 함수.

    Args:
        responses (list[dict]): API 응답 리스트.

    Returns:
        dict: 병합된 응답 결과.
    """
    if not responses:
        return {}

    merged = copy.deepcopy(responses[0])

    if "LawSearch" not in merged:
        return merged

    all_items = []

    for response in responses:
        if "LawSearch" in response and "law" in response["LawSearch"]:
            items = response["LawSearch"]["law"]
            if isinstance(items, list):
                all_items.extend(items)
            else:
                all_items.append(items)

    merged["LawSearch"]["law"] = all_items
    return merged

async def get_api_amend(query: UpdatedLawRequestParams):
    """
    개정 법령 목록을 조회하고 병합된 결과를 반환합니다.

    Args:
        query (UpdatedLawRequestParams): 개정일자를 포함한 요청 파라미터.

    Returns:
        dict: 병합된 개정 법령 응답 결과.
    """
    main_logger.debug(f"[get_api_amend] API 요청 시작: {query.regDt}")
    response = await get_all_api_responses(query, fetch_api_amend, merge_amend_responses)
    main_logger.debug(f"[get_api_amend] API 요청 성공: {query.regDt}")
    return response
