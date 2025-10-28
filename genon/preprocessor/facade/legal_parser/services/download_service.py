from typing import Union

from api.law_client import fetch_api
from commons.file_handler import export_json_input
from schemas.params import (
    AdmRuleRequestParams,
    APIEndpoints,
    LawItemRequestParams,
)


async def download_data(query: Union[LawItemRequestParams, AdmRuleRequestParams]):
    """
    주어진 쿼리 매개변수를 사용하여 데이터를 다운로드하고 JSON 파일로 저장하는 함수.

    Args:
        query (Union[LawItemRequestParams, AdmRuleRequestParams]): 
            법령 또는 행정규칙 데이터를 요청하기 위한 쿼리 매개변수.
    """
    # API URL 생성
    api_url = APIEndpoints().get_item_url(query.get_query_params())
    
    # 요청 ID 설정 (법령 ID 또는 행정규칙 ID)
    id = query.MST if isinstance(query, LawItemRequestParams) else query.ID
    
    # API 호출을 통해 데이터 가져오기
    result = await fetch_api(api_url)
    
    # 가져온 데이터를 JSON 파일로 저장
    export_json_input(result, id)