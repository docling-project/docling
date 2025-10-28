from typing import Any, Union

import aiohttp

from commons.loggers import ErrorLogger, MainLogger
from commons.settings import settings
from schemas.params import VectorAPIEndpoints
from schemas.vdb_schema import (
    VDBRegisterRequest,
    VDBRegisterResponse,
    VDBUploadFile,
    VDBUploadResponse,
)

# 로그 인스턴스 초기화
main_logger = MainLogger.instance()
error_logger = ErrorLogger.instance()


def get_headers() -> dict[str, str]:
    """
    VDB API 호출에 필요한 HTTP 헤더를 반환합니다.

    Returns:
        dict[str, str]: 인증 토큰과 JSON 응답을 요청하는 Accept 헤더를 포함한 딕셔너리.
    """
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {settings.genos_admin_token}"
    }
    return headers


async def request_post(url: str, data: Any = None, is_json: bool = True) -> Union[dict[str, Any], None]:
    """
    주어진 URL로 POST 요청을 전송하고 JSON 응답을 반환합니다.

    Args:
        url (str): POST 요청을 보낼 API 엔드포인트 URL.
        data (Any, optional): 요청에 포함할 데이터. pydantic 객체 또는 FormData.
        is_json (bool, optional): 데이터가 JSON 형식인지 여부. 기본값은 True.

    Returns:
        Union[dict[str, Any], None]: JSON 응답을 파싱한 결과. 실패 시 None 반환.
    """
    headers = get_headers()

    try:
        async with aiohttp.ClientSession() as session:
            request_kwargs: dict[str, Any] = {}

            if data:
                if is_json:
                    request_kwargs["json"] = data.model_dump()  # pydantic 모델 -> dict
                else:
                    request_kwargs["data"] = data  # FormData

            main_logger.debug(f"[request_post] POST {url} data={data}")

            async with session.post(url, headers=headers, **request_kwargs) as response:
                response.raise_for_status()  # 상태코드가 400 이상이면 예외 발생
                response_json = await response.json()
                main_logger.debug(f"[request_post] POST 성공 {url} status={response.status}")
                return response_json

    except aiohttp.ClientResponseError as e:
        error_logger.vdb_error(f"[request_post] HTTP Error {e.status} {e.message} for {url}", e)
    except aiohttp.ClientError as e:
        error_logger.vdb_error(f"[request_post] Client Error for {url}", e)
    except Exception as e:
        error_logger.vdb_error(f"[request_post] 알 수 없는 에러 for {url}", e)

    return None


async def upload_file(request: list[VDBUploadFile]) -> Union[VDBUploadResponse, None]:
    """
    파일 리스트를 VDB 서버에 업로드합니다.

    Args:
        request (list[VDBUploadFile]): 업로드할 파일 목록.

    Returns:
        Union[VDBUploadResponse, None]: 업로드 결과 응답. 실패 시 None.
    """
    url: str = VectorAPIEndpoints().get_upload_route()

    try:
        form = aiohttp.FormData()

        # 각 파일을 multipart/form-data 형식으로 추가
        for file in request:
            form.add_field(
                name="files",
                value=file.file_content,
                filename=file.file_name,
                content_type="application/json"
            )

        response_json = await request_post(
            url=url,
            data=form,
            is_json=False
        )

        if response_json:
            response = VDBUploadResponse(**response_json)
            main_logger.debug(f"[upload_file] VDB 파일 업로드 성공: {response.data.files}")
            return response

    except Exception as e:
        error_logger.vdb_error(f"[upload_file] 파일 업로드 실패 {request[0]}", e)

    return None


async def register_vector(request: VDBRegisterRequest) -> Union[VDBRegisterResponse, None]:
    """
    업로드된 파일에 대해 벡터 등록 요청을 보냅니다.

    Args:
        request (VDBRegisterRequest): 벡터 등록에 필요한 정보가 담긴 요청 객체.

    Returns:
        Union[VDBRegisterResponse, None]: 등록 결과 응답. 실패 시 None.
    """
    url: str = VectorAPIEndpoints().get_register_route()

    try:
        response_json = await request_post(
            url=url,
            data=request,
            is_json=True
        )

        if response_json:
            response = VDBRegisterResponse(**response_json)
            main_logger.debug(
                f"[upload_file] VDB 벡터 등록 성공: {response.data.doc_ids[0]}, {response.data.upsert_ids[0]}"
            )
            return response

    except Exception as e:
        error_logger.vdb_error(f"[register_vector] 벡터 등록 실패, {request.files}", e)

    return None
