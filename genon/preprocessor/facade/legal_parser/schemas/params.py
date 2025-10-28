from datetime import datetime, timedelta
from typing import Literal, Optional
from urllib.parse import urlencode

import pytz
from pydantic import AnyHttpUrl, BaseModel, Field, field_validator, model_validator

from commons.settings import settings
from commons.utils import get_kst_yesterday_str

# ====================== 국가법령정보 공동활용 API Endpoint & Query Params =======================

class APIEndpoints(BaseModel):
    """
    국가법령정보 공동활용 API의 엔드포인트를 정의하는 클래스
    """
    base_url: AnyHttpUrl = "https://www.law.go.kr/DRF/"
    item_endpoint: str = "lawService.do"
    list_endpoint: str = "lawSearch.do"

    def get_item_url(self, query: str) -> str:
        """법령 상세조회용 전체 URL 생성"""
        return f"{self.base_url}{self.item_endpoint}?{query}"

    def get_list_url(self, query: str) -> str:
        """법령 검색 목록조회용 전체 URL 생성"""
        return f"{self.base_url}{self.list_endpoint}?{query}"


class BaseRequestParams(BaseModel):
    """
    모든 요청에서 공통으로 사용되는 파라미터
    """
    OC: str = Field(settings.oc)
    type: Literal["HTML", "XML", "JSON"] = Field(
        "JSON", description="출력 형태: HTML/XML/JSON (필수)"
    )

    def get_query_params(self) -> str:
        """
        인스턴스의 데이터를 URL 쿼리 문자열로 변환
        """
        params_dict = self.model_dump(exclude_none=True)
        query_keyword = params_dict.pop("query", None)
        encoded_params = urlencode(params_dict)

        if query_keyword:
            return f"{encoded_params}&query={query_keyword}"
        return encoded_params


class LawItemRequestParams(BaseRequestParams):
    """
    개별 법령 본문 조회를 위한 요청 파라미터
    """
    target: Literal["law"] = Field("law")
    ID: Optional[str] = Field(None, description="법령 ID (ID 또는 KEY 중 하나 필수)")
    MST: Optional[str] = Field(None, description="법령 마스터 번호", examples=["264627"])
    JO: Optional[str] = Field(
        None,
        description="조번호: 생략 시 전체 조 표시. 예: 000200 (2조), 001002 (10조의 2)"
    )

    @model_validator(mode="after")
    def validate_id_or_mst(cls, values: "LawItemRequestParams") -> "LawItemRequestParams":
        """ID 또는 MST는 반드시 하나 이상 존재해야 함"""
        if not values.ID and not values.MST:
            raise ValueError("ID 또는 MST 중 하나는 반드시 입력해야 합니다.")
        return values

    @field_validator("JO")
    def validate_jo_format(cls, value):
        """JO 필드가 6자리 숫자인지 검사"""
        if value and not (00000 <= value <= 999999):
            raise ValueError("JO는 6자리 숫자여야 합니다 (예: 000200).")
        return value


class LawSystemRequestParams(BaseRequestParams):
    """
    법령 체계도 요청 파라미터
    """
    target: Literal["lsStmd"] = "lsStmd"
    ID: Optional[str] = Field(None, description="법령 ID")
    MST: Optional[str] = Field(None, description="법령 마스터 번호")
    LM: Optional[str] = Field(None, description="법령명")
    LD: Optional[int] = Field(None, description="공포일자")
    LN: Optional[int] = Field(None, description="공포번호")

    @model_validator(mode="after")
    def validate_id_or_mst(cls, values: "LawSystemRequestParams") -> "LawSystemRequestParams":
        """ID 또는 MST는 반드시 하나 이상 존재해야 함"""
        if not values.ID and not values.MST:
            raise ValueError("ID 또는 MST 중 하나는 반드시 입력해야 합니다.")
        return values


class AdmRuleRequestParams(BaseRequestParams):
    """
    행정규칙 본문 요청 파라미터
    """
    target: Literal["admrul"] = "admrul"
    ID: Optional[str] = Field("2100000213205", description="행정규칙 일련번호")
    LID: Optional[str] = Field(None, description="행정규칙 ID")
    LM: Optional[str] = Field(None, description="정확한 행정규칙명")

    @model_validator(mode="after")
    def validate_id_or_mst(cls, values: "AdmRuleRequestParams") -> "AdmRuleRequestParams":
        """ID 또는 LID는 반드시 하나 이상 존재해야 함"""
        if not values.ID and not values.LID:
            raise ValueError("ID 또는 LID 중 하나는 반드시 입력해야 합니다.")
        return values


class LicBylRequestParams(BaseRequestParams):
    """
    법령 별표 서식 목록 요청 파라미터
    """
    target: Literal["licbyl"] = "licbyl"
    search: Optional[int] = Field(
        2, description="검색범위: 1=이름, 2=법령, 3=본문", ge=1, le=3
    )
    query: Optional[str] = Field("*", description="검색 질의어 (default=*)")
    display: Optional[int] = Field(20, description="결과 개수 (max=100)", le=100)
    page: Optional[int] = Field(1, description="결과 페이지 번호")
    sort: Literal["lasc", "ldes"] = Field(
        "lasc", description="정렬: lasc=오름차순, ldes=내림차순"
    )
    org: Optional[str] = Field(None, description="소관부처 코드")
    knd: Optional[str] = Field(None, description="별표 종류")
    gana: Optional[str] = Field(None, description="가나다 검색 키")
    popYn: Optional[str] = Field(None, description="팝업창 표시 여부")
    mulOrg: Literal["OR", "AND"] = Field(
        "OR", description="소관부처 복수 조건 (OR/AND)"
    )


class AdmBylRequestParams(BaseRequestParams):
    """
    행정규칙 별표 서식 목록 요청 파라미터
    """
    target: Literal["admbyl"] = "admbyl"
    knd: Optional[str] = Field(None, description="별표종류 (1~3)")
    search: Optional[int] = Field(
        2, description="검색범위: 1=이름, 2=법령, 3=본문", ge=1, le=3
    )
    query: Optional[str] = Field("*", description="검색 질의어 (default=*)")
    display: Optional[int] = Field(20, description="결과 개수 (max=100)", le=100)
    page: Optional[int] = Field(1, description="결과 페이지 번호")
    sort: Literal["lasc", "ldes"] = Field(
        "lasc", description="정렬 옵션"
    )
    org: Optional[str] = Field(None, description="소관부처 코드")
    gana: Optional[str] = Field(None, description="가나다 검색 키")
    popYn: Optional[str] = Field(None, description="팝업 여부")


class UpdatedLawRequestParams(BaseRequestParams):
    """
    일자별 개정된 법령 이력 조회 요청 파라미터
    """
    target: Literal["lsHstInf"] = Field(
        "lsHstInf", description="서비스 대상: 일자별 법령 개정 이력 목록 조회(필수)"
    )
    regDt: Optional[str] = Field(
        default_factory=get_kst_yesterday_str,
        description="법령 개정일 (YYYYMMDD 형식)"
    )
    display: Optional[int] = Field(30, description="결과 개수", ge=1, le=100)
    page: Optional[int] = Field(1, description="페이지 번호")

    @field_validator("regDt", mode="before")
    def validate_reg_dt(cls, value: Optional[str]) -> Optional[str]:
        """
        regDt는 오늘 또는 그 이후일 수 없으며, 포맷이 올바르지 않으면 어제로 대체됨
        """
        kst_yesterday = datetime.now(pytz.timezone("Asia/Seoul")) - timedelta(days=1)

        if value is None:
            return get_kst_yesterday_str()
        try:
            reg_date = datetime.strptime(value, "%Y%m%d")
        except ValueError:
            return get_kst_yesterday_str()

        if reg_date > kst_yesterday:
            return get_kst_yesterday_str()
        return value

# ====================== Genos VDB API Endpoint & Params =======================

class VectorAPIEndpoints(BaseModel):
    """
    Genos VDB와 통신하기 위한 벡터 API 엔드포인트 클래스
    """
    upload_endpoint: str = "/document/upload/token"
    register_endpoint: str = "/document/register/token"
    base_url: AnyHttpUrl = Field(default_factory=lambda: VectorAPIEndpoints.create_base_url())

    @staticmethod
    def create_base_url() -> str:
        """
        설정값을 기반으로 벡터 DB API의 기본 URL을 생성
        """
        host = settings.admin_api_host
        port = settings.admin_api_port
        root_path = settings.admin_api_root_path

        if not host or not port or not root_path:
            raise ValueError("환경 변수 ADMIN_API_HOST, ADMIN_API_PORT, ADMIN_ROOT_PATH가 모두 필요합니다.")

        return f"http://{host}:{port}{root_path}/data/vectordb"

    def get_upload_route(self) -> str:
        """문서 업로드 엔드포인트 URL 반환"""
        return f"{self.base_url}{self.upload_endpoint}"

    def get_register_route(self) -> str:
        """문서 등록 엔드포인트 URL 반환 (사용자 ID 포함)"""
        user_id = settings.genos_admin_user_id
        return f"{self.base_url}{self.register_endpoint}/{user_id}"
