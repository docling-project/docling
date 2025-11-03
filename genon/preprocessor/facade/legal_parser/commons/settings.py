from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    애플리케이션에서 사용하는 주요 환경변수 설정 클래스.
    `.env` 파일 또는 환경변수로부터 값을 로드합니다.
    """

    # 운영 환경 설정
    oc: str = Field(..., alias="OC")

    # Genos 플랫폼 관련 인증 정보
    genos_admin_token: str = Field(..., alias="GENOS_ADMIN_TOKEN")
    genos_admin_user_id: int = Field(..., alias="GENOS_ADMIN_USER_ID")

    # 테스트용 VDB 구성 정보
    genos_test_vdb_index: str = Field(..., alias="GENOS_TEST_VDB_INDEX")
    genos_test_vdb_id: str = Field(..., alias="GENOS_TEST_VDB_ID")
    genos_test_serving_id: int = Field(..., alias="GENOS_TEST_SERVING_ID")
    genos_test_serving_rev_id: int = Field(..., alias="GENOS_TEST_SERVING_REV_ID")
    genos_test_preprocessor_id: int = Field(..., alias="GENOS_TEST_PREPROCESSOR_ID")

    # 내부 Admin API 서버 정보
    admin_api_host: str = Field(..., alias="ADMIN_API_HOST")
    admin_api_port: int = Field(..., alias="ADMIN_API_PORT")
    admin_api_root_path: str = Field(..., alias="ADMIN_API_ROOT_PATH")

    # LLMOps에서 사용하는 Weaviate DB 접속 정보
    llmops_weaviate_host: str = Field(..., alias="LLMOPS_WEAVIATE_HOST")
    llmops_weaviate_port: str = Field(..., alias="LLMOPS_WEAVIATE_PORT")
    llmops_weaviate_replicas: str = Field(..., alias="LLMOPS_WEAVIATE_REPLICAS")

    class Config:
        env_file = ".env"             # 환경변수 파일 위치
        case_sensitive = False        # 대소문자 구분 없이 매핑

# 전역 설정 인스턴스
settings = Settings()
