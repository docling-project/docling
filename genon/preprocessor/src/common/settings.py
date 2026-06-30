import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings

PROFILE = os.getenv("PROFILE", "dev")

_HOSTNAME = os.environ.get("HOSTNAME", None)
_ID: str | None = os.environ.get("PREPROCESSOR_ID", None)
_POD_ID = _HOSTNAME.split("-")[-1] if _HOSTNAME else None


def get_env_path(profile: str) -> str:
    """환경 파일 경로를 생성합니다."""
    current_dir = Path(__file__).resolve()
    project_root = current_dir.parent.parent.parent
    return str(project_root / f'env/.env.{profile}')


class BaseConfig:
    extra = "allow"
    env_file_encoding = 'utf-8'
    env_file = [get_env_path(PROFILE)]
    if PROFILE == 'prod':
        env_file.append(get_env_path('global'))


class Settings(BaseSettings):
    class Config(BaseConfig):
        pass

    PREPROCESSOR_ID: Optional[str] = _ID
    POD_ID: str = _POD_ID
    LOG_PATH: list[str] = [
        "/var/log/supervisor/gunicorn_stderr.log",
        "/var/log/supervisor/gunicorn_stdout.log"
    ]


class MsgQueueConfig(BaseSettings):
    class Config(BaseConfig):
        pass

    MQ_HOST: str
    MQ_PORT: str
    MQ_USER: str
    MQ_PASSWORD: str
    MQ_VHOST: str
    MQ_EXCHANGE_TYPE: str

    # Input / Output Mongo에 쌓을거라면 추가
    # MQ_EXCHANGE_NAME: str
    # MQ_QUEUE_NAME: str
    # MQ_QUEUE_BIND_ROUTING_KEY: str
    # MQ_ROUTING_KEY_REQUEST: str
    # MQ_ROUTING_KEY_RESPONSE: str

    MQ_EXCHANGE_NAME_LOG: str
    MQ_QUEUE_NAME_LOG: str
    MQ_QUEUE_BIND_ROUTING_KEY_LOG: str
    MQ_ROUTING_KEY_LOG: str = f'log.preprocessor.{_ID}.{_POD_ID}'


class MinioConfig(BaseSettings):
    class Config(BaseConfig):
        pass

    MINIO_ENDPOINT: Optional[str] = None
    MINIO_ACCESS_KEY: Optional[str] = None
    MINIO_SECRET_KEY: Optional[str] = None

    def is_configured(self) -> bool:
        return all([self.MINIO_ENDPOINT, self.MINIO_ACCESS_KEY, self.MINIO_SECRET_KEY])


settings = Settings()
msg_queue_config = MsgQueueConfig()
# MinioConfig 는 사용 시점(util/minio_resource.download_resource_files)에서 인스턴스화한다.
# 모듈 import 시점에 무조건 인스턴스화하면 MinIO 를 쓰지 않는 환경에서도
# MINIO_* 환경변수 누락만으로 워커 boot 가 실패함. main.py:49 에서 PREPROCESSOR_ID
# 가 있을 때만 MinIO 가 사용되므로, 검증도 그 흐름 안에서 일어나야 한다.