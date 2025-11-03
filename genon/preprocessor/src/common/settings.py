import os
from pathlib import Path
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


settings = Settings()
msg_queue_config = MsgQueueConfig()