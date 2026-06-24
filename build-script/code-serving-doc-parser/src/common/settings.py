import os
from typing import Optional

from pydantic_settings import BaseSettings

from common.env_loader import get_env_file_list, customise_sources


MODULE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = os.path.dirname(MODULE_DIR)


class Config:
    extra = "allow"
    env_file = get_env_file_list(module_dir=MODULE_DIR, root_dir=ROOT_DIR)
    env_file_encoding = 'utf-8'

BaseSettings.settings_customise_sources = classmethod(
    lambda cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings:
    customise_sources(init_settings, env_settings, dotenv_settings, file_secret_settings)
)


class Settings(BaseSettings):
    class Config(Config):
        pass

    LOG_PATH: list[str] = [
        "/var/log/supervisor/gunicorn_stderr.log",
        "/var/log/supervisor/gunicorn_stdout.log",
    ]
    
    HOSTNAME: Optional[str] = None
    CODE_SERVING_ID: Optional[int] = None
    CODE_SERVING_DEPLOYMENT_ID: Optional[int] = None
    COMMIT_HASH: Optional[str] = None
    REPOSITORY_URL: Optional[str] = None
    
    @property
    def POD_ID(self) -> Optional[str]:
        if self.HOSTNAME:
            return self.HOSTNAME.split('-')[-1]
        return None


class MsgQueueConfig(BaseSettings):
    class Config(Config):
        pass

    MQ_HOST: str
    MQ_PORT: str
    MQ_USER: str
    MQ_PASSWORD: str
    MQ_VHOST: str
    MQ_EXCHANGE_TYPE: str

    MQ_EXCHANGE_NAME_LOG: str
    MQ_QUEUE_NAME_LOG: str
    MQ_QUEUE_BIND_ROUTING_KEY_LOG: str
    
    @property
    def MQ_ROUTING_KEY_LOG(self) -> str:
        return f'log.code_serving.{settings.CODE_SERVING_ID}.{settings.POD_ID}'


settings = Settings()
msg_queue_config = MsgQueueConfig()
