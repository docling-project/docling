from __future__ import annotations

import os
import importlib.resources
from dataclasses import dataclass
from typing import Any

try:
    import tomllib  # py>=3.11
except ModuleNotFoundError:
    import tomli as tomllib  # py<=3.10

@dataclass
class DBConfig:
    username: str
    password: str
    host: str
    port: int
    name: str

    @property
    def url(self) -> str:
        return f"mysql+asyncmy://{self.username}:{self.password}@{self.host}:{self.port}/{self.name}"

@dataclass
class WeaviateConfig:
    host: str
    port: int
    grpc_port: int
    skip_init_checks: bool = True

@dataclass
class AppConfig:
    output_dir: str = "/app"

@dataclass
class ProfileConfig:
    db: DBConfig
    weaviate: WeaviateConfig
    app: AppConfig

ENV_MAP = {
    "DB_USERNAME": ("db", "username"),
    "DB_PASSWORD": ("db", "password"),
    "DB_HOST": ("db", "host"),
    "DB_PORT": ("db", "port"),
    "DB_NAME": ("db", "name"),
    "WEAVIATE_HOST": ("weaviate", "host"),
    "WEAVIATE_PORT": ("weaviate", "port"),
    "WEAVIATE_GRPC_PORT": ("weaviate", "grpc_port"),
    "WEAVIATE_SKIP_INIT": ("weaviate", "skip_init_checks"),
    "APP_OUTPUT_DIR": ("app", "output_dir"),
}

def _load_toml(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)

def _find_config_path(explicit: str | None) -> str:
    """
    설정 파일 탐색 순서:
    1) --config 인자
    2) 환경 변수 GENOS_TOOLS_CONFIG
    3) 패키지 내부(genos_tools/config.toml 또는 config.example.toml)
    4) 홈 디렉토리 ~/.config/genos-tools/config.toml
    """
    # ① 명시적 인자
    if explicit and os.path.exists(explicit):
        return explicit

    # ② 환경 변수
    env_path = os.environ.get("GENOS_TOOLS_CONFIG")
    if env_path and os.path.exists(env_path):
        return env_path

    # ③ 패키지 내부 기본값 (설치된 genos_tools/config.toml)
    try:
        pkg_root = os.path.dirname(importlib.resources.files("genos_tools"))
        internal_path = os.path.join(pkg_root, "config.toml")
        if os.path.exists(internal_path):
            return internal_path
        # fallback: config.example.toml
        example_path = os.path.join(pkg_root, "config.toml")
        if os.path.exists(example_path):
            return example_path
    except Exception:
        pass

    # ④ 홈 디렉토리 경로
    home_cfg = os.path.expanduser("~/.config/genos-tools/config.toml")
    if os.path.exists(home_cfg):
        return home_cfg

    raise SystemExit(
        "[config] config.toml 파일을 찾지 못했습니다. "
        "패키지 내부에도 없으며, --config 또는 GENOS_TOOLS_CONFIG 로 지정해야 합니다."
    )

def load_profile_config(profile: str, config_path: str | None = None) -> ProfileConfig:
    path = _find_config_path(config_path)
    data = _load_toml(path)
    try:
        raw = data["profiles"][profile]
    except KeyError:
        raise SystemExit(f"[config] profile '{profile}' not found in {path}")

    # ENV override
    for env_key, (section, key) in ENV_MAP.items():
        if env_key in os.environ:
            val: Any = os.environ[env_key]
            if (section, key) in {("db", "port"), ("weaviate", "port"), ("weaviate", "grpc_port")}:
                val = int(val)
            if (section, key) == ("weaviate", "skip_init_checks"):
                val = str(val).lower() in {"1", "true", "yes", "y"}
            raw.setdefault(section, {})
            raw[section][key] = val

    db = DBConfig(
        username=raw["db"]["username"],
        password=raw["db"]["password"],
        host=raw["db"]["host"],
        port=int(raw["db"]["port"]),
        name=raw["db"]["name"],
    )
    weav = WeaviateConfig(
        host=raw["weaviate"]["host"],
        port=int(raw["weaviate"]["port"]),
        grpc_port=int(raw["weaviate"]["grpc_port"]),
        skip_init_checks=bool(raw["weaviate"].get("skip_init_checks", True)),
    )
    app = AppConfig(
        output_dir=raw.get("app", {}).get("output_dir", "/app")
    )
    return ProfileConfig(db=db, weaviate=weav, app=app)
