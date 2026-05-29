"""enrichment_config.py — Enrichment 설정 파싱 전담 모듈.

parser_processor.py 의 enrichment 설정 읽기 로직을 단일 typed dataclass 로 집결시킨다.

지원 YAML 포맷:
  Format A (dict):  enrichment: {do_toc: true, api_url: "...", toc: {...}, ...}
  Format B (list):  enrichment: [{toc: {...}}, {metadata: {...}}, ...]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger(__name__)


# ── module-private helpers ────────────────────────────────────────────────────

def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _parse_optional_bool(value: Any, key: str = "") -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    if key:
        _log.warning(
            f"[EnrichmentConfig] Invalid bool value for '{key}': {value!r}. Falling back to default."
        )
    return None


def _parse_optional_int(value: Any, key: str = "") -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        if key:
            _log.warning(
                f"[EnrichmentConfig] Invalid int value for '{key}': {value!r}. Falling back to default."
            )
        return None


# enricher 이름 alias 매핑
_ENRICHMENT_LIST_NAMES: dict[str, set[str]] = {
    "toc": {"toc", "toc_enricher"},
    "metadata": {"metadata", "extract_metadata", "metadata_enricher", "extract_metadata_enricher"},
    "image_description": {"image_description", "image_description_enricher"},
    "custom_fields": {"custom_fields", "custom_fields_enricher"},
}


# ── Sub-dataclasses ───────────────────────────────────────────────────────────

@dataclass
class _TocConfig:
    do_toc: bool
    url: str
    api_key: str
    model: str
    temperature: float
    top_p: float
    seed: int
    max_tokens: int
    precheck_enabled: Optional[bool]
    precheck_max_context_tokens: Optional[int]
    precheck_completion_reserved_tokens: Optional[int]
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    doc_type: str = "law"


@dataclass
class _MetadataConfig:
    do_metadata: bool
    url: str
    api_key: str
    model: str
    precheck_enabled: Optional[bool]
    precheck_max_context_tokens: Optional[int]
    precheck_completion_reserved_tokens: Optional[int]
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    parser: dict = None
    output_fields: list = None
    max_tokens: int = 10000
    temperature: float = 0.0
    timeout: int = 3600
    pages: Optional[list] = None

    def __post_init__(self):
        if self.parser is None:
            self.parser = {}
        if self.output_fields is None:
            self.output_fields = []


# ── Main dataclass ────────────────────────────────────────────────────────────

@dataclass
class EnrichmentConfig:
    """Enrichment 설정의 typed 표현.

    toc / metadata / image_description / custom_fields 네 enricher 의 설정을 담는다.
    api_url / api_key / model 은 Format A 의 global fallback 값을 보존해
    ImageDescriptionOptions.from_config 가 fallback 인자로 수신할 수 있게 한다.
    """

    toc: _TocConfig
    metadata: _MetadataConfig
    image_description_cfg: dict
    custom_fields_cfgs: list
    api_url: str
    api_key: str
    model: str

    @classmethod
    def from_raw(
        cls,
        raw: "list | dict | None",
        config_dir: Path,
        parent_cfg: "dict | None" = None,
    ) -> "EnrichmentConfig":
        """YAML enrichment 값을 EnrichmentConfig 로 변환한다.

        Args:
            raw: cfg.get("enrichment") — list 또는 dict.
            config_dir: custom_fields resource_path 자동 주입에 사용.
            parent_cfg: 최상위 config dict. Format A 에서 legacy top-level 키 fallback에 사용.
        """
        if isinstance(raw, list):
            return cls._from_list(raw, config_dir)
        return cls._from_dict(_as_dict(raw), config_dir, _as_dict(parent_cfg))

    # ── Format B (list) ───────────────────────────────────────────────────────

    @classmethod
    def _from_list(cls, items: list, config_dir: Path) -> "EnrichmentConfig":
        toc_opts: dict = {}
        toc_enabled = False
        toc_precheck: dict = {}

        metadata_opts: dict = {}
        metadata_enabled = False
        metadata_precheck: dict = {}

        image_desc_cfg: dict = {"enabled": False}
        custom_fields_cfgs: list = []

        for item in items:
            if not isinstance(item, dict):
                continue
            raw_name = next(iter(item), None)
            if not raw_name:
                continue
            opts = dict(_as_dict(item.get(raw_name)))
            enabled = opts.pop("enable", True)
            name = raw_name.lower()
            if name.endswith("_enricher"):
                name = name[:-9]
            category = next(
                (k for k, aliases in _ENRICHMENT_LIST_NAMES.items() if name in aliases),
                name,
            )

            if category == "toc":
                if enabled:
                    toc_enabled = True
                    toc_precheck = opts.pop("precheck", {})
                    toc_opts = opts
            elif category == "metadata":
                if enabled:
                    metadata_enabled = True
                    metadata_precheck = opts.pop("precheck", {})
                    metadata_opts = opts
                else:
                    metadata_opts = {}
            elif category == "image_description":
                if enabled:
                    opts.setdefault("enabled", True)
                    image_desc_cfg = opts
                else:
                    image_desc_cfg = {"enabled": False}
            elif category == "custom_fields":
                if enabled and opts:
                    if "resource_path" not in opts:
                        opts["resource_path"] = str(config_dir)
                    custom_fields_cfgs.append(opts)

        toc_system_prompt = toc_opts.get("system_prompt") or None
        toc_user_prompt = toc_opts.get("user_prompt") or None
        if isinstance(toc_system_prompt, str):
            toc_system_prompt = toc_system_prompt.strip() or None
        if isinstance(toc_user_prompt, str):
            toc_user_prompt = toc_user_prompt.strip() or None

        meta_system_prompt = metadata_opts.get("system_prompt") or None
        meta_user_prompt = metadata_opts.get("user_prompt") or None
        if isinstance(meta_system_prompt, str):
            meta_system_prompt = meta_system_prompt.strip() or None
        if isinstance(meta_user_prompt, str):
            meta_user_prompt = meta_user_prompt.strip() or None
        meta_pages = metadata_opts.get("pages")
        if not isinstance(meta_pages, list) or not meta_pages:
            meta_pages = None

        return cls(
            toc=_TocConfig(
                do_toc=toc_enabled,
                url=str(toc_opts.get("url") or ""),
                api_key=str(toc_opts.get("api_key") or ""),
                model=str(toc_opts.get("model") or "model"),
                temperature=float(toc_opts.get("temperature", 0.0)),
                top_p=float(toc_opts.get("top_p", 0.00001)),
                seed=int(toc_opts.get("seed", 33)),
                max_tokens=int(toc_opts.get("max_tokens", 10000)),
                precheck_enabled=_parse_optional_bool(toc_precheck.get("enabled")),
                precheck_max_context_tokens=_parse_optional_int(
                    toc_precheck.get("max_context_tokens", toc_precheck.get("max_context"))
                ),
                precheck_completion_reserved_tokens=_parse_optional_int(
                    toc_precheck.get("completion_reserved_tokens")
                ),
                system_prompt=toc_system_prompt,
                user_prompt=toc_user_prompt,
                doc_type=str(toc_opts.get("doc_type", "law") or "law"),
            ),
            metadata=_MetadataConfig(
                do_metadata=metadata_enabled,
                url=str(metadata_opts.get("url") or ""),
                api_key=str(metadata_opts.get("api_key") or ""),
                model=str(metadata_opts.get("model") or "model"),
                precheck_enabled=_parse_optional_bool(metadata_precheck.get("enabled")),
                precheck_max_context_tokens=_parse_optional_int(
                    metadata_precheck.get("max_context_tokens", metadata_precheck.get("max_context"))
                ),
                precheck_completion_reserved_tokens=_parse_optional_int(
                    metadata_precheck.get("completion_reserved_tokens")
                ),
                system_prompt=meta_system_prompt,
                user_prompt=meta_user_prompt,
                parser=dict(metadata_opts.get("parser") or {}),
                output_fields=list(metadata_opts.get("output_fields") or []),
                max_tokens=int(metadata_opts.get("max_tokens", 10000)),
                temperature=float(metadata_opts.get("temperature", 0.0)),
                timeout=int(metadata_opts.get("timeout", 3600)),
                pages=meta_pages,
            ),
            image_description_cfg=image_desc_cfg,
            custom_fields_cfgs=custom_fields_cfgs,
            api_url="",
            api_key="",
            model="model",
        )

    # ── Format A (dict) ───────────────────────────────────────────────────────

    @classmethod
    def _from_dict(cls, cfg: dict, config_dir: Path, parent_cfg: dict) -> "EnrichmentConfig":
        global_url = str(
            cfg.get("api_url") or parent_cfg.get("enrichment_api_base_url", "")
        )
        global_key = str(
            cfg.get("api_key") or parent_cfg.get("enrichment_api_key", "")
        )
        global_model = str(
            cfg.get("model") or parent_cfg.get("enrichment_model", "model") or "model"
        )

        toc_cfg = _as_dict(cfg.get("toc"))
        meta_cfg = _as_dict(cfg.get("metadata", {}))

        precheck_cfg = _as_dict(cfg.get("precheck"))
        toc_precheck_cfg = _as_dict(precheck_cfg.get("toc"))
        meta_precheck_cfg = _as_dict(precheck_cfg.get("metadata"))

        common_max_context_tokens = _parse_optional_int(
            precheck_cfg.get(
                "max_context_tokens",
                precheck_cfg.get("max_context", parent_cfg.get("enrichment_max_context_tokens")),
            )
        )
        common_reserved_tokens = _parse_optional_int(
            precheck_cfg.get(
                "completion_reserved_tokens",
                parent_cfg.get("enrichment_completion_reserved_tokens"),
            )
        )

        raw_cf = cfg.get("custom_fields")
        if isinstance(raw_cf, dict) and raw_cf:
            cf_list = [dict(raw_cf)]
        elif isinstance(raw_cf, list):
            cf_list = [dict(_as_dict(c)) for c in raw_cf if isinstance(c, dict) and c]
        else:
            cf_list = []
        for _cf in cf_list:
            if "resource_path" not in _cf:
                _cf["resource_path"] = str(config_dir)

        toc_sp = toc_cfg.get("system_prompt") or None
        toc_up = toc_cfg.get("user_prompt") or None
        if isinstance(toc_sp, str):
            toc_sp = toc_sp.strip() or None
        if isinstance(toc_up, str):
            toc_up = toc_up.strip() or None

        meta_sp = meta_cfg.get("system_prompt") or None
        meta_up = meta_cfg.get("user_prompt") or None
        if isinstance(meta_sp, str):
            meta_sp = meta_sp.strip() or None
        if isinstance(meta_up, str):
            meta_up = meta_up.strip() or None
        meta_pages_d = meta_cfg.get("pages")
        if not isinstance(meta_pages_d, list) or not meta_pages_d:
            meta_pages_d = None

        return cls(
            toc=_TocConfig(
                do_toc=bool(cfg.get("do_toc", parent_cfg.get("do_toc", True))),
                url=str(toc_cfg.get("url", global_url) or global_url),
                api_key=str(toc_cfg.get("api_key", global_key) or global_key),
                model=str(toc_cfg.get("model", global_model) or global_model),
                temperature=float(toc_cfg.get("temperature", parent_cfg.get("toc_temperature", 0.0))),
                top_p=float(toc_cfg.get("top_p", parent_cfg.get("toc_top_p", 0.00001))),
                seed=int(toc_cfg.get("seed", parent_cfg.get("toc_seed", 33))),
                max_tokens=int(toc_cfg.get("max_tokens", parent_cfg.get("toc_max_tokens", 10000))),
                precheck_enabled=_parse_optional_bool(
                    toc_precheck_cfg.get(
                        "enabled",
                        precheck_cfg.get("enabled", parent_cfg.get("toc_precheck_enabled")),
                    )
                ),
                precheck_max_context_tokens=_parse_optional_int(
                    toc_precheck_cfg.get(
                        "max_context_tokens",
                        toc_precheck_cfg.get("max_context", common_max_context_tokens),
                    )
                ),
                precheck_completion_reserved_tokens=_parse_optional_int(
                    toc_precheck_cfg.get("completion_reserved_tokens", common_reserved_tokens)
                ),
                system_prompt=toc_sp,
                user_prompt=toc_up,
                doc_type=str(toc_cfg.get("doc_type", parent_cfg.get("toc_doc_type", "law")) or "law"),
            ),
            metadata=_MetadataConfig(
                do_metadata=bool(cfg.get("do_metadata", parent_cfg.get("do_metadata", True))),
                url=str(meta_cfg.get("url", global_url) or global_url),
                api_key=str(meta_cfg.get("api_key", global_key) or global_key),
                model=str(meta_cfg.get("model", global_model) or global_model),
                precheck_enabled=_parse_optional_bool(
                    meta_precheck_cfg.get(
                        "enabled",
                        precheck_cfg.get("enabled", parent_cfg.get("metadata_precheck_enabled")),
                    )
                ),
                precheck_max_context_tokens=_parse_optional_int(
                    meta_precheck_cfg.get(
                        "max_context_tokens",
                        meta_precheck_cfg.get("max_context", common_max_context_tokens),
                    )
                ),
                precheck_completion_reserved_tokens=_parse_optional_int(
                    meta_precheck_cfg.get(
                        "completion_reserved_tokens",
                        common_reserved_tokens,
                    )
                ),
                system_prompt=meta_sp,
                user_prompt=meta_up,
                parser=dict(meta_cfg.get("parser") or {}),
                output_fields=list(meta_cfg.get("output_fields") or []),
                max_tokens=int(meta_cfg.get("max_tokens", 10000)),
                temperature=float(meta_cfg.get("temperature", 0.0)),
                timeout=int(meta_cfg.get("timeout", 3600)),
                pages=meta_pages_d,
            ),
            image_description_cfg=_as_dict(cfg.get("image_description")),
            custom_fields_cfgs=cf_list,
            api_url=global_url,
            api_key=global_key,
            model=global_model,
        )
