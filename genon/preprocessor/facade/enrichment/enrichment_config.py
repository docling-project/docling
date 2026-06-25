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

from .prompt_files import read_prompt_file

_log = logging.getLogger(__name__)


# enricher 별 built-in default system prompt.
# user 가 user_prompt 만 지정(또는 user_prompt_file 만 지정)한 경우,
# system prompt 는 이 기본값을 사용한다. (우선순위: file > inline > default)
_DEFAULT_METADATA_SYSTEM_PROMPT = (
    "You are a professional document extraction assistant. "
    "Your job is to carefully extract metadata from semi-structured or "
    "unstructured documents. Always follow the requested output format exactly."
)


def _resolve_prompt(
    inline: Any,
    file_ref: Any,
    default: Optional[str],
    config_dir: Path,
) -> Optional[str]:
    """prompt 값을 우선순위에 따라 해석한다: file > inline > default.

    Args:
        inline: YAML 의 inline prompt 문자열 (`system_prompt`/`user_prompt`).
        file_ref: YAML 의 prompt 파일 경로 (`system_prompt_file`/`user_prompt_file`).
        default: 둘 다 없을 때 사용할 built-in 기본값 (없으면 None).
        config_dir: 상대경로 해석 기준 디렉토리.

    Raises:
        ValueError / FileNotFoundError: file_ref 가 잘못된 경우 (fail-fast).
    """
    if isinstance(file_ref, str) and file_ref.strip():
        return read_prompt_file(file_ref.strip(), config_dir)
    if isinstance(inline, str):
        stripped = inline.strip()
        if stripped:
            return stripped
    elif inline:
        return inline
    return default


def _parse_template_opts(opts: dict, global_mode: Optional[str] = None) -> "tuple[dict, str]":
    """enricher 블록에서 user-defined `variables` 와 `template.mode` 를 파싱한다.

    mode 우선순위: 블록의 template.mode / template_mode > global_mode > "strict".
    """
    variables = opts.get("variables")
    variables = dict(variables) if isinstance(variables, dict) else {}
    tmpl = opts.get("template")
    mode = tmpl.get("mode") if isinstance(tmpl, dict) else None
    mode = mode or opts.get("template_mode") or global_mode or "strict"
    mode = str(mode).strip().lower()
    if mode not in {"strict", "lenient"}:
        mode = "strict"
    return variables, mode


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


def _parse_optional_float(value: Any, key: str = "") -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        if key:
            _log.warning(
                f"[EnrichmentConfig] Invalid float value for '{key}': {value!r}. Falling back to default."
            )
        return None


def _parse_thinking(opts: dict) -> "tuple[str, str]":
    """enricher 블록에서 thinking / thinking_dialect 를 파싱한다.

    thinking 미지정 → ("off", "standard"): 기본적으로 추론을 끈다(차단 토큰 전송).
    아무것도 안 보내려면(모델 자동 판단) thinking: auto 로 명시.
    """
    raw = opts.get("thinking")
    thinking = "off" if raw is None else str(raw).strip().lower()
    dialect = str(opts.get("thinking_dialect", "standard") or "standard").strip().lower()
    if dialect not in {"standard", "hcx"}:
        dialect = "standard"
    return thinking, dialect


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
    repetition_penalty: Optional[float] = None  # >1.0 반복(degeneration) 억제. None=미전송
    # thinking(추론) 모드. 기본 "off"(차단 토큰 전송). "off"|"on"|"auto", dialect="standard"|"hcx"
    # 아무것도 안 보내려면(모델 자동 판단) thinking="auto".
    thinking: str = "off"
    thinking_dialect: str = "standard"
    # Split (carry-over refine) TOC extraction, page-based. None = use code defaults.
    split_enabled: Optional[bool] = None
    split_pages_per_chunk: Optional[int] = None
    split_page_overlap: Optional[int] = None
    split_carryover_max_tokens: Optional[int] = None


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
    field_transforms: list = None
    has_custom_metadata: bool = False
    variables: dict = None
    template_mode: str = "strict"
    # thinking(추론) 모드. 기본 "off"(차단 토큰 전송). "off"|"on"|"auto", dialect="standard"|"hcx"
    # 아무것도 안 보내려면(모델 자동 판단) thinking="auto".
    thinking: str = "off"
    thinking_dialect: str = "standard"

    def __post_init__(self):
        if self.parser is None:
            self.parser = {}
        if self.output_fields is None:
            self.output_fields = []
        if self.field_transforms is None:
            self.field_transforms = []
        if self.variables is None:
            self.variables = {}


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

        # toc 는 별도 built-in default 가 없다 (없으면 docling 레이어가 자체 기본값 사용).
        toc_system_prompt = _resolve_prompt(
            toc_opts.get("system_prompt"), toc_opts.get("system_prompt_file"), None, config_dir
        )
        toc_user_prompt = _resolve_prompt(
            toc_opts.get("user_prompt"), toc_opts.get("user_prompt_file"), None, config_dir
        )

        meta_system_prompt = _resolve_prompt(
            metadata_opts.get("system_prompt"),
            metadata_opts.get("system_prompt_file"),
            _DEFAULT_METADATA_SYSTEM_PROMPT,
            config_dir,
        )
        meta_user_prompt = _resolve_prompt(
            metadata_opts.get("user_prompt"),
            metadata_opts.get("user_prompt_file"),
            None,
            config_dir,
        )
        # 커스텀 metadata enricher opt-in 여부: 사용자가 prompt/파일/output_fields/parser 중
        # 하나라도 제공하면 True. (built-in default system prompt 가 게이트를 흔들지 않게 함)
        meta_has_custom = bool(
            metadata_opts.get("system_prompt")
            or metadata_opts.get("user_prompt")
            or metadata_opts.get("system_prompt_file")
            or metadata_opts.get("user_prompt_file")
            or metadata_opts.get("output_fields")
            or metadata_opts.get("parser")
        )
        meta_variables, meta_mode = _parse_template_opts(metadata_opts)
        meta_pages = metadata_opts.get("pages")
        if not isinstance(meta_pages, list) or not meta_pages:
            meta_pages = None

        toc_thinking, toc_thinking_dialect = _parse_thinking(toc_opts)
        meta_thinking, meta_thinking_dialect = _parse_thinking(metadata_opts)

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
                repetition_penalty=_parse_optional_float(toc_opts.get("repetition_penalty")),
                thinking=toc_thinking,
                thinking_dialect=toc_thinking_dialect,
                split_enabled=_parse_optional_bool(
                    _as_dict(toc_opts.get("split")).get("enabled")
                ),
                split_pages_per_chunk=_parse_optional_int(
                    _as_dict(toc_opts.get("split")).get("pages_per_chunk")
                ),
                split_page_overlap=_parse_optional_int(
                    _as_dict(toc_opts.get("split")).get("page_overlap")
                ),
                split_carryover_max_tokens=_parse_optional_int(
                    _as_dict(toc_opts.get("split")).get("carryover_max_tokens")
                ),
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
                field_transforms=list(metadata_opts.get("field_transforms") or []),
                has_custom_metadata=meta_has_custom,
                variables=meta_variables,
                template_mode=meta_mode,
                thinking=meta_thinking,
                thinking_dialect=meta_thinking_dialect,
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

        toc_sp = _resolve_prompt(
            toc_cfg.get("system_prompt"), toc_cfg.get("system_prompt_file"), None, config_dir
        )
        toc_up = _resolve_prompt(
            toc_cfg.get("user_prompt"), toc_cfg.get("user_prompt_file"), None, config_dir
        )

        meta_sp = _resolve_prompt(
            meta_cfg.get("system_prompt"),
            meta_cfg.get("system_prompt_file"),
            _DEFAULT_METADATA_SYSTEM_PROMPT,
            config_dir,
        )
        meta_up = _resolve_prompt(
            meta_cfg.get("user_prompt"),
            meta_cfg.get("user_prompt_file"),
            None,
            config_dir,
        )
        meta_has_custom = bool(
            meta_cfg.get("system_prompt")
            or meta_cfg.get("user_prompt")
            or meta_cfg.get("system_prompt_file")
            or meta_cfg.get("user_prompt_file")
            or meta_cfg.get("output_fields")
            or meta_cfg.get("parser")
        )
        global_mode = _as_dict(cfg.get("template")).get("mode")
        meta_variables, meta_mode = _parse_template_opts(meta_cfg, global_mode)
        meta_pages_d = meta_cfg.get("pages")
        if not isinstance(meta_pages_d, list) or not meta_pages_d:
            meta_pages_d = None

        toc_thinking, toc_thinking_dialect = _parse_thinking(toc_cfg)
        meta_thinking, meta_thinking_dialect = _parse_thinking(meta_cfg)

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
                repetition_penalty=_parse_optional_float(toc_cfg.get("repetition_penalty")),
                thinking=toc_thinking,
                thinking_dialect=toc_thinking_dialect,
                split_enabled=_parse_optional_bool(
                    _as_dict(toc_cfg.get("split")).get("enabled")
                ),
                split_pages_per_chunk=_parse_optional_int(
                    _as_dict(toc_cfg.get("split")).get("pages_per_chunk")
                ),
                split_page_overlap=_parse_optional_int(
                    _as_dict(toc_cfg.get("split")).get("page_overlap")
                ),
                split_carryover_max_tokens=_parse_optional_int(
                    _as_dict(toc_cfg.get("split")).get("carryover_max_tokens")
                ),
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
                field_transforms=list(meta_cfg.get("field_transforms") or []),
                has_custom_metadata=meta_has_custom,
                variables=meta_variables,
                template_mode=meta_mode,
                thinking=meta_thinking,
                thinking_dialect=meta_thinking_dialect,
            ),
            image_description_cfg=_as_dict(cfg.get("image_description")),
            custom_fields_cfgs=cf_list,
            api_url=global_url,
            api_key=global_key,
            model=global_model,
        )
