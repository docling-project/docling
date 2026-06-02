"""배포 YAML enrichment 설정의 구조 불변식 회귀 테스트.

브랜치 task/109-enrichment-enrichment-prompt 에서 추가/이동된 설정 키와 enrichment
프롬프트 구조가 이후 변경으로 깨지지 않도록 고정한다. 실제 서버 요청 없이 yaml 파싱과
EnrichmentConfig 변환만 수행하므로 GitHub CI(내부망 미접근)에서도 안전하게 동작한다.

고정 대상:
- defaults.log_level (int)            — `config yaml 에 log_lavel 추가`
- ocr.paddle.ocr_endpoint (위치/존재) — `paddle ocr endpoint 옵션 위치 변경`
- layout.genos_layout.max_completion_tokens (int) — `layout max token 값 늘림`
- enrichment 프롬프트 구조 (toc/metadata: system_prompt 존재, user_prompt 의 {{raw_text}} 치환자)
- attachment 설정에는 enrichment 섹션이 없음
"""

from pathlib import Path

import pytest
import yaml

from facade.enrichment.enrichment_config import EnrichmentConfig

# enrichment 을 사용하는 설정(메타데이터/TOC 추출)
ENRICHMENT_CONFIGS = [
    "resource/intelligent_processor_config.yaml",
    "resource/parser_processor_config.yaml",
    "resource_dev/intelligent_processor_config.yaml",
    "resource_dev/parser_processor_config.yaml",
]

ATTACHMENT_CONFIGS = [
    "resource/attachment_processor_config.yaml",
    "resource_dev/attachment_processor_config.yaml",
]


def _load(repo_root: Path, rel: str) -> dict:
    path = repo_root / rel
    if not path.exists():
        pytest.skip(f"config not present: {rel}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict), f"{rel} 최상위는 mapping 이어야 함"
    return cfg


def _parse_enrichment(repo_root: Path, rel: str) -> EnrichmentConfig:
    path = repo_root / rel
    cfg = _load(repo_root, rel)
    return EnrichmentConfig.from_raw(cfg.get("enrichment"), path.parent, parent_cfg=cfg)


_PLACEHOLDERS = ("{{raw_text}}", "{raw_text}")


# ── 최근 변경된 설정 키 위치/타입 고정 ─────────────────────────────────────────

@pytest.mark.regression
@pytest.mark.parametrize("rel", ENRICHMENT_CONFIGS + ATTACHMENT_CONFIGS)
def test_log_level_present_and_int(repo_root, rel):
    cfg = _load(repo_root, rel)
    defaults = cfg.get("defaults", {})
    assert "log_level" in defaults, f"{rel}: defaults.log_level 누락"
    assert isinstance(defaults["log_level"], int), f"{rel}: log_level 은 int 여야 함"


@pytest.mark.regression
@pytest.mark.parametrize("rel", ENRICHMENT_CONFIGS)
def test_paddle_ocr_endpoint_under_paddle(repo_root, rel):
    """ocr_endpoint 는 ocr.paddle 하위에 위치한다(옵션 위치 변경 회귀 방지)."""
    cfg = _load(repo_root, rel)
    paddle = cfg.get("ocr", {}).get("paddle", {})
    assert "ocr_endpoint" in paddle, f"{rel}: ocr.paddle.ocr_endpoint 누락"
    assert isinstance(paddle["ocr_endpoint"], str)
    # 한 단계 위(ocr.ocr_endpoint)로 되돌아가지 않았는지 확인
    assert "ocr_endpoint" not in cfg.get("ocr", {}), f"{rel}: ocr_endpoint 가 ocr 직하로 회귀함"


@pytest.mark.regression
@pytest.mark.parametrize("rel", ENRICHMENT_CONFIGS)
def test_layout_max_completion_tokens_int(repo_root, rel):
    cfg = _load(repo_root, rel)
    genos = cfg.get("layout", {}).get("genos_layout", {})
    assert "max_completion_tokens" in genos, f"{rel}: layout.genos_layout.max_completion_tokens 누락"
    assert isinstance(genos["max_completion_tokens"], int)
    assert genos["max_completion_tokens"] > 0


# ── enrichment 프롬프트 구조 고정 ──────────────────────────────────────────────

@pytest.mark.regression
@pytest.mark.parametrize("rel", ENRICHMENT_CONFIGS)
def test_toc_prompt_structure(repo_root, rel):
    ec = _parse_enrichment(repo_root, rel)
    if not ec.toc.do_toc:
        pytest.skip(f"{rel}: toc 비활성")
    assert ec.toc.system_prompt, f"{rel}: toc.system_prompt 가 비어있음"
    assert ec.toc.user_prompt, f"{rel}: toc.user_prompt 가 비어있음"
    assert any(p in ec.toc.user_prompt for p in _PLACEHOLDERS), \
        f"{rel}: toc.user_prompt 에 raw_text 치환자가 없음"


@pytest.mark.regression
@pytest.mark.parametrize("rel", ENRICHMENT_CONFIGS)
def test_metadata_prompt_structure(repo_root, rel):
    ec = _parse_enrichment(repo_root, rel)
    if not ec.metadata.do_metadata:
        pytest.skip(f"{rel}: metadata 비활성")
    assert ec.metadata.system_prompt, f"{rel}: metadata.system_prompt 가 비어있음"
    assert ec.metadata.user_prompt, f"{rel}: metadata.user_prompt 가 비어있음"
    assert any(p in ec.metadata.user_prompt for p in _PLACEHOLDERS), \
        f"{rel}: metadata.user_prompt 에 raw_text 치환자가 없음"
    assert isinstance(ec.metadata.parser, dict)
    assert str(ec.metadata.parser.get("type", "json")).lower() in {"json", "python"}
    assert isinstance(ec.metadata.output_fields, list) and ec.metadata.output_fields


# ── attachment 은 enrichment 미사용 ────────────────────────────────────────────

@pytest.mark.regression
@pytest.mark.parametrize("rel", ATTACHMENT_CONFIGS)
def test_attachment_has_no_enrichment(repo_root, rel):
    cfg = _load(repo_root, rel)
    assert cfg.get("enrichment") is None, f"{rel}: attachment 에 enrichment 가 생기면 안 됨"
