"""enrichment 설정 + YAML 파싱 단위 테스트.

intelligent_processor / parser_processor / attachment_processor 가 사용하는 yaml 설정과
그로부터 만들어지는 EnrichmentConfig 를 검증한다.

이 파일은 순수 로직(facade.enrichment.* 는 stdlib 만 의존)만 import 하므로 docling/httpx
등 무거운 의존성이나 내부 서버 요청 없이 어디서든(로컬/GitHub CI) 실행된다.
"""

from pathlib import Path

import pytest
import yaml

from facade.enrichment.enrichment_config import EnrichmentConfig, _MetadataConfig, _TocConfig
from facade.enrichment.field_transforms import DEFAULT_METADATA_FIELD_TRANSFORMS


# 실제 배포되는 설정 파일들 (repo_root = genon/preprocessor 기준 상대 경로)
SHIPPED_CONFIGS = [
    "resource_dev/intelligent_processor_config.yaml",
    "resource_dev/parser_processor_config.yaml",
    "resource_dev/attachment_processor_config.yaml",
    "resource_dev/convert_processor_config.yaml",
    "resource/intelligent_processor_config.yaml",
    "resource/parser_processor_config.yaml",
    "resource/attachment_processor_config.yaml",
    "resource/convert_processor_config.yaml",
]

# enrichment 섹션이 있는(벡터/메타데이터 추출) 설정들
ENRICHMENT_CONFIGS = [
    "resource_dev/intelligent_processor_config.yaml",
    "resource_dev/parser_processor_config.yaml",
    "resource_dev/convert_processor_config.yaml",
    "resource/intelligent_processor_config.yaml",
    "resource/parser_processor_config.yaml",
    "resource/convert_processor_config.yaml",
]


def _load_cfg(repo_root: Path, rel: str) -> dict:
    path = repo_root / rel
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_enrichment(repo_root: Path, rel: str) -> EnrichmentConfig:
    """processor.__init__ 과 동일하게 enrichment 섹션을 EnrichmentConfig 로 파싱."""
    path = repo_root / rel
    cfg = _load_cfg(repo_root, rel)
    return EnrichmentConfig.from_raw(cfg.get("enrichment"), path.parent, parent_cfg=cfg)


# ── 배포 설정 파일 유효성 ──────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("rel", SHIPPED_CONFIGS)
def test_shipped_config_is_valid_yaml_mapping(repo_root, rel):
    """배포 yaml 들이 valid YAML 이며 최상위가 dict(mapping) 인지 확인."""
    path = repo_root / rel
    if not path.exists():
        pytest.skip(f"config not present: {rel}")
    cfg = _load_cfg(repo_root, rel)
    assert isinstance(cfg, dict), f"{rel} 최상위는 mapping 이어야 함"


@pytest.mark.unit
@pytest.mark.parametrize("rel", ENRICHMENT_CONFIGS)
def test_shipped_enrichment_parses_without_error(repo_root, rel):
    """enrichment 섹션이 EnrichmentConfig 로 예외 없이 파싱되고 타입이 일관되는지."""
    path = repo_root / rel
    if not path.exists():
        pytest.skip(f"config not present: {rel}")
    ec = _parse_enrichment(repo_root, rel)

    assert isinstance(ec, EnrichmentConfig)
    assert isinstance(ec.toc, _TocConfig)
    assert isinstance(ec.metadata, _MetadataConfig)
    assert isinstance(ec.toc.do_toc, bool)
    assert isinstance(ec.metadata.do_metadata, bool)
    assert isinstance(ec.metadata.output_fields, list)
    assert isinstance(ec.metadata.field_transforms, list)
    assert isinstance(ec.metadata.parser, dict)
    assert isinstance(ec.custom_fields_cfgs, list)
    assert isinstance(ec.image_description_cfg, dict)


@pytest.mark.unit
@pytest.mark.parametrize("rel", ENRICHMENT_CONFIGS)
def test_shipped_prompt_files_resolve_non_empty(repo_root, rel):
    """배포 config 의 prompt 가 .md 파일에서 정상 로드되어 비어있지 않은지.

    prompt 를 별도 .md 로 분리한 뒤 파일 누락/오타가 있으면 from_raw 가 즉시
    FileNotFoundError 를 던지므로, 이 테스트는 모든 prompt 파일의 존재/가독성도 함께 검증한다.
    """
    path = repo_root / rel
    if not path.exists():
        pytest.skip(f"config not present: {rel}")
    ec = _parse_enrichment(repo_root, rel)
    # toc/metadata 가 활성인 경우 resolve 된 prompt 가 비어있지 않아야 한다.
    if ec.toc.do_toc:
        assert ec.toc.system_prompt and ec.toc.system_prompt.strip()
        assert ec.toc.user_prompt and ec.toc.user_prompt.strip()
    if ec.metadata.has_custom_metadata:
        assert ec.metadata.system_prompt and ec.metadata.system_prompt.strip()
        assert ec.metadata.user_prompt and ec.metadata.user_prompt.strip()


# ── intelligent_processor 설정 ─────────────────────────────────────────────────

@pytest.mark.unit
def test_intelligent_dev_enrichment_values(repo_root):
    """intelligent dev 설정: toc/metadata enable, output_fields, field_transforms 미지정(=[])."""
    rel = "resource_dev/intelligent_processor_config.yaml"
    if not (repo_root / rel).exists():
        pytest.skip(f"config not present: {rel}")
    ec = _parse_enrichment(repo_root, rel)

    assert ec.toc.do_toc is True
    assert ec.metadata.do_metadata is True
    assert ec.metadata.output_fields == ["created_date", "authors"]
    # custom metadata enricher 활성 조건은 has_custom_metadata (output_fields 만으로도 opt-in)
    assert ec.metadata.has_custom_metadata is True
    assert ec.metadata.system_prompt  # file 또는 default 로 resolve 되어 비어있지 않음
    # yaml 에서 field_transforms 는 주석 처리되어 있어 빈 list 로 파싱되고,
    # processor.__init__ 에서 DEFAULT 로 대체된다([] or DEFAULT == DEFAULT).
    assert ec.metadata.field_transforms == []
    assert (ec.metadata.field_transforms or DEFAULT_METADATA_FIELD_TRANSFORMS) == DEFAULT_METADATA_FIELD_TRANSFORMS
    # image_description enable 값은 배포 설정마다 다를 수 있어 존재/타입만 확인
    assert isinstance(ec.image_description_cfg.get("enabled"), bool)


# ── parser_processor 설정 ──────────────────────────────────────────────────────

@pytest.mark.unit
def test_parser_dev_enrichment_output_fields_match_prompt(repo_root):
    """parser dev 설정: metadata output_fields 가 영문 default 프롬프트(created_date/authors)와 일치하는지.

    (prompt md 분리 후 영문 default 프롬프트로 통일 — output_fields 도 영문 키여야 추출이 정상 동작.)
    """
    rel = "resource_dev/parser_processor_config.yaml"
    if not (repo_root / rel).exists():
        pytest.skip(f"config not present: {rel}")
    ec = _parse_enrichment(repo_root, rel)

    assert ec.metadata.do_metadata is True
    assert ec.metadata.output_fields == ["created_date", "authors"]
    assert ec.toc.do_toc is True


# ── attachment_processor 설정 (enrichment 없음) ────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("rel", [
    "resource_dev/attachment_processor_config.yaml",
    "resource/attachment_processor_config.yaml",
])
def test_attachment_config_has_no_enrichment_section(repo_root, rel):
    """attachment 설정은 enrichment 를 쓰지 않는다(파서/청킹 전용)."""
    path = repo_root / rel
    if not path.exists():
        pytest.skip(f"config not present: {rel}")
    cfg = _load_cfg(repo_root, rel)
    assert cfg.get("enrichment") is None
    # chunking/loaders 등 attachment 고유 섹션 존재
    assert "chunking" in cfg or "defaults" in cfg


@pytest.mark.unit
def test_from_raw_handles_missing_enrichment_gracefully():
    """enrichment 가 없는 설정(None)에서도 EnrichmentConfig 가 안전하게 기본값을 만든다."""
    ec = EnrichmentConfig.from_raw(None, Path("."), parent_cfg={"chunking": {}})
    assert isinstance(ec, EnrichmentConfig)
    assert isinstance(ec.metadata, _MetadataConfig)
    assert ec.metadata.output_fields == []
    assert ec.metadata.field_transforms == []
    assert ec.custom_fields_cfgs == []


# ── EnrichmentConfig.from_raw 동작 (합성 입력) ─────────────────────────────────

@pytest.mark.unit
class TestEnrichmentConfigFromList:
    """Format B(list) enrichment 파싱."""

    def test_toc_and_metadata_enabled(self):
        ec = EnrichmentConfig.from_raw(
            [
                {"toc": {"enable": True, "url": "http://toc", "model": "m"}},
                {"metadata": {
                    "enable": True,
                    "url": "http://meta",
                    "model": "m",
                    "system_prompt": "sys",
                    "user_prompt": "usr {{raw_text}}",
                    "output_fields": ["created_date", "authors"],
                    "parser": {"type": "json"},
                    "pages": [1, 4],
                }},
            ],
            Path("."),
        )
        assert ec.toc.do_toc is True
        assert ec.toc.url == "http://toc"
        assert ec.metadata.do_metadata is True
        assert ec.metadata.output_fields == ["created_date", "authors"]
        assert ec.metadata.parser == {"type": "json"}
        assert ec.metadata.pages == [1, 4]
        assert ec.metadata.system_prompt == "sys"

    def test_enable_false_disables(self):
        ec = EnrichmentConfig.from_raw(
            [
                {"toc": {"enable": False}},
                {"metadata": {"enable": False}},
            ],
            Path("."),
        )
        assert ec.toc.do_toc is False
        assert ec.metadata.do_metadata is False

    def test_image_description_enabled_and_disabled(self):
        ec_on = EnrichmentConfig.from_raw(
            [{"image_description": {"enable": True, "url": "http://x", "model": "m"}}],
            Path("."),
        )
        assert ec_on.image_description_cfg.get("enabled") is True

        ec_off = EnrichmentConfig.from_raw(
            [{"image_description": {"enable": False}}],
            Path("."),
        )
        assert ec_off.image_description_cfg == {"enabled": False}

    def test_single_custom_fields_injects_resource_path(self):
        ec = EnrichmentConfig.from_raw(
            [{"custom_fields": {"enable": True, "output_fields": ["authors"], "system_prompt": "s"}}],
            Path("/tmp/cfgdir"),
        )
        assert len(ec.custom_fields_cfgs) == 1
        assert ec.custom_fields_cfgs[0]["resource_path"] == "/tmp/cfgdir"

    def test_multiple_custom_fields(self):
        ec = EnrichmentConfig.from_raw(
            [
                {"custom_fields": {"enable": True, "output_fields": ["a"], "system_prompt": "s"}},
                {"custom_fields": {"enable": True, "config_file": "custom_field_authors.yaml"}},
            ],
            Path("."),
        )
        assert len(ec.custom_fields_cfgs) == 2

    def test_disabled_custom_fields_excluded(self):
        ec = EnrichmentConfig.from_raw(
            [{"custom_fields": {"enable": False, "output_fields": ["a"]}}],
            Path("."),
        )
        assert ec.custom_fields_cfgs == []

    def test_field_transforms_parsed(self):
        transforms = [{"source": ["doc_date"], "target": "created_date", "type": "date_int"}]
        ec = EnrichmentConfig.from_raw(
            [{"metadata": {"enable": True, "system_prompt": "s",
                           "output_fields": ["doc_date"], "field_transforms": transforms}}],
            Path("."),
        )
        assert ec.metadata.field_transforms == transforms

    def test_enricher_name_aliases(self):
        """별칭(metadata_enricher, toc_enricher)도 정식 카테고리로 매핑된다."""
        ec = EnrichmentConfig.from_raw(
            [
                {"toc_enricher": {"enable": True, "url": "http://t"}},
                {"metadata_enricher": {"enable": True, "system_prompt": "s", "output_fields": ["x"]}},
            ],
            Path("."),
        )
        assert ec.toc.do_toc is True
        assert ec.metadata.do_metadata is True
        assert ec.metadata.output_fields == ["x"]


@pytest.mark.unit
class TestEnrichmentConfigFromDict:
    """Format A(dict) enrichment 파싱 + global fallback."""

    def test_dict_format_basic(self):
        ec = EnrichmentConfig.from_raw(
            {
                "api_url": "http://global",
                "api_key": "k",
                "model": "gm",
                "metadata": {"system_prompt": "s", "output_fields": ["created_date"]},
            },
            Path("."),
            parent_cfg={},
        )
        # metadata url 미지정 시 global fallback 사용
        assert ec.metadata.url == "http://global"
        assert ec.metadata.output_fields == ["created_date"]
        assert ec.api_url == "http://global"
        assert ec.model == "gm"

    def test_dict_field_transforms(self):
        transforms = [{"source": "d", "type": "date_int"}]
        ec = EnrichmentConfig.from_raw(
            {"metadata": {"system_prompt": "s", "field_transforms": transforms}},
            Path("."),
            parent_cfg={},
        )
        assert ec.metadata.field_transforms == transforms
