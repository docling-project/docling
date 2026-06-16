"""분할(Split, carry-over refine) TOC 추출 단위 테스트.

분할 옵션(`toc.split.enabled`)이 켜지면 문서를 페이지(N개) 단위로 청크화하고, 모든 청크에서
설정 프롬프트(enrichment.toc)를 사용하되 2번째 청크부터는 직전까지 누적된 outline을
프롬프트 앞에 컨텍스트로 덧붙여 순차 추출한다. 옵션이 꺼져 있으면 기존 단일 호출로 동작한다.

네트워크 비의존: PromptManager.call_ai_model 을 mock 한다. docling 코어 import 가
불가한 환경에서는 모듈 전체를 skip 한다(precheck 단위테스트와 동일 가정).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

try:
    from docling.utils.document_enrichment import DocumentEnrichmentUtils
    from docling.prompts.prompt_manager import LLMApiError
    from docling.datamodel.pipeline_options import DataEnrichmentOptions
except Exception:  # noqa: BLE001 - 환경 의존 import 실패
    pytest.skip("docling import 불가(환경 의존)", allow_module_level=True)


def _make_util(**opts):
    """do_toc_enrichment=True 로 prompt_manager 가 초기화된 util 생성."""
    options = DataEnrichmentOptions(do_toc_enrichment=True, **opts)
    return DocumentEnrichmentUtils(options)


def _fake_document(lines):
    """document.texts 형태의 최소 가짜 문서(페이지 정보 없음)."""
    return SimpleNamespace(texts=[SimpleNamespace(text=l, prov=[]) for l in lines])


def _doc_with_pages(items):
    """(page_no|None, text) 튜플 리스트로 prov.page_no를 가진 가짜 문서 생성."""
    texts = []
    for page_no, text in items:
        prov = [] if page_no is None else [SimpleNamespace(page_no=page_no)]
        texts.append(SimpleNamespace(text=text, prov=prov))
    return SimpleNamespace(texts=texts)


# ── _chunk_by_pages ────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestChunkByPages:
    def test_groups_n_pages_per_chunk(self):
        util = _make_util()
        # 4개 페이지, 페이지당 1줄 → pages_per_chunk=2 면 2청크
        doc = _doc_with_pages([(1, "p1"), (2, "p2"), (3, "p3"), (4, "p4")])
        chunks = util._chunk_by_pages(doc, pages_per_chunk=2, page_overlap=0)
        assert len(chunks) == 2
        assert "p1" in chunks[0] and "p2" in chunks[0]
        assert "p3" in chunks[1] and "p4" in chunks[1]

    def test_page_overlap_repeats_boundary_page(self):
        util = _make_util()
        doc = _doc_with_pages([(1, "p1"), (2, "p2"), (3, "p3"), (4, "p4")])
        no_ov = util._chunk_by_pages(doc, pages_per_chunk=2, page_overlap=0)
        ov = util._chunk_by_pages(doc, pages_per_chunk=2, page_overlap=1)
        # overlap 으로 경계 페이지가 다음 청크에 중복 → 총 청크 수 증가
        assert len(ov) > len(no_ov)

    def test_missing_prov_carries_forward_page(self):
        util = _make_util()
        # prov 없는 item 은 직전 페이지로 귀속
        doc = _doc_with_pages([(1, "p1a"), (None, "p1b"), (2, "p2")])
        chunks = util._chunk_by_pages(doc, pages_per_chunk=1, page_overlap=0)
        # 페이지1: p1a + p1b, 페이지2: p2 → 2청크
        assert len(chunks) == 2
        assert "p1a" in chunks[0] and "p1b" in chunks[0]
        assert "p2" in chunks[1]

    def test_empty_pages_are_skipped(self):
        util = _make_util()
        # 페이지 1, 3 에만 텍스트(2는 비어있음). pages_per_chunk=1 → 빈 창(2) skip
        doc = _doc_with_pages([(1, "p1"), (3, "p3")])
        chunks = util._chunk_by_pages(doc, pages_per_chunk=1, page_overlap=0)
        assert len(chunks) == 2
        assert "p1" in chunks[0] and "p3" in chunks[1]

    def test_overlap_clamped_prevents_infinite_loop(self):
        util = _make_util()
        doc = _doc_with_pages([(i, f"p{i}") for i in range(1, 6)])
        # page_overlap >= pages_per_chunk 여도 무한루프 없이 종료
        chunks = util._chunk_by_pages(doc, pages_per_chunk=2, page_overlap=5)
        assert len(chunks) >= 1
        joined = "\n".join(chunks)
        for i in range(1, 6):
            assert f"p{i}" in joined


# ── _is_token_overflow ─────────────────────────────────────────────────────────

@pytest.mark.unit
class TestIsTokenOverflow:
    def test_status_400_is_overflow(self):
        util = _make_util()
        assert util._is_token_overflow(LLMApiError("bad", status_code=400)) is True

    def test_message_signal_is_overflow(self):
        util = _make_util()
        assert util._is_token_overflow(
            LLMApiError("maximum context length exceeded", status_code=500)) is True

    def test_korean_signal_is_overflow(self):
        util = _make_util()
        assert util._is_token_overflow(
            LLMApiError("프롬프트 입력 토큰 초과 하였습니다", status_code=None)) is True

    def test_non_overflow_error(self):
        util = _make_util()
        assert util._is_token_overflow(
            LLMApiError("connection reset", status_code=502)) is False


# ── _build_continuation_user ───────────────────────────────────────────────────

@pytest.mark.unit
class TestBuildContinuationUser:
    """{prior_toc} 없는 커스텀 md용 fallback prepend (base_user, prior) 2-arg."""

    def test_prepends_prior_and_preserves_base(self):
        util = _make_util()
        base = "원래 지시문 {{raw_text}}"
        prior = "TITLE:DOC\n1. 개요 {중괄호}"
        out = util._build_continuation_user(base, prior)
        # 누적 목차 컨텍스트 + 분석 금지 지시 + base 보존
        assert "누적 목차" in out
        assert "분석" in out
        assert base in out
        # prior 항목 포함 + 중괄호 이스케이프(format 안전)
        assert "1. 개요" in out
        assert "{{중괄호}}" in out

    def test_none_base_is_safe(self):
        util = _make_util()
        out = util._build_continuation_user(None, "1. a")
        assert "누적 목차" in out


# ── _extract_toc_split (carry-over refine chain) ───────────────────────────────

@pytest.mark.unit
class TestExtractTocSplit:
    def test_carryover_injected_into_continuation_calls(self):
        util = _make_util()
        document = _fake_document(["t1", "t2", "t3", "t4"])
        calls = []

        def fake_call(category, prompt_type, raise_on_error=False, raw_text=None,
                      custom_system=None, custom_user=None, **kw):
            calls.append({"prompt_type": prompt_type, "custom_user": custom_user})
            n = len(calls)
            return f"<toc>TITLE:DOC\n1. sec{n}</toc>" if n == 1 else f"<toc>1. sec{n}</toc>"

        with patch.object(DocumentEnrichmentUtils, "_chunk_by_pages",
                          return_value=["chunk0", "chunk1"]), \
             patch.object(util.prompt_manager, "call_ai_model", side_effect=fake_call):
            result = util._extract_toc_split(
                document, prompt_type="korean_document",
                custom_system=None, custom_user=None,
            )

        assert len(calls) == 2
        # 모든 청크가 동일한 base prompt_type 사용(_continuation 아님)
        assert calls[0]["prompt_type"] == "korean_document"
        assert calls[1]["prompt_type"] == "korean_document"
        # 첫 호출엔 carry-over 없음, 두번째 호출 custom_user 에 누적 목차 주입
        assert calls[0]["custom_user"] is None
        assert calls[1]["custom_user"] is not None
        assert "누적 목차" in calls[1]["custom_user"]
        assert "sec1" in calls[1]["custom_user"]
        assert result is not None and "TITLE:DOC" in result

    def test_unified_md_with_prior_toc_injects_via_placeholder(self):
        """통합 md(= {prior_toc} 보유)면 모든 청크가 동일 md 사용 + prior_toc kwarg 주입."""
        util = _make_util()
        document = _fake_document(["t1", "t2"])
        unified_md = "통합 TOC 프롬프트\n<previous_outline>\n{prior_toc}\n</previous_outline>\n문서:\n{{raw_text}}"
        calls = []

        def fake_call(category, prompt_type, raise_on_error=False, raw_text=None,
                      custom_system=None, custom_user=None, prior_toc=None, **kw):
            calls.append({"custom_user": custom_user, "prior_toc": prior_toc})
            return "<toc>TITLE:D\n1. a</toc>" if len(calls) == 1 else "<toc>1. b</toc>"

        with patch.object(DocumentEnrichmentUtils, "_chunk_by_pages",
                          return_value=["c0", "c1"]), \
             patch.object(util.prompt_manager, "call_ai_model", side_effect=fake_call):
            util._extract_toc_split(
                document, prompt_type="korean_document",
                custom_system="sys", custom_user=unified_md,
            )
        # 첫 청크: md 그대로 + prior_toc=""
        assert calls[0]["custom_user"] == unified_md
        assert calls[0]["prior_toc"] == ""
        # 두번째 청크: md 그대로(자리표시자 주입) + prior_toc 채움
        assert calls[1]["custom_user"] == unified_md
        assert calls[1]["prior_toc"] and "a" in calls[1]["prior_toc"]

    def test_custom_md_without_prior_toc_falls_back_to_prepend(self):
        """{prior_toc} 없는 커스텀 md면 반복 시 코드가 prior 블록을 prepend(carry-over 유지)."""
        util = _make_util()
        document = _fake_document(["t1", "t2"])
        configured_user = "설정된 TOC 프롬프트 {{raw_text}}"
        calls = []

        def fake_call(category, prompt_type, raise_on_error=False, raw_text=None,
                      custom_system=None, custom_user=None, **kw):
            calls.append(custom_user)
            return "<toc>TITLE:D\n1. a</toc>" if len(calls) == 1 else "<toc>1. b</toc>"

        with patch.object(DocumentEnrichmentUtils, "_chunk_by_pages",
                          return_value=["c0", "c1"]), \
             patch.object(util.prompt_manager, "call_ai_model", side_effect=fake_call):
            util._extract_toc_split(
                document, prompt_type="korean_document",
                custom_system="sys", custom_user=configured_user,
            )
        # 첫 청크: 설정 프롬프트 그대로
        assert calls[0] == configured_user
        # 두번째 청크: prior 블록 prepend(설정 프롬프트 포함 + 누적 목차)
        assert configured_user in calls[1]
        assert "누적 목차" in calls[1]

    def test_overflow_chunk_skipped_others_kept(self):
        util = _make_util()
        document = _fake_document(["t1", "t2", "t3"])

        def fake_call(category, prompt_type, raise_on_error=False, raw_text=None,
                      custom_system=None, custom_user=None, **kw):
            if custom_user and "누적 목차" in custom_user:
                raise LLMApiError("토큰 초과", status_code=400)
            return "<toc>TITLE:DOC\n1. first</toc>"

        with patch.object(DocumentEnrichmentUtils, "_chunk_by_pages",
                          return_value=["c0", "c1"]), \
             patch.object(util.prompt_manager, "call_ai_model", side_effect=fake_call):
            result = util._extract_toc_split(
                document, prompt_type="korean_document",
                custom_system=None, custom_user=None,
            )
        assert result is not None and "first" in result

    def test_chunk_without_toc_block_is_skipped(self):
        """가드: <toc> 없는 응답(분석문/절단)은 병합 제외 → analysis 미주입, 청크0만 반영."""
        util = _make_util()
        document = _fake_document(["t1", "t2"])
        calls = []

        def fake_call(category, prompt_type, raise_on_error=False, raw_text=None,
                      custom_system=None, custom_user=None, **kw):
            calls.append(custom_user)
            if len(calls) == 1:
                return "<toc>TITLE:DOC\n1. 제1장 총칙</toc>"
            # continuation 이 <analysis> 만 내놓고 절단된 경우(<toc> 없음)
            return "<analysis>\n1. **분석**: 끝없는 추론...\n번호를 6? 7?"

        with patch.object(DocumentEnrichmentUtils, "_chunk_by_pages",
                          return_value=["c0", "c1"]), \
             patch.object(util.prompt_manager, "call_ai_model", side_effect=fake_call):
            result = util._extract_toc_split(
                document, prompt_type="korean_document",
                custom_system=None, custom_user=None,
            )
        assert result is not None
        assert "제1장 총칙" in result
        # analysis 텍스트가 목차로 주입되지 않음
        assert "분석" not in result
        assert "<analysis>" not in result

    def test_extract_toc_block_helper(self):
        util = _make_util()
        assert util._extract_toc_block("<toc>TITLE:A\n1. x</toc>") == "TITLE:A\n1. x"
        assert util._extract_toc_block("<analysis>no toc here</analysis>") is None
        assert util._extract_toc_block("") is None
        assert util._extract_toc_block(None) is None

    def test_extract_toc_block_ignores_inline_mention_in_analysis(self):
        """분석문이 <toc>를 인라인 언급해도, 실제 마지막 <toc>...</toc> 블록만 추출."""
        util = _make_util()
        piece = (
            "<analysis>\n"
            "Do not output <analysis>. Output <toc> only. No TITLE.\n"
            "Wait, the previous outline had 3.3 ...\n"
            "</analysis>\n"
            "<toc>\n1. 제4장 결산\n1.1. 제77조(손익금의 이체)\n</toc>"
        )
        block = util._extract_toc_block(piece)
        assert block == "1. 제4장 결산\n1.1. 제77조(손익금의 이체)"
        # 분석문 텍스트가 추출 블록에 섞이지 않음
        assert "previous outline" not in block
        assert "Output <toc> only" not in block

    def test_split_merge_excludes_analysis_when_toc_present(self):
        """분석문 + 말미 정상 <toc> 인 continuation 응답 → 병합 결과에 분석문 미포함."""
        util = _make_util()
        document = _fake_document(["t1", "t2"])

        responses = [
            "<toc>TITLE:DOC\n1. 제1장 총칙</toc>",
            ("<analysis>\nOutput <toc> only. Wait, the previous outline had ...\n"
             "-   1.1. 제71조(원화본지점 환대사)\n</analysis>\n"
             "<toc>\n1. 제4절 환대사\n1.1. 제71조(원화본지점 환대사)\n</toc>"),
        ]
        seq = iter(responses)

        def fake_call2(category, prompt_type, raise_on_error=False, raw_text=None,
                       custom_system=None, custom_user=None, **kw):
            return next(seq)

        with patch.object(DocumentEnrichmentUtils, "_chunk_by_pages",
                          return_value=["c0", "c1"]), \
             patch.object(util.prompt_manager, "call_ai_model", side_effect=fake_call2):
            result = util._extract_toc_split(
                document, prompt_type="korean_document",
                custom_system=None, custom_user=None,
            )
        assert result is not None
        assert "제1장 총칙" in result and "제4절 환대사" in result
        # 분석문 라인 미주입
        assert "previous outline" not in result
        assert "Output <toc>" not in result

    def test_non_overflow_error_propagates(self):
        util = _make_util()
        document = _fake_document(["t1", "t2"])

        def fake_call(**kw):
            raise LLMApiError("connection reset", status_code=502)

        with patch.object(DocumentEnrichmentUtils, "_chunk_by_pages",
                          return_value=["c0", "c1"]), \
             patch.object(util.prompt_manager, "call_ai_model", side_effect=fake_call):
            with pytest.raises(LLMApiError):
                util._extract_toc_split(
                    document, prompt_type="korean_document",
                    custom_system=None, custom_user=None,
                )


# ── apply_toc_enrichment: 분할 옵션 모드 분기 ──────────────────────────────────

@pytest.mark.unit
class TestApplyTocSplitMode:
    def test_split_enabled_runs_split_path(self):
        """split ON 이면 길이/초과와 무관하게 _extract_toc_split 경로를 탄다."""
        util = _make_util(toc_split_enabled=True)
        document = _fake_document(["t1", "t2"])

        with patch.object(DocumentEnrichmentUtils, "_extract_toc_split",
                          return_value="") as split_mock, \
             patch.object(util.prompt_manager, "call_ai_model") as call_mock:
            count = util.apply_toc_enrichment(document)

        split_mock.assert_called_once()
        call_mock.assert_not_called()  # 단일 호출 경로 미사용
        assert count == 0  # 빈 결과 → 문서 변형 없음

    def test_split_disabled_uses_single_call(self):
        """split OFF(기본) 이면 기존 단일 호출, _extract_toc_split 미호출."""
        util = _make_util()  # toc_split_enabled=None → False
        document = _fake_document(["t1", "t2"])

        with patch.object(util.prompt_manager, "call_ai_model",
                          return_value="") as call_mock, \
             patch.object(DocumentEnrichmentUtils, "_extract_toc_split") as split_mock:
            count = util.apply_toc_enrichment(document)

        call_mock.assert_called_once()
        split_mock.assert_not_called()
        assert count == 0

    def test_split_disabled_overflow_propagates(self):
        """split OFF + 단일 호출 토큰초과 → 기존처럼 LLMApiError 전파(fallback 없음)."""
        util = _make_util()
        document = _fake_document(["t1", "t2"])

        with patch.object(util.prompt_manager, "call_ai_model",
                          side_effect=LLMApiError("토큰 초과", status_code=400)), \
             patch.object(DocumentEnrichmentUtils, "_extract_toc_split") as split_mock:
            with pytest.raises(LLMApiError):
                util.apply_toc_enrichment(document)
        split_mock.assert_not_called()


# ── combine_windowed_toc 병합 정합성 ───────────────────────────────────────────

@pytest.mark.unit
class TestCombineWindowedToc:
    def test_dedupe_and_renumber(self):
        util = _make_util()
        prior = "TITLE:DOC\n1. 개요\n1.1. 목적"
        new = "1. 개요\n1. 범위"
        merged = util.combine_windowed_toc([prior, new])
        assert "TITLE:DOC" in merged
        assert "1. 개요" in merged
        assert "범위" in merged

    def test_boundary_overlap_articles_deduped(self):
        """page_overlap로 재추출된 경계 조항/섹션이 1회만 남고 신규는 보존(중복 제거)."""
        util = _make_util()
        # chunk1: 제3절~제40조까지. chunk2: overlap 재추출(제3절(계속), 제34~40조) + 신규 제41~42조.
        c1 = (
            "TITLE:보안업무 시행세칙\n"
            "1. 제3장 문서보안\n"
            "1.1. 제3절 비밀의 발송 및 접수\n"
            "1.1.1. 제34조 (비밀의 발송)\n"
            "1.1.2. 제38조 (비밀접수증)\n"
            "1.2. 제4절 비밀의 보관관리\n"
            "1.2.1. 제39조 (비밀의 보관)\n"
            "1.2.2. 제40조 (비밀보관책임자)"
        )
        c2 = (
            "1. 제3절 비밀의 발송 및 접수 (계속)\n"
            "1.1. 제34조 (비밀의 발송)\n"
            "1.2. 제38조 (비밀접수증)\n"
            "2. 제4절 비밀의 보관관리\n"
            "2.1. 제39조 (비밀의 보관)\n"
            "2.2. 제40조 (비밀보관책임자)\n"
            "2.3. 제41조 (비밀의 보관)\n"
            "2.4. 제42조 (비밀관리기록부)"
        )
        merged = util.combine_windowed_toc([c1, c2])
        # 경계 조항/섹션은 1회만
        assert merged.count("제34조") == 1
        assert merged.count("제40조") == 1
        assert merged.count("제4절 비밀의 보관관리") == 1
        assert merged.count("제3절 비밀의 발송 및 접수") == 1
        assert "(계속)" not in merged
        # 신규 항목은 보존
        assert "제41조" in merged and "제42조" in merged

    def test_legit_repeated_section_name_preserved(self):
        """장이 달라 정당하게 반복되는 일반 섹션명(제1절 통칙)은 보존(전역 dedup 금지)."""
        util = _make_util()
        c1 = (
            "TITLE:규정\n"
            "1. 제1장 총칙\n"
            "1.1. 제1절 통칙\n"
            "1.1.1. 제1조 (목적)\n"
            "1.1.2. 제2조 (정의)"
        )
        # 신규 창: 새로운 장의 '제1절 통칙'(정당한 반복) + 신규 조항. 경계 overlap 아님.
        c2 = (
            "1. 제2장 회계\n"
            "1.1. 제1절 통칙\n"
            "1.1.1. 제3조 (적용)"
        )
        merged = util.combine_windowed_toc([c1, c2])
        assert merged.count("제1절 통칙") == 2  # 정당한 반복 보존
        assert "제2장 회계" in merged and "제3조" in merged

    def test_byeolpyo_with_article_reference_not_dropped(self):
        """별표/별지 제목의 '(제N조 관련)' 참조 때문에 실제 조항과 충돌해 삭제되면 안 됨(regression)."""
        util = _make_util()
        c1 = (
            "TITLE:보안업무 시행세칙\n"
            "1. 제1장 총칙\n"
            "1.1. 제9조 (분야별 보안책임자 및 임무)\n"
            "1.2. 제40조 (비밀보관책임자)"
        )
        # 신규 창: 별표/별지가 '(제N조 관련)' 상호참조를 포함 — 그 조항과 충돌하면 안 됨.
        c2 = (
            "1. 부칙\n"
            "2. [별표 1] 분야별 보안책임자 및 임무(제9조 관련)\n"
            "3. [별표 2] 비밀 인계 인수(제40조 관련)\n"
            "4. [별지 제1호서식] 보안심사(실무)위원회 회의록(제5조 관련)"
        )
        merged = util.combine_windowed_toc([c1, c2])
        # 별표/별지가 보존되어야 한다(조항 참조로 오인되어 dedup 삭제되면 안 됨)
        assert "별표 1" in merged and "별표 2" in merged
        assert "별지 제1호서식" in merged
        # 실제 조항도 그대로
        assert "제9조" in merged and "제40조" in merged

    def test_multi_yakgwan_repeated_article_numbers_preserved(self):
        """다중 약관(보험약관): 약관마다 제1조~ 가 재시작 → 같은 조번호 반복은 모두 보존(전역 dedup 금지)."""
        util = _make_util()
        # 두 특약이 동일한 조번호(제1~3조)를 각각 가짐. 경계 overlap 아님(연속 시퀀스 겹침 없음).
        c1 = (
            "TITLE:보험약관\n"
            "1. 특약 약관: 무배당 보철치료특약\n"
            "1.1. 제 1 조 [목적]\n"
            "1.2. 제 2 조 [용어의 정의]\n"
            "1.3. 제 3 조 [보험금의 지급사유]"
        )
        c2 = (
            "1. 특약 약관: 무배당 보존치료특약\n"
            "1.1. 제 1 조 [목적]\n"
            "1.2. 제 2 조 [용어의 정의]\n"
            "1.3. 제 3 조 [보존치료의 정의]"
        )
        merged = util.combine_windowed_toc([c1, c2])
        # 두 특약 모두 보존 + 각자의 제1~3조 보존(붕괴 없음)
        assert "무배당 보철치료특약" in merged and "무배당 보존치료특약" in merged
        assert merged.count("제 1 조") == 2  # 약관마다 제1조 → 2회 보존
        assert merged.count("제 3 조") == 2
        assert "[보존치료의 정의]" in merged  # 후순위 특약 고유 조도 보존

    def test_boundary_overlap_with_repeated_numbers_still_stitched(self):
        """경계 overlap(연속 시퀀스 동일)은 조번호가 반복돼도 그 겹침 구간만 제거."""
        util = _make_util()
        c1 = (
            "TITLE:보험약관\n"
            "1. 특약 약관: 무배당 보존치료특약\n"
            "1.1. 제 1 조 [목적]\n"
            "1.2. 제 2 조 [용어의 정의]\n"
            "1.3. 제 3 조 [보존치료의 정의]"
        )
        # overlap: c2 head가 c1 tail(제2조,제3조)을 그대로 재추출 후 신규(제4조) 이어감.
        c2 = (
            "1. 제 2 조 [용어의 정의]\n"
            "2. 제 3 조 [보존치료의 정의]\n"
            "3. 제 4 조 [보험금의 지급]"
        )
        merged = util.combine_windowed_toc([c1, c2])
        assert merged.count("제 2 조") == 1  # 경계 겹침 제거
        assert merged.count("제 3 조") == 1
        assert "제 4 조" in merged  # 신규 보존


@pytest.mark.unit
class TestTocItemKey:
    """별표/별지 제목의 (제N조 관련) 참조를 조항으로 오분류하지 않는다(키 우선순위/앵커)."""

    def test_byeolpyo_key_ignores_article_reference(self):
        k = DocumentEnrichmentUtils._toc_item_key(
            {"title": "[별표 1] 분야별 보안책임자 및 임무(제9조 관련)", "full_text": ""})
        assert k == "BP:별표1"

    def test_byeolji_key_ignores_article_reference(self):
        k = DocumentEnrichmentUtils._toc_item_key(
            {"title": "[별지 제1호서식] 보안심사(실무)위원회 회의록(제5조 관련)", "full_text": ""})
        assert k == "BJ:별지제1호"

    def test_article_key_anchored_at_start(self):
        assert DocumentEnrichmentUtils._toc_item_key(
            {"title": "제40조 (비밀보관책임자)", "full_text": ""}) == "JO:제40조"
        assert DocumentEnrichmentUtils._toc_item_key(
            {"title": "제31조의2 (재분류의 표시)", "full_text": ""}) == "JO:제31조의2"

    def test_section_key_is_title(self):
        assert DocumentEnrichmentUtils._toc_item_key(
            {"title": "제4절 비밀의 보관관리", "full_text": ""}) == "T:제4절비밀의보관관리"
