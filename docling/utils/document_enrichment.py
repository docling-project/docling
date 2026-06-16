# 0708 매칭 수정된 코드
import json
import logging
import re
import difflib
from typing import Dict, Any, Optional, List, Tuple
from copy import deepcopy
from difflib import SequenceMatcher

from docling_core.types.doc import (
    DocItemLabel, SectionHeaderItem, TextItem, DoclingDocument
)
from docling_core.types.doc.document import (
    GraphData, GraphCell, GraphCellLabel, ListItem
)
from docling.prompts import PromptManager
from docling.prompts.prompt_manager import LLMApiError
from docling.datamodel.pipeline_options import DataEnrichmentOptions

from collections import Counter

_log = logging.getLogger(__name__)


class DocumentEnrichmentUtils:
    """문서 enrichment 유틸리티 클래스 - TOC 추출 및 메타데이터 추출 기능 제공"""

    # Split(carry-over refine) TOC 추출 기본값. YAML/옵션이 None이면 아래 값 사용.
    _DEFAULT_PAGES_PER_CHUNK = 5
    _DEFAULT_PAGE_OVERLAP = 0
    _DEFAULT_CARRYOVER_MAX_TOKENS = 1500

    def __init__(self, enrichment_options: DataEnrichmentOptions):
        """
        Args:
            enrichment_options: 데이터 enrichment 옵션
        """
        self.enrichment_options = enrichment_options
        self.prompt_manager = None

        # 개별 기능이 하나라도 활성화되어 있으면 프롬프트 매니저 초기화
        if enrichment_options.do_toc_enrichment or enrichment_options.extract_metadata:
            self._initialize_prompt_manager()

        # 맨 앞의 괄호 블록( [], (), <> )과 나머지 텍스트를 분리하는 정규식
        self.BRACKET_TITLE_PATTERN = re.compile(
            r"""
            ^\s*
            (                                                # group 1: 괄호 블록 전체
                \[(?=[^\]]*(?:별지|별표))[^\]]*\]            # [ ... ] 내부에 '별지|별표'
                |
                \((?=[^)]*(?:별지|별표))[^)]*\)              # ( ... )
                |
                <(?=[^>]*(?:별지|별표))[^>]*>                # < ... >
            )
            \s*
            (.*)$                                            # group 2: 괄호 뒤 전체 제목
            """,
            re.VERBOSE,
        )

    def _initialize_prompt_manager(self):
        """프롬프트 매니저 초기화"""
        custom_prompts = self._build_custom_prompts()
        custom_api_configs = self._build_custom_api_configs()

        self.prompt_manager = PromptManager(
            custom_prompts=custom_prompts,
            custom_api_configs=custom_api_configs
        )

    def _build_custom_prompts(self) -> Dict[str, Any]:
        """사용자 정의 프롬프트 딕셔너리 구성"""
        custom_prompts = {}

        # TOC 관련 사용자 정의 프롬프트
        if (self.enrichment_options.toc_system_prompt or
            self.enrichment_options.toc_user_prompt):
            if "toc_extraction" not in custom_prompts:
                custom_prompts["toc_extraction"] = {}
            if "korean_document" not in custom_prompts["toc_extraction"]:
                custom_prompts["toc_extraction"]["korean_document"] = {}

            if self.enrichment_options.toc_system_prompt:
                custom_prompts["toc_extraction"]["korean_document"]["system"] = self.enrichment_options.toc_system_prompt

            if self.enrichment_options.toc_user_prompt:
                custom_prompts["toc_extraction"]["korean_document"]["user"] = self.enrichment_options.toc_user_prompt

        # 메타데이터 관련 사용자 정의 프롬프트
        if (self.enrichment_options.metadata_system_prompt or
            self.enrichment_options.metadata_user_prompt):
            if "metadata_extraction" not in custom_prompts:
                custom_prompts["metadata_extraction"] = {}
            if "korean_financial" not in custom_prompts["metadata_extraction"]:
                custom_prompts["metadata_extraction"]["korean_financial"] = {}

            if self.enrichment_options.metadata_system_prompt:
                custom_prompts["metadata_extraction"]["korean_financial"]["system"] = self.enrichment_options.metadata_system_prompt

            if self.enrichment_options.metadata_user_prompt:
                custom_prompts["metadata_extraction"]["korean_financial"]["user"] = self.enrichment_options.metadata_user_prompt

        return custom_prompts

    def _build_custom_api_configs(self) -> Dict[str, Dict[str, Any]]:
        """카테고리별 사용자 정의 API 설정 딕셔너리 구성"""
        custom_api_configs = {}

        # TOC API 설정
        if (self.enrichment_options.toc_api_provider or
            self.enrichment_options.toc_api_key or
            self.enrichment_options.toc_api_base_url or
            self.enrichment_options.toc_model or
            self.enrichment_options.toc_precheck_enabled is not None or
            self.enrichment_options.toc_max_context_tokens is not None or
            self.enrichment_options.toc_completion_reserved_tokens is not None or
            self.enrichment_options.toc_repetition_penalty is not None):

            toc_config = {}
            toc_config["provider"] = self.enrichment_options.toc_api_provider or "openrouter"
            toc_config["api_base_url"] = self.enrichment_options.toc_api_base_url or "https://openrouter.ai/api/v1"
            toc_config["model"] = self.enrichment_options.toc_model or "google/gemma-3-27b-it:free"

            if self.enrichment_options.toc_api_key:
                toc_config["api_key"] = self.enrichment_options.toc_api_key

            # TOC 선택적 파라미터들
            if self.enrichment_options.toc_temperature is not None:
                toc_config["temperature"] = self.enrichment_options.toc_temperature
            if self.enrichment_options.toc_top_p is not None:
                toc_config["top_p"] = self.enrichment_options.toc_top_p
            if self.enrichment_options.toc_seed is not None:
                toc_config["seed"] = self.enrichment_options.toc_seed
            if self.enrichment_options.toc_max_tokens is not None:
                toc_config["max_tokens"] = self.enrichment_options.toc_max_tokens
            if self.enrichment_options.toc_repetition_penalty is not None:
                toc_config["repetition_penalty"] = self.enrichment_options.toc_repetition_penalty
            if self.enrichment_options.toc_precheck_enabled is not None:
                toc_config["precheck_enabled"] = self.enrichment_options.toc_precheck_enabled
            if self.enrichment_options.toc_max_context_tokens is not None:
                toc_config["max_context_tokens"] = self.enrichment_options.toc_max_context_tokens
            if self.enrichment_options.toc_completion_reserved_tokens is not None:
                toc_config["completion_reserved_tokens"] = (
                    self.enrichment_options.toc_completion_reserved_tokens
                )

            custom_api_configs["toc_extraction"] = toc_config

        # Metadata API 설정
        if (self.enrichment_options.metadata_api_provider or
            self.enrichment_options.metadata_api_key or
            self.enrichment_options.metadata_api_base_url or
            self.enrichment_options.metadata_model or
            self.enrichment_options.metadata_precheck_enabled is not None or
            self.enrichment_options.metadata_max_context_tokens is not None or
            self.enrichment_options.metadata_completion_reserved_tokens is not None):

            metadata_config = {}
            metadata_config["provider"] = self.enrichment_options.metadata_api_provider or "openrouter"
            metadata_config["api_base_url"] = self.enrichment_options.metadata_api_base_url or "https://openrouter.ai/api/v1"
            metadata_config["model"] = self.enrichment_options.metadata_model or "google/gemma-3-27b-it:free"

            if self.enrichment_options.metadata_api_key:
                metadata_config["api_key"] = self.enrichment_options.metadata_api_key

            # Metadata 선택적 파라미터들
            if self.enrichment_options.metadata_temperature is not None:
                metadata_config["temperature"] = self.enrichment_options.metadata_temperature
            if self.enrichment_options.metadata_top_p is not None:
                metadata_config["top_p"] = self.enrichment_options.metadata_top_p
            if self.enrichment_options.metadata_seed is not None:
                metadata_config["seed"] = self.enrichment_options.metadata_seed
            if self.enrichment_options.metadata_max_tokens is not None:
                metadata_config["max_tokens"] = self.enrichment_options.metadata_max_tokens
            if self.enrichment_options.metadata_precheck_enabled is not None:
                metadata_config["precheck_enabled"] = self.enrichment_options.metadata_precheck_enabled
            if self.enrichment_options.metadata_max_context_tokens is not None:
                metadata_config["max_context_tokens"] = (
                    self.enrichment_options.metadata_max_context_tokens
                )
            if self.enrichment_options.metadata_completion_reserved_tokens is not None:
                metadata_config["completion_reserved_tokens"] = (
                    self.enrichment_options.metadata_completion_reserved_tokens
                )

            custom_api_configs["metadata_extraction"] = metadata_config
            custom_api_configs["document_checking"] = metadata_config # 문서 품질 검사도 같은 설정 사용

        return custom_api_configs

    def apply_toc_enrichment(self, document: DoclingDocument) -> int:
        """
        문서에 TOC enrichment 적용

        Args:
            document: DoclingDocument 객체

        Returns:
            int: 생성된 섹션 헤더 개수
        """
        if not self.enrichment_options.do_toc_enrichment or not self.prompt_manager:
            return 0

        try:
            _log.info("TOC 추출 시작... (apply_toc_enrichment)")

            # 원시 텍스트 추출
            raw_text = self._extract_raw_text_for_toc(document)

            # 사용자 정의 프롬프트 가져오기
            custom_system = self.enrichment_options.toc_system_prompt
            custom_user = self.enrichment_options.toc_user_prompt

            # 분할 옵션이 켜져 있으면 청크 분할 추출, 아니면 기존 단일 호출
            if self._split_enabled():
                _log.info("TOC 분할 추출 수행 (korean_document)")
                toc_content = self._extract_toc_split(
                    document,
                    prompt_type="korean_document",
                    custom_system=custom_system,
                    custom_user=custom_user,
                )
            else:
                toc_content = self.prompt_manager.call_ai_model(
                    category="toc_extraction",
                    prompt_type="korean_document",
                    custom_system=custom_system,
                    custom_user=custom_user,
                    raise_on_error=True,
                    raw_text=raw_text,
                    prior_toc=""
                )

            if toc_content:
                # 모든 SectionHeaderItem을 TextItem으로 변환
                self._convert_section_headers_to_text(document)
                # <toc> 블록만 추출(분석문 오염 방지). 블록이 없으면 원문 유지.
                toc_content = self.extract_content(toc_content)
                # 목차를 기반으로 SectionHeader 적용
                matched_count = self._apply_toc_to_document(document, toc_content)
                _log.info(f"TOC 추출 완료 - {matched_count}개 섹션 헤더 생성")
                return matched_count
            else:
                _log.warning("TOC 생성 실패")
                return 0

        except LLMApiError:
            raise
        except Exception as e:
            _log.error(f"TOC 추출 중 오류 발생: {str(e)}")
            return 0

    # 경계 stitch 시 직전 누적의 tail에서 비교할 항목 수(overlap 페이지 분량보다 넉넉히).
    _STITCH_TAIL_WINDOW = 80

    @staticmethod
    def _toc_item_key(item: Dict[str, Any]) -> str:
        """TOC 항목의 정규화된 식별 키.

        - 별표(별표 N)·별지(별지 제N호)·조항(제N조/제N조의M)은 규정 내 고유 식별자 → 전역 dedup용.
        - 그 외(장/절/관 섹션·일반 항목)는 제목 정규화 키(꼬리표 '(계속)/(continued)' 제거). 같은 이름의
          섹션이 장이 달라 정당하게 반복될 수 있으므로 전역 dedup엔 쓰지 않고 경계 stitch 비교에만 사용.

        주의: 별표/별지 제목에는 "(제N조 관련)" 상호참조가 흔하므로, 별표/별지를 조항보다 먼저 검사하고
        모든 식별자를 **제목 맨 앞(anchored)** 에서만 매칭한다(re.match). 그래야 "[별표 1] …(제9조 관련)"
        가 그 항목 자체인 별표 1로 키잉되고, 괄호 안 조항 참조(제9조)로 오분류되어 전역 dedup에 삭제되는
        regression을 막는다.
        """
        title = (item.get('title') or '').strip()
        m = re.match(r'\[?\s*별표\s*제?\s*(\d+)', title)
        if m:
            return 'BP:별표' + m.group(1)
        m = re.match(r'\[?\s*별지\s*제\s*(\d+)\s*호', title)
        if m:
            return 'BJ:별지제' + m.group(1) + '호'
        m = re.match(r'제\s*\d+\s*조(?:\s*의\s*\d+)?', title)
        if m:
            return 'JO:' + re.sub(r'\s+', '', m.group(0))
        base = re.sub(r'\s*[\(（]\s*(?:계속|continued)\s*[\)）]\s*$', '', title, flags=re.I)
        base = re.sub(r'\s+', '', base)
        return 'T:' + base if base else ''

    @staticmethod
    def _is_continuation_header(item: Dict[str, Any]) -> bool:
        """제목이 '…(계속)'/'…(continued)' 로 끝나는 재나열 헤더인지."""
        title = (item.get('title') or '')
        return bool(re.search(r'[\(（]\s*(?:계속|continued)\s*[\)）]\s*$', title, flags=re.I))

    def _overlap_prefix_len(
        self, acc: list, items: list, *, max_skip: int = 3, min_run: int = 1
    ) -> int:
        """경계 overlap으로 재추출된 `items`의 선행 구간 길이를 반환(제거 대상).

        'acc 의 suffix' == 'items 의 prefix' 가 되는 가장 긴 연속 시퀀스를 찾아 제거한다.
        키는 `_toc_item_key`(별표/별지/조 식별자 또는 정규화 제목).
        - 선행 건너뛰기(o)는 **'(계속)' 재나열 헤더에 한해서만** 허용한다. (임의 항목을 건너뛰면,
          다중 약관 문서에서 새 약관의 일반 조(제1조[목적]…)가 직전 약관의 동일 조와 우연히 일치해
          새 약관이 통째로 삭제되는 오류가 난다.) acc 의 tail 은 항상 '직전 내용의 끝'이고 새 head 는
          '겹침' 또는 '새 약관의 시작'이므로, o=0 매칭은 진짜 겹침에서만 성립한다.
        - min_run: 우연한 단발 일치 방지를 위한 최소 연속 길이(기본 1; suffix==prefix 자체가 안전).
        반환값 0이면 겹침 없음.
        """
        if not acc or not items:
            return 0
        acc_keys = [self._toc_item_key(x) for x in acc]
        new_keys = [self._toc_item_key(x) for x in items]
        window = self._STITCH_TAIL_WINDOW
        o = 0
        while o <= max_skip and o < len(items):
            max_len = min(len(acc), len(items) - o, window)
            for cand in range(max_len, min_run - 1, -1):
                if cand <= 0:
                    break
                if acc_keys[-cand:] == new_keys[o:o + cand]:
                    return o + cand
            # 다음 항목으로 진행은 현재 head 가 '(계속)' 재나열 헤더일 때만.
            if not self._is_continuation_header(items[o]):
                break
            o += 1
        return 0

    # ===== 유사도 기반 중복 제거 (경계 중복 완화) =====
    def _similar(self, a: str, b: str, thr: float = 0.92) -> bool:
        return SequenceMatcher(a=a.lower(), b=b.lower()).ratio() >= thr

    def _dedupe_items(self, items):
        """
        인접 또는 가까운 항목이 거의 같은 텍스트일 때 앞의 항목을 유지하고 뒤를 제거.
        같은 레벨이거나 레벨 차이가 1 이내일 때만 중복으로 간주.
        """
        # 'number': '',
        # 'title': cleaned_line,
        # 'level': 1,
        # 'full_text': cleaned_line

        deduped = []
        for item in items:
            number = item['number']
            title = item['title']
            level = item['level']
            full_text = item['full_text']
            if deduped:
                pnumber, ptitle, plevel, pfull_text = deduped[-1]
                if abs(plevel - level) <= 1 and self._similar(ptitle, title):
                    # 뒤 항목을 버리고 이전 것을 유지
                    continue
            deduped.append((number, title, level, full_text))
        return deduped

    # ===== 레벨 구조를 기반으로 번호 재생성 =====
    def _renumber(self, items) -> List[str]:
        """
        (level, heading) → "n.n.n. heading" 문자열 목록으로 재번호 부여.
        레벨은 1 이상. 역행 방지 및 비정상 레벨은 보정해 1로 시작하도록 맞춤.
        """
        out: List[str] = []
        counters: Dict[int, int] = {}

        # 가장 작은 레벨이 1이 아니면 전체를 내려서 시작을 1로 맞춤
        min_lvl = min((level for number, title, level, full_text in items), default=1)
        shift = (min_lvl - 1) if min_lvl > 1 else 0

        for number, title, level, full_text in items:
            L = max(1, level - shift)  # 보정
            # 상위 카운터 초기화/유지
            counters[L] = counters.get(L, 0) + 1
            # 하위 레벨 카운터는 제거
            for k in list(counters.keys()):
                if k > L:
                    del counters[k]
            # 번호 문자열 조립
            parts = [str(counters[i]) for i in range(1, L + 1)]
            out.append(f"{'.'.join(parts)}. {title}")
        return out

    def combine_windowed_toc(self, window_texts: List[str], *, joiner: str = "\n") -> str:
        """
        창별 응답 문자열들을 하나의 최종 TOC 문자열로 결합:
          1) TITLE 1회 채택
          2) 모든 항목 수집 → 경계 중복 제거 → 번호 재생성
        반환 포맷:
            TITLE:<제목> (있는 경우)
            1. ...
            1.1. ...
            2. ...
        """
        final_title: Optional[str] = None
        parsed_windows: List[list] = []

        for txt in window_texts:
            parsed_data = self._parse_toc_content(txt)
            title = parsed_data['title']
            if title and not final_title:
                final_title = title
            parsed_windows.append(parsed_data['toc_items'])

        # 경계 overlap stitch(시퀀스 기반): page_overlap으로 두 창에 걸친 페이지를 모델이 재추출하면,
        #   '직전 누적의 tail'과 '새 창의 head'가 동일 항목의 '연속 시퀀스'로 겹친다. 그 겹침 구간만 제거.
        #   - 전역 식별자 dedup은 쓰지 않는다(보험약관 등 다중 약관 문서는 약관마다 제1조~ 가 정당하게
        #     반복되므로 전역 1회화하면 후순위 약관의 조가 통째로 사라짐).
        #   - 시퀀스(연속) 일치만 제거하므로, 약관 경계에서 우연히 같은 조번호가 있어도(누적 tail은 늘
        #     '직전 내용의 끝'이고 새 head는 겹침 또는 새 약관의 '시작'이라) 오삭제되지 않는다.
        merged_items: list = []
        for items in parsed_windows:
            if not merged_items:
                merged_items.extend(items)
                continue
            drop = self._overlap_prefix_len(merged_items, items)
            merged_items.extend(items[drop:])

        collected = merged_items
        if not collected and not final_title:
            return ""

        # 인접 유사 중복 정리(경계 잔여) 후 번호 재생성
        deduped = self._dedupe_items(collected)
        # print("--- Deduped TOC Items ---")
        # print(deduped)
        # 번호 재생성
        renum = self._renumber(deduped)
        # print("--- Renumbered TOC Items ---")
        # print(renum)

        lines = []
        if final_title:
            lines.append(f"TITLE:{final_title}")
        lines.extend(renum)
        return joiner.join(lines)

    def extract_content(self, text: str) -> str:

        # -------------------------------------
        # <toc> 블록 추출 (분석문 내 인라인 <toc> 언급에 오염되지 않도록 robust 추출)
        # -------------------------------------
        block = self._extract_toc_block(text)
        if block is not None:
            return block

        return (text or "").strip()

    # ===== Split(carry-over refine) TOC 추출 =====
    def _split_enabled(self) -> bool:
        """분할 추출 수행 여부(명시적 옵션). 옵션이 None이면 기본 False(기존 단일 호출)."""
        enabled = self.enrichment_options.toc_split_enabled
        return bool(enabled) if enabled is not None else False

    def _is_token_overflow(self, err: LLMApiError) -> bool:
        """LLM 호출 에러가 토큰(컨텍스트 길이) 초과인지 판별."""
        if getattr(err, "status_code", None) == 400:
            return True
        msg = (getattr(err, "raw_error_message", "") or str(err)).lower()
        signals = [
            "초과",
            "context length",
            "maximum context",
            "context_length_exceeded",
            "too long",
            "reduce the length",
            "max_tokens",
        ]
        return any(s in msg for s in signals)

    def _estimate_tokens(self, text: str) -> int:
        """prompt_manager와 동일한 척도로 토큰 추정(없으면 문자 수 fallback)."""
        if self.prompt_manager is not None:
            return self.prompt_manager.estimate_tokens(text)
        return len(text or "")

    @staticmethod
    def _item_page_no(item, default_page_no: int) -> Optional[int]:
        """text item의 prov[0].page_no를 안전하게 얻는다(없으면 None)."""
        prov_list = getattr(item, "prov", None) or []
        if not prov_list:
            return None
        page_no = getattr(prov_list[0], "page_no", None)
        if isinstance(page_no, int) and page_no > 0:
            return page_no
        return None

    def _chunk_by_pages(
        self, document, pages_per_chunk: int, page_overlap: int
    ) -> List[str]:
        """문서를 페이지(N개) 단위로 청크화. 페이지 경계를 보존한다.

        - document.texts를 순회하며 각 item의 prov page_no로 페이지를 구해
          {page_no: [정규화된 줄]} 로 모은다. prov가 없거나 비정상이면 직전 페이지를
          carry-forward(초기 1)하여 reading order 상 자연스러운 페이지에 귀속시킨다.
        - 관측된 최소~최대 페이지 범위를 pages_per_chunk 창으로 끊어 각 창의 텍스트를 join.
          page_overlap 만큼 다음 창 시작을 앞당긴다(빈 창은 skip).
        """
        step = max(1, pages_per_chunk)
        overlap = max(0, page_overlap)
        if overlap >= step:
            overlap = step - 1  # 진행 보장(무한루프 방지)

        page_lines: Dict[int, List[str]] = {}
        last_page = 1
        for text in document.texts:
            cleaned = re.sub(r'\s+', ' ', text.text.strip())
            if not cleaned:
                continue
            pno = self._item_page_no(text, last_page)
            if pno is None:
                pno = last_page
            last_page = pno
            page_lines.setdefault(pno, []).append(cleaned)

        if not page_lines:
            return []

        first_page = min(page_lines)
        last_page_no = max(page_lines)

        chunks: List[str] = []
        start = first_page
        while start <= last_page_no:
            end = start + step - 1
            lines: List[str] = []
            for p in range(start, end + 1):
                lines.extend(page_lines.get(p, []))
            if lines:
                chunks.append("\n".join(lines))
            if end >= last_page_no:
                break
            start = end + 1 - overlap
        return chunks

    def _shrink_outline(self, toc: str, max_tokens: int) -> str:
        """누적 outline이 max_tokens를 넘으면 상위 레벨만 남겨 축약(계층 컨텍스트 유지)."""
        if not toc or max_tokens <= 0:
            return toc
        if self._estimate_tokens(toc) <= max_tokens:
            return toc

        lines = toc.split("\n")
        title_lines = [l for l in lines if l.strip().startswith("TITLE:")]
        body = [l for l in lines if not l.strip().startswith("TITLE:")]

        def _level(line: str) -> int:
            m = re.match(r'^(\d+(?:\.\d+)*)\.', line.strip())
            return m.group(1).count('.') + 1 if m else 1

        # 1) 레벨 2까지 → 2) 레벨 1까지 순으로 축약 시도
        for max_level in (2, 1):
            kept = [l for l in body if _level(l) <= max_level]
            candidate = "\n".join(title_lines + kept)
            if self._estimate_tokens(candidate) <= max_tokens:
                return candidate

        # 3) 최후: TITLE + 앞부분 줄을 토큰 예산까지만 유지
        kept = list(title_lines)
        acc_tok = self._estimate_tokens("\n".join(kept))
        for l in body:
            t = self._estimate_tokens(l) + 1
            if acc_tok + t > max_tokens:
                break
            kept.append(l)
            acc_tok += t
        return "\n".join(kept)

    def _build_continuation_user(self, base_user: Optional[str], prior: str) -> str:
        """`{prior_toc}` 자리표시자가 없는 커스텀 md용 fallback continuation 프롬프트.

        설정 md가 통합 프롬프트(=`{prior_toc}` 보유)가 아니면, 이어쓰기 지시 + 누적 목차 블록을
        base_user 앞에 prepend 하여 carry-over를 유지한다.
        - placeholder 주입 경로와 달리 prior가 템플릿 문자열에 직접 삽입되므로 중괄호를 이스케이프한다.
        - base_user의 `{{raw_text}}`/`{raw_text}` 자리표시자는 보존되어 format 단계에서 렌더된다.
        """
        prior_safe = (prior or "").replace("{", "{{").replace("}", "}}")
        preamble = (
            "이 문서는 더 긴 문서의 이어지는 부분입니다. 분석·설명·사고과정을 출력하지 말고, "
            "아래 '누적 목차'를 참고하여 문서에서 새로 등장하는 항목만 <toc>...</toc> 로 출력하세요. "
            "이미 누적된 항목은 다시 출력하지 말고, 번호는 1부터 다시 시작해도 됩니다"
            "(최종 번호는 후처리에서 재부여).\n\n"
            f"## 누적 목차\n{prior_safe}\n\n---\n\n"
        )
        return preamble + (base_user or "")

    def _extract_toc_split(
        self,
        document: DoclingDocument,
        prompt_type: str,
        custom_system: Optional[str],
        custom_user: Optional[str],
    ) -> Optional[str]:
        """긴 문서를 청크로 나눠 누적 outline(carry-over)을 이어받아 순차 TOC 추출.

        - 모든 청크에서 설정 프롬프트(prompt_type + custom_system/custom_user)를 사용한다.
        - 두 번째 이후 청크는 설정 user 프롬프트 앞에 직전까지 누적된 outline을 컨텍스트로
          덧붙여(carry-over) 계층/번호 일관성을 유지한다.
        - 매 스텝 결과를 combine_windowed_toc로 누적 병합(중복 제거 + 번호 재생성).
        """
        if self.prompt_manager is None:
            return None

        pages_per_chunk = (
            self.enrichment_options.toc_pages_per_chunk
            or self._DEFAULT_PAGES_PER_CHUNK
        )
        page_overlap = self.enrichment_options.toc_page_overlap
        if page_overlap is None:
            page_overlap = self._DEFAULT_PAGE_OVERLAP
        carryover_max = (
            self.enrichment_options.toc_carryover_max_tokens
            or self._DEFAULT_CARRYOVER_MAX_TOKENS
        )

        chunks = self._chunk_by_pages(document, pages_per_chunk, page_overlap)
        if not chunks:
            return None
        _log.info(
            f"TOC 분할 추출: {len(chunks)}개 청크 "
            f"(pages_per_chunk={pages_per_chunk}, page_overlap={page_overlap})"
        )

        acc = ""
        for i, chunk in enumerate(chunks):
            try:
                if i == 0:
                    # 첫 청크: 통합 md를 그대로 사용. prior_toc="" → 빈 누적 목차(첫 추출 모드).
                    piece = self.prompt_manager.call_ai_model(
                        category="toc_extraction",
                        prompt_type=prompt_type,
                        custom_system=custom_system,
                        custom_user=custom_user,
                        raise_on_error=True,
                        raw_text=chunk,
                        prior_toc="",
                    )
                else:
                    prior = self._shrink_outline(acc, carryover_max)
                    # 통합 md(= {prior_toc} 보유)면 자리표시자 주입, 아니면 코드가 prior 블록 prepend.
                    base_user = custom_user or self.prompt_manager.get_user_prompt_template(
                        "toc_extraction", prompt_type
                    )
                    if base_user and "{prior_toc}" in base_user:
                        continuation_user = base_user
                    else:
                        continuation_user = self._build_continuation_user(base_user, prior)
                    piece = self.prompt_manager.call_ai_model(
                        category="toc_extraction",
                        prompt_type=prompt_type,
                        custom_system=custom_system,
                        custom_user=continuation_user,
                        raise_on_error=True,
                        raw_text=chunk,
                        prior_toc=prior,
                    )
            except LLMApiError as e:
                # 청크 1개가 여전히 초과하면 해당 청크만 건너뛰고 누적 결과는 유지
                if self._is_token_overflow(e):
                    _log.warning(
                        f"TOC 분할 추출: 청크 {i + 1}/{len(chunks)} 토큰 초과로 건너뜀"
                    )
                    continue
                raise

            _log.debug(f"--- TOC Chunk {i + 1}/{len(chunks)} ---\n{piece}")

            # 가드: <toc> 블록이 없는 응답(분석문/절단 등)은 병합하지 않고 건너뜀.
            # (이전엔 extract_content가 원문 전체를 반환해 추론문이 목차로 주입되는 오염 발생)
            toc_block = self._extract_toc_block(piece)
            if toc_block is None:
                _log.warning(
                    f"TOC 분할 추출: 청크 {i + 1}/{len(chunks)} 유효한 <toc> 미생성"
                    f"(분석문/절단 추정) → 건너뜀"
                )
                continue

            acc = self.combine_windowed_toc([acc, toc_block])

        return acc or None

    @staticmethod
    def _extract_toc_block(piece: Optional[str]) -> Optional[str]:
        """응답에서 <toc>...</toc> 블록 내부 텍스트를 반환. 블록이 없으면 None.

        '마지막 </toc> 직전의 마지막 <toc>' 사이만 추출한다. 모델이 분석문(<analysis>) 안에서
        '<toc>'를 인라인 언급(예: "Output <toc> only")한 경우, 비탐욕 정규식은 그 인라인 언급부터
        실제 닫는 태그까지 한 번에 매칭해 분석문을 통째로 삼킨다(매치가 1개뿐이라 matches[-1]도 오염).
        rfind 기반으로 실제 블록만 안전하게 추출한다.
        """
        if not piece:
            return None
        low = piece.lower()
        end = low.rfind("</toc>")
        if end == -1:
            return None
        start = low.rfind("<toc>", 0, end)
        if start == -1:
            return None
        return piece[start + len("<toc>"):end].replace("```", "").strip()

    def apply_law_toc_enrichment(self, document: DoclingDocument) -> int:
        """
        문서에 TOC enrichment 적용

        Args:
            document: DoclingDocument 객체

        Returns:
            int: 생성된 섹션 헤더 개수
        """
        if not self.enrichment_options.do_toc_enrichment or not self.prompt_manager:
            return 0

        try:
            _log.info("TOC 추출 시작... (apply_law_toc_enrichment)")

            # 원시 텍스트 추출
            raw_text = self._extract_raw_text_for_toc(document)

            # 사용자 정의 프롬프트 가져오기
            custom_system = self.enrichment_options.toc_system_prompt
            custom_user = self.enrichment_options.toc_user_prompt

            # 분할 옵션이 켜져 있으면 청크 분할 추출, 아니면 기존 단일 호출
            if self._split_enabled():
                _log.info("TOC 분할 추출 수행 (law_document)")
                toc_content = self._extract_toc_split(
                    document,
                    prompt_type="law_document",
                    custom_system=custom_system,
                    custom_user=custom_user,
                )
            else:
                toc_content = self.prompt_manager.call_ai_model(
                    category="toc_extraction",
                    prompt_type="law_document",
                    custom_system=custom_system,
                    custom_user=custom_user,
                    raise_on_error=True,
                    raw_text=raw_text,
                    prior_toc=""
                )

            _log.debug(f"--- TOC ---\n{toc_content}")

            if toc_content:
                # 모든 SectionHeaderItem을 TextItem으로 변환
                self._convert_section_headers_to_text(document)

                toc_content = self.extract_content(toc_content)
                _log.debug(f"--- only TOC ---\n{toc_content}")

                # 목차를 기반으로 SectionHeader 적용
                matched_count = self._apply_toc_to_law_document(document, toc_content)
                _log.info(f"TOC 추출 완료 - {matched_count}개 섹션 헤더 생성")
                return matched_count
            else:
                _log.warning("TOC 생성 실패")
                return 0

        except LLMApiError:
            raise
        except Exception as e:
            _log.error(f"TOC 추출 중 오류 발생: {str(e)}")
            return 0

    def apply_metadata_enrichment(self, document: DoclingDocument, **kwargs: dict) -> bool:
        """
        문서에 메타데이터 enrichment 적용

        Args:
            document: DoclingDocument 객체

        Returns:
            bool: 메타데이터 추출 성공 여부
        """
        if not self.enrichment_options.extract_metadata or not self.prompt_manager:
            return False

        try:
            # 문서의 처음 4페이지에서 텍스트 추출
            temp_content = ""
            total_pages = len(document.pages)
            for page in range(1, min(5, total_pages + 1)):
                temp_content += document.export_to_markdown(page_no=page)

            metadata = self._extract_document_metadata_date(temp_content, **kwargs)
            if metadata:
                _log.info(f"추출된 메타데이터: {json.dumps(metadata, ensure_ascii=False, indent=2)}")

                # KeyValueItem 생성을 위한 GraphData 구성
                graph_cells = []
                cell_id = 0

                # 메타데이터 딕셔너리를 그대로 key-value로 변환
                for key, value in metadata.items():
                    graph_cells.append(GraphCell(
                        label=GraphCellLabel.KEY,
                        cell_id=cell_id,
                        text=key,
                        orig=key
                    ))
                    cell_id += 1

                    graph_cells.append(GraphCell(
                        label=GraphCellLabel.VALUE,
                        cell_id=cell_id,
                        text=json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value),
                        orig=json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
                    ))
                    cell_id += 1

                # GraphData 생성
                graph_data = GraphData(cells=graph_cells, links=[])

                # KeyValueItem을 문서에 추가
                document.add_key_values(
                    graph=graph_data,
                    prov=None,
                    parent=None
                )
                return True
            else:
                return False

        except LLMApiError:
            raise
        except Exception as e:
            _log.error(f"메타데이터 추출 중 오류 발생: {str(e)}")
            return False

    def check_document_with_good_text(self, document: DoclingDocument) -> bool:
        """
        문서의 텍스트 품질을 검사하여 OCR이 필요한지 판단

        Args:
            document: DoclingDocument 객체

        Returns:
            bool: OCR이 필요하지 않으면 True, 필요하면 False
        """
        if not self.prompt_manager:
            return 0

        def get_text_by_page(doc: DoclingDocument, last_page_no: int = 0):
            page_texts = ""

            for item, level in doc.iterate_items():
                if isinstance(item, TextItem) and hasattr(item, 'prov') and item.prov:
                    page_no = item.prov[0].page_no
                    if last_page_no != 0 and page_no <= last_page_no:
                        page_texts += item.text

            return page_texts

        try:
            _log.info("문서 품질 검사 시작...")

            page_texts = get_text_by_page(document, last_page_no=10)

            text  = page_texts
            if len(text) > 3000:
                # text = text[:3000] + "..."
                text = self._extract_substrings(text, length=1000)
            if len(text) == 0:
                return False  # 텍스트가 없으면 OCR 필요

            text_len = len(text)
            non_ascii_ratio = sum(1 for c in text if self._is_non_meaningful_char(c)) / text_len if text_len > 0 else 0
            space_ratio = text.count(' ') / text_len if text_len > 0 else 1.0

            response = self.prompt_manager.call_ai_model(
                category="document_checking",
                prompt_type="text_checking",
                content=text,
                text_len=text_len,
                non_ascii_ratio=non_ascii_ratio,
                space_ratio=space_ratio
            )
            if response:
                decision_match = re.search(r'<decision>\s*(YES|NO)\s*</decision>', response, re.IGNORECASE)
                if decision_match:
                    decision = decision_match.group(1).strip()
                else:
                    decision = "YES"
            else:
                decision = "YES"

            return False if decision == "YES" else True # OCR이 필요하지 않으면 True, 필요하면 False

        except Exception as e:
            _log.error(f"문서 품질 검사 중 오류 발생: {str(e)}")
            return False

    def _is_non_meaningful_char(self, c):
        # 공백은 제외
        if c.isspace():
            return False
        # 한글 (가-힣, ㄱ-ㅎ, ㅏ-ㅣ)
        if '가' <= c <= '힣' or 'ㄱ' <= c <= 'ㅎ' or 'ㅏ' <= c <= 'ㅣ':
            return False
        # 한자 (CJK Unified Ideographs)
        if '\u4e00' <= c <= '\u9fff':
            return False
        # 일본어 (가타카나, 히라가나)
        if '\u3040' <= c <= '\u30ff':
            return False
        # 숫자, 알파벳, 기본 문장 부호
        if c.isascii():
            return False
        # 위 조건에 해당하지 않는 문자는 비의미(non-meaningful)로 처리
        return True

    def _extract_substrings(self, text, length=1000):
        """
        원본 문자열에서 20%, 50%, 80% 위치의 부분 문자열을 추출

        Args:
            text (str): 원본 문자열
            length (int): 각 부분 문자열의 길이 (기본값: 1000)

        Returns:
            dict: 각 위치별 부분 문자열을 담은 딕셔너리
        """
        text_len = len(text)

        # 최소 길이 체크
        if text_len < length * 3:
            return text

        # 20%, 50%, 80% 위치 계산
        pos_20 = int(text_len * 0.2)
        pos_50 = int(text_len * 0.5)
        pos_80 = int(text_len * 0.8)

        # 각 위치를 중심으로 하는 구간의 시작점과 끝점 계산
        half_length = length // 2
        ranges = [
            (max(0, pos_20 - half_length), min(text_len, pos_20 + half_length)),     # 20% 중심
            (max(0, pos_50 - half_length), min(text_len, pos_50 + half_length)),     # 50% 중심
            (max(0, pos_80 - half_length), min(text_len, pos_80 + half_length))      # 80% 중심
        ]

        # 실제 길이가 요청된 길이와 다를 수 있으므로 조정
        for i in range(len(ranges)):
            start, end = ranges[i]
            actual_length = end - start

            # 길이가 부족한 경우 조정
            if actual_length < length:
                shortage = length - actual_length

                # 앞쪽으로 확장 가능한지 확인
                if start > 0:
                    extend_front = min(shortage, start)
                    start -= extend_front
                    shortage -= extend_front

                # 뒤쪽으로 확장 가능한지 확인
                if shortage > 0 and end < text_len:
                    extend_back = min(shortage, text_len - end)
                    end += extend_back

                ranges[i] = (start, end)

        # 중복 체크 및 조정
        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                start1, end1 = ranges[i]
                start2, end2 = ranges[j]

                # 겹치는 경우 뒤의 구간을 뒤로 이동
                if start2 < end1:
                    shift = end1 - start2
                    ranges[j] = (start2 + shift, end2 + shift)

        # 마지막 구간이 텍스트 길이를 초과하는지 체크
        if ranges[-1][1] > text_len:
            ranges[-1] = (ranges[-1][0], text_len)

        # 텍스트 추출
        result = ""

        for start, end in ranges:
            result += text[start:end] + "\n"

        return result

    def _convert_section_headers_to_text(self, document):
        """문서의 모든 SectionHeaderItem을 TextItem으로 변환"""
        new_texts = []

        for item in document.texts:
            if isinstance(item, SectionHeaderItem):
                new_item = TextItem(
                    self_ref=item.self_ref,
                    parent=item.parent,
                    children=item.children,
                    content_layer=item.content_layer,
                    label=DocItemLabel.TEXT,
                    prov=item.prov,
                    orig=item.orig,
                    text=item.text,
                    formatting=item.formatting,
                    hyperlink=getattr(item, 'hyperlink', None)
                )
                new_texts.append(new_item)
            else:
                new_texts.append(item)

        document.texts = new_texts

    def _extract_raw_text_for_toc(self, document):
        """문서에서 원시 텍스트 추출"""
        raw_texts = ""
        for text in document.texts:
            cleaned_text = re.sub(r'\s+', ' ', text.text.strip())
            raw_texts += cleaned_text + "\n"
        return raw_texts

    def _parse_toc_content(self, toc_content: str):
        """목차 내용을 파싱해서 구조화된 데이터로 변환 (문서 제목 포함)"""
        toc_items = []
        document_title = None
        lines = toc_content.split('\n')

        for line in lines:
            cleaned_line = line.strip()
            if not cleaned_line:
                continue

            # 문서 제목 추출
            if cleaned_line.startswith('TITLE:'):
                document_title = cleaned_line[6:].strip()  # 'TITLE: ' 제거
                continue

            # 숫자 패턴들 (레벨별 매칭)
            patterns = [
                r'^(\d+\.\d+\.\d+\.\d+)\.\s*(.+)$',   # 1.1.1.1. 제목 (4단계)
                r'^(\d+\.\d+\.\d+)\.\s*(.+)$',        # 1.1.1. 제목 (3단계)
                r'^(\d+\.\d+)\.\s*(.+)$',              # 1.1. 제목 (2단계)
                r'^(\d+)\.\s*(.+)$',                    # 1. 제목 (1단계)
            ]

            matched = False
            for pattern in patterns:
                match = re.match(pattern, cleaned_line)
                if match:
                    number_part = match.group(1)
                    title_part = match.group(2).strip()
                    level = number_part.count('.') + 1

                    toc_items.append({
                        'number': number_part,
                        'title': title_part,
                        'level': level,
                        'full_text': cleaned_line
                    })
                    matched = True
                    break

            if not matched and cleaned_line:
                # 패턴에 맞지 않는 줄은 레벨 1로 처리
                toc_items.append({
                    'number': '',
                    'title': cleaned_line,
                    'level': 1,
                    'full_text': cleaned_line
                })
        return {'title': document_title, 'toc_items': toc_items}

    def _apply_toc_to_document(self, document, toc_content: str, threshold: float = 0.5):
        parsed_data = self._parse_toc_content(toc_content)
        document_title = parsed_data['title']
        toc_items = parsed_data['toc_items']

        converted_indices = set()
        text_items = [
            (i, item.text.strip())
            for i, item in enumerate(document.texts)
            if isinstance(item, TextItem)
            and item.label == DocItemLabel.TEXT
            and len(item.text.strip()) >= 2
        ]
        text_items_reversed = text_items[::-1]
        matched_count = 0
        section_matched = []

        # 제목 매칭 (앞에서부터)
        if document_title and text_items:
            title_clean = document_title.strip()
            text_only = [text for _, text in text_items]
            close_matches = difflib.get_close_matches(title_clean, text_only, n=3, cutoff=0.3)
            if close_matches:
                best_match_text = close_matches[0]
                best_match_idx = next((idx for idx, text in text_items if text == best_match_text), None)
                if best_match_idx is not None and best_match_idx not in converted_indices:
                    similarity = difflib.SequenceMatcher(None, title_clean.lower(), best_match_text.lower()).ratio()
                    if similarity >= 0.5:
                        original_item = document.texts[best_match_idx]
                        original_item.label = DocItemLabel.TITLE
                        converted_indices.add(best_match_idx)
                        matched_count += 1
                        _log.info(f"문서 제목 설정: {title_clean}")

        # SectionHeader 매칭 (뒤에서부터)
        for toc_item in toc_items:
            toc_full = toc_item['full_text']
            toc_title = toc_item['title']
            target_level = toc_item['level']
            if len(toc_full) < 2:
                continue

            # 1. 후보 텍스트에 대해 유사도 평가 (단, 이미 변환된 인덱스는 제외)
            scored_candidates = []
            for idx, text in text_items_reversed:
                if idx in converted_indices:
                    continue

                sim_full = difflib.SequenceMatcher(None, toc_full.lower(), text.lower()).ratio()
                sim_title = difflib.SequenceMatcher(None, toc_title.lower(), text.lower()).ratio()
                similarity = max(sim_full, sim_title)
                source = "full_text" if sim_full >= sim_title else "title"

                if similarity >= threshold:
                    scored_candidates.append((
                        idx, similarity, text, source, sim_full, sim_title
                    ))

            # 2. 유사도 기준으로 정렬 → top n개 추출
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            top_matches = scored_candidates[:5]

            # 3. 매칭 가능한 가장 첫 번째 후보 선택
            if top_matches:
                best_match_idx, best_similarity, best_match_text, best_match_source, sim_full, sim_title = top_matches[0]
                original_item = document.texts[best_match_idx]
                section_matched.append(best_match_idx)
                section_header = SectionHeaderItem(
                    self_ref=original_item.self_ref,
                    parent=original_item.parent,
                    children=original_item.children,
                    content_layer=original_item.content_layer,
                    prov=original_item.prov,
                    orig=original_item.orig,
                    text=original_item.text,
                    formatting=original_item.formatting,
                    hyperlink=getattr(original_item, 'hyperlink', None),
                    level=target_level
                )
                document.texts[best_match_idx] = section_header
                converted_indices.add(best_match_idx)
                matched_count += 1

        return matched_count

    def _select_best_toc_text_matching(
        self,
        candidate_matches: List[Tuple[int, List[Any]]]
    ) -> List[Dict[str, Any]]:
        """
        TOC 항목과 텍스트 항목 간의 후보 매칭 결과에서
        TOC 순서 및 텍스트 순서를 보존하면서
        총 score 합이 최대인 조합을 선택. DP 알고리즘 사용.

        candidate_matches:
            [
            (toc_idx, [(text_idx, score, text), ...]),
            ...
            ]

        반환:
            [
            {"toc_idx": int, "text_idx": int, "score": float},
            ...
            ]
            - toc_idx 오름차순 & text_idx 오름차순
            - 총 score 합이 최대인 조합
        """

        # 1) (toc_idx, text_idx, score) 로 평탄화
        edges = []
        for toc_idx, text_list in candidate_matches:
            for text_info in text_list:
                edges.append({
                    "toc_idx": toc_idx,
                    "text_idx": text_info[0],
                    "score":  text_info[1],
                })

        if not edges:
            return []

        # 2) TOC 순서, Text 순서 기준으로 정렬
        edges.sort(key=lambda x: (x["toc_idx"], x["text_idx"]))

        n = len(edges)

        # dp[i]  : i번째 edge로 끝나는 최대 점수
        # prev[i]: 그 직전 edge의 인덱스 (경로 복원용)
        dp = [0.0] * n
        prev = [-1] * n

        for i in range(n):
            dp[i] = edges[i]["score"]  # 자기 자신만 선택했을 때
            prev[i] = -1

            for j in range(i):
                # 순서 보존 조건:
                #   toc_j < toc_i AND text_j < text_i
                if (edges[j]["toc_idx"] < edges[i]["toc_idx"] and
                    edges[j]["text_idx"] < edges[i]["text_idx"]):

                    # j를 거쳐서 i로 오는 경우가 더 점수가 크면 갱신
                    if dp[j] + edges[i]["score"] > dp[i]:
                        dp[i] = dp[j] + edges[i]["score"]
                        prev[i] = j

        # 3) 최적 끝점 찾기
        best_end = max(range(n), key=lambda i: dp[i])

        # 4) prev[]를 이용해 경로 역추적
        sequence_indices = []
        cur = best_end
        while cur != -1:
            sequence_indices.append(cur)
            cur = prev[cur]

        sequence_indices.reverse()  # 앞에서부터 순서대로

        # 5) 결과 변환 (정렬은 이미 보장되어 있음)
        result = [
            {
                "toc_idx": edges[i]["toc_idx"],
                "text_idx": edges[i]["text_idx"],
                "score": edges[i]["score"],
            }
            for i in sequence_indices
        ]
        return result

    def _split_bracket_title(self, text: str) -> Optional[Tuple[str, str]]:
        """
        제목에서 맨 앞의 괄호 블록과 나머지 제목을 분리합니다.

        예:
            "[별표 1] 제목"              -> ("[별표 1]", "제목")
            "[별지 제3호 서식] 각서"      -> ("[별지 제3호 서식]", "각서")
            "<별표 3> 평가기준"          -> ("<별표 3>", "평가기준")
            "(별지 제4호 서식) 신청서"    -> ("(별지 제4호 서식)", "신청서")

        매칭 안 되면 None 반환
        """

        m = self.BRACKET_TITLE_PATTERN.match(text)
        if not m:
            return None

        bracket_part = m.group(1).strip()
        title_part = m.group(2).strip()
        return bracket_part, title_part

    def _match_toc_to_document(self, text_items, toc_items: List[Dict[str, Any]], toc_range=None, threshold: float = 0.7):
        """
        목차 항목들을 문서의 텍스트 항목들과 매칭합니다.
        """

        if toc_range is None:
            toc_range = (0, len(toc_items))

        # 후보 텍스트 전처리
        text_items_reversed = text_items[::-1]
        for i, (idx, text) in enumerate(text_items_reversed):
            #  여러 칸 공백을 단일 공백으로 치환하고 소문자 변환
            new_text = re.sub(r" {2,}", " ", text.lower())
            text_items_reversed[i] = (idx, new_text)

        # 1. TOC 항목별로 매칭 시도
        match_results = []
        for i_toc in range(toc_range[0], toc_range[1]):
            toc_item = toc_items[i_toc]
            toc_full = toc_item['full_text']
            toc_title = toc_item['title']
            if len(toc_full) < 2:
                match_results.append((i_toc, []))
                continue

            toc_comp_list = [toc_title.lower()]
            split_result = self._split_bracket_title(toc_title)
            if split_result is not None:
                for part in split_result:
                    if len(part) > 0 and part not in toc_comp_list:
                        toc_comp_list.append(part.lower())

            # 후보 텍스트에 대해 유사도 평가 (단, 이미 변환된 인덱스는 제외)
            scored_candidates = []
            for idx, text in text_items_reversed:

                similarity = 0
                for toc_text in toc_comp_list:
                    sim = difflib.SequenceMatcher(None, toc_text, text[:len(toc_text)]).ratio()
                    similarity = max(similarity, sim)

                if similarity >= threshold:
                    scored_candidates.append((idx, similarity, text))

            # 유사도 기준으로 정렬 → top n개 추출
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            top_matches = scored_candidates[:5]
            match_results.append((i_toc, top_matches))

        # 2. 최적 매칭 선택. toc 순서와 text 순서가 최대한 보존되면서, 유사도가 높은 조합 선택 (다이나믹 프로그래밍 활용)
        best_matches = self._select_best_toc_text_matching(match_results)
        return best_matches

    def _apply_toc_to_law_document(self, document, toc_content: str, threshold: float = 0.7):
        """규정문서의 목차(TOC)를 적용합니다.

        Args:
            document (_type_): TOC가 적용될 문서입니다.
            toc_content (str): TOC의 내용입니다.
            threshold (float, optional): 섹션 헤더 매칭을 위한 유사도 기준입니다. 기본값은 0.5입니다.
        """
        parsed_data = self._parse_toc_content(toc_content)
        document_title = parsed_data['title']
        toc_items = parsed_data['toc_items']

        converted_indices = set()
        text_items = [
            (i, item.text.strip())
            for i, item in enumerate(document.texts)
            if isinstance(item, (TextItem, ListItem))
            and item.label in [DocItemLabel.TEXT, DocItemLabel.LIST_ITEM, DocItemLabel.PAGE_HEADER, DocItemLabel.PARAGRAPH]
            and len(item.text.strip()) >= 2
        ]
        text_items_reversed = text_items[::-1]
        matched_count = 0
        section_matched = []

        # 제목 매칭 (앞에서부터)
        if document_title and text_items:
            title_clean = document_title.strip()
            text_only = [text for _, text in text_items]
            close_matches = difflib.get_close_matches(title_clean, text_only, n=3, cutoff=0.3)
            if close_matches:
                best_match_text = close_matches[0]
                best_match_idx = next((idx for idx, text in text_items if text == best_match_text), None)
                if best_match_idx is not None and best_match_idx not in converted_indices:
                    similarity = difflib.SequenceMatcher(None, title_clean.lower(), best_match_text.lower()).ratio()
                    if similarity >= 0.5:
                        original_item = document.texts[best_match_idx]
                        original_item.label = DocItemLabel.TITLE
                        converted_indices.add(best_match_idx)
                        matched_count += 1
                        _log.info(f"문서 제목 설정: {title_clean}")

        # SectionHeader 매칭
        best_matches = self._match_toc_to_document(
            text_items=text_items,
            toc_items=toc_items,
            threshold=threshold
        )

        _log.debug(f"--- Best Matches ---")
        for match_idx, match in enumerate(best_matches):
            toc_idx = match['toc_idx']
            text_idx = match['text_idx']
            toc_item = toc_items[toc_idx]
            target_level = toc_item['level']

            if text_idx == -1:
                continue

            original_item = document.texts[text_idx]
            section_matched.append(text_idx)
            section_header = SectionHeaderItem(
                self_ref=original_item.self_ref,
                parent=original_item.parent,
                children=original_item.children,
                content_layer=original_item.content_layer,
                prov=original_item.prov,
                # orig=original_item.orig,
                orig=toc_item['title'], # 짧은 제목을 orig에 저장
                text=original_item.text,
                formatting=original_item.formatting,
                hyperlink=getattr(original_item, 'hyperlink', None),
                level=target_level
            )
            document.texts[text_idx] = section_header
            converted_indices.add(text_idx)
            matched_count += 1

            _log.debug(f"[{match_idx}] {toc_item['full_text']}")
            _log.debug(f"  text_idx: {text_idx}, Sim: {match['score']:.4f}, {original_item.text}")

        return matched_count

    def _extract_document_metadata(self, document_content):
        """
        문서 내용에서 메타데이터 정보를 추출하는 함수

        Args:
            document_content (str): 문서 내용

        Returns:
            dict: 추출된 메타데이터 딕셔너리
        """
        try:
            # 사용자 정의 프롬프트 가져오기
            custom_system = self.enrichment_options.metadata_system_prompt
            custom_user = self.enrichment_options.metadata_user_prompt

            # 프롬프트 매니저를 사용하여 AI 모델 호출
            response = self.prompt_manager.call_ai_model(
                category="metadata_extraction",
                prompt_type="korean_financial",
                custom_system=custom_system,
                custom_user=custom_user,
                raise_on_error=True,
                document_content=document_content
            )

            if not response:
                return {"작성일": None, "작성자": []}

            # JSON 찾기
            json_pattern = r'```json\s*([\s\S]*?)\s*```'
            match = re.search(json_pattern, response)

            if match:
                try:
                    metadata = json.loads(match.group(1))
                    return metadata
                except:
                    return {"작성일": None, "작성자": []}
            else:
                try:
                    # JSON 블록이 없는 경우 전체 응답을 JSON으로 파싱 시도
                    return json.loads(response)
                except:
                    return {"작성일": None, "작성자": []}

        except LLMApiError:
            raise
        except Exception as e:
            _log.error(f"메타데이터 추출 중 오류 발생: {str(e)}")
            return {"작성일": None, "작성자": []}


    def _extract_document_metadata_date(self, document_content: str, **kwargs: dict) -> Dict[str, Any]:
        """
        문서 내용에서 메타데이터 정보를 추출하는 함수

        Args:
            document_content (str): 문서 내용

        Returns:
            dict: 추출된 메타데이터 딕셔너리
        """
        try:
            # 사용자 정의 프롬프트 가져오기
            custom_system = self.enrichment_options.metadata_system_prompt
            custom_user = self.enrichment_options.metadata_user_prompt

            # 불필요한 태그 제거 (이미지 정보를 나타내는 태그 제거)
            document_content = document_content.replace("<!-- image -->", "")

            # document_content 에 추가 정보 삽입
            if 'org_filename' in kwargs:
                document_content = f"filename: {kwargs['org_filename']}\n\n{document_content}"

            document_content = document_content.strip()

            if len(document_content) == 0:
                return {"작성일": None, "작성자": []}

            # 프롬프트 매니저를 사용하여 AI 모델 호출
            response = self.prompt_manager.call_ai_model(
                category="metadata_extraction",
                prompt_type="korean_financial_date",
                custom_system=custom_system,
                custom_user=custom_user,
                raise_on_error=True,
                document_content=document_content
            )

            if not response:
                return {"작성일": None, "작성자": []}

            # date 찾기
            match = re.search(r"<date>(.*?)</date>", response)

            if match:
                try:
                    return {"작성일": match.group(1), "작성자": []}
                except:
                    return {"작성일": None, "작성자": []}
            else:
                try:
                    # JSON 블록이 없는 경우 전체 응답을 JSON으로 파싱 시도
                    return json.loads(response)
                except:
                    return {"작성일": None, "작성자": []}

        except LLMApiError:
            raise
        except Exception as e:
            _log.error(f"메타데이터 추출 중 오류 발생: {str(e)}")
            return {"작성일": None, "작성자": []}


# 간단한 함수형 API
def enrich_document(document: DoclingDocument, enrichment_options: DataEnrichmentOptions, **kwargs: dict) -> DoclingDocument:
    """
    DoclingDocument에 enrichment를 적용한 새로운 DoclingDocument를 반환

    Args:
        document: 원본 DoclingDocument 객체
        enrichment_options: 데이터 enrichment 옵션

    Returns:
        DoclingDocument: enrichment가 적용된 새로운 DoclingDocument
    """
    if not document:
        return document

    # 개별 기능이 하나도 활성화되지 않았으면 원본 반환
    if not enrichment_options.do_toc_enrichment and not enrichment_options.extract_metadata:
        return document

    try:
        # DoclingDocument 복사 (일반적으로 pickle 가능)
        enriched_doc = deepcopy(document)
        enricher = DocumentEnrichmentUtils(enrichment_options)

        # enrichment 적용
        toc_count = 0
        metadata_extracted = False

        if enrichment_options.do_toc_enrichment:
            if enrichment_options.toc_doc_type is None or enrichment_options.toc_doc_type == 'normal':
                toc_count = enricher.apply_toc_enrichment(enriched_doc)
            elif enrichment_options.toc_doc_type == 'law':
                toc_count = enricher.apply_law_toc_enrichment(enriched_doc)

        if enrichment_options.extract_metadata:
            metadata_extracted = enricher.apply_metadata_enrichment(enriched_doc, **kwargs)
        _log.info(f"Document enrichment 완료: TOC {toc_count}개, 메타데이터 {metadata_extracted}")

        return enriched_doc

    except LLMApiError:
        raise
    except Exception as e:
        _log.error(f"Document enrichment 중 오류 발생: {str(e)}")
        # 실패 시 원본 반환
        return document


def enrich_document_inplace(document: DoclingDocument, enrichment_options: DataEnrichmentOptions) -> Dict[str, Any]:
    """
    원본 DoclingDocument를 직접 수정하는 방식 (복사 없음)

    Args:
        document: 수정할 DoclingDocument 객체
        enrichment_options: 데이터 enrichment 옵션

    Returns:
        dict: enrichment 결과 정보
    """
    if not document:
        return {'toc_count': 0, 'metadata_extracted': False}

    # 개별 기능이 하나도 활성화되지 않았으면 빈 결과 반환
    if not enrichment_options.do_toc_enrichment and not enrichment_options.extract_metadata:
        return {'toc_count': 0, 'metadata_extracted': False}

    enricher = DocumentEnrichmentUtils(enrichment_options)

    result = {
        'toc_count': 0,
        'metadata_extracted': False
    }

    try:
        # 원본 document 직접 수정
        if enrichment_options.do_toc_enrichment:
            result['toc_count'] = enricher.apply_toc_enrichment(document)

        if enrichment_options.extract_metadata:
            result['metadata_extracted'] = enricher.apply_metadata_enrichment(document)

        _log.info(f"Document enrichment 완료 (in-place): {result}")

    except Exception as e:
        _log.error(f"Document enrichment 중 오류 발생: {str(e)}")

    return result


def add_toc(document: DoclingDocument, enrichment_options: DataEnrichmentOptions) -> DoclingDocument:
    """TOC만 추가한 새로운 DoclingDocument 반환"""
    toc_options = DataEnrichmentOptions(
        do_toc_enrichment=True,
        extract_metadata=False,
        **{k: v for k, v in enrichment_options.model_dump().items() if k.startswith('toc_')}
    )

    return enrich_document(document, toc_options)


def add_metadata(document: DoclingDocument, enrichment_options: DataEnrichmentOptions) -> DoclingDocument:
    """메타데이터만 추가한 새로운 DoclingDocument 반환"""
    metadata_options = DataEnrichmentOptions(
        do_toc_enrichment=False,
        extract_metadata=True,
        **{k: v for k, v in enrichment_options.model_dump().items() if k.startswith('metadata_')}
    )

    return enrich_document(document, metadata_options)


def check_document(document: DoclingDocument, enrichment_options: DataEnrichmentOptions) -> bool:
    """
    문서의 텍스트 품질을 검사하여 OCR이 필요한지 판단하고, 필요시 OCR 적용

    Args:
        document: 원본 DoclingDocument 객체
        enrichment_options: 데이터 enrichment 옵션

    Returns:
        bool: OCR이 필요하지 않으면 True, 필요하면 False
    """
    if not document:
        return False

    try:
        enricher = DocumentEnrichmentUtils(enrichment_options)
        return enricher.check_document_with_good_text(document)

    except Exception as e:
        _log.error(f"문서 품질 검사 중 오류 발생: {str(e)}")
        return False
