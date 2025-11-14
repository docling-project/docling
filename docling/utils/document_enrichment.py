# 0708 매칭 수정된 코드
import json
import logging
import re
import difflib
from typing import Dict, Any, Optional, List
from copy import deepcopy
from difflib import SequenceMatcher

from docling_core.types.doc import (
    DocItemLabel, SectionHeaderItem, TextItem, DoclingDocument
)
from docling_core.types.doc.document import (
    GraphData, GraphCell, GraphCellLabel, ListItem
)
from docling.prompts import PromptManager
from docling.datamodel.pipeline_options import DataEnrichmentOptions

from collections import Counter

_log = logging.getLogger(__name__)


class DocumentEnrichmentUtils:
    """문서 enrichment 유틸리티 클래스 - TOC 추출 및 메타데이터 추출 기능 제공"""

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
            self.enrichment_options.toc_model):

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

            custom_api_configs["toc_extraction"] = toc_config

        # Metadata API 설정
        if (self.enrichment_options.metadata_api_provider or
            self.enrichment_options.metadata_api_key or
            self.enrichment_options.metadata_api_base_url or
            self.enrichment_options.metadata_model):

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
            _log.info("TOC 추출 시작...")

            # 원시 텍스트 추출
            raw_text = self._extract_raw_text_for_toc(document)

            # 사용자 정의 프롬프트 가져오기
            custom_system = self.enrichment_options.toc_system_prompt
            custom_user = self.enrichment_options.toc_user_prompt

            # AI로 목차 생성
            toc_content = self.prompt_manager.call_ai_model(
                category="toc_extraction",
                prompt_type="korean_document",
                custom_system=custom_system,
                custom_user=custom_user,
                raw_text=raw_text
            )

            if toc_content:
                # 모든 SectionHeaderItem을 TextItem으로 변환
                self._convert_section_headers_to_text(document)
                # 목차를 기반으로 SectionHeader 적용
                matched_count = self._apply_toc_to_document(document, toc_content)
                _log.info(f"TOC 추출 완료 - {matched_count}개 섹션 헤더 생성")
                return matched_count
            else:
                _log.warning("TOC 생성 실패")
                return 0

        except Exception as e:
            _log.error(f"TOC 추출 중 오류 발생: {str(e)}")
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
        collected = []

        for txt in window_texts:
            parsed_data = self._parse_toc_content(txt)
            title = parsed_data['title']
            items = parsed_data['toc_items']
            if title and not final_title:
                final_title = title
            collected.extend(items)

        # print("--- Combined TOC Items ---")
        # print(collected)

        if not collected and not final_title:
            return ""

        # 중복 제거(경계 부근의 같은 항목 제거)
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
        # 1) TITLE: 기준 추출
        # -------------------------------------
        title_match = re.search(r"^TITLE:.*", text, re.MULTILINE)
        if title_match:
            return text[title_match.start():].replace("```", "").strip()

        # -------------------------------------
        # 2) 코드블록 ``` ... ``` 추출
        # -------------------------------------
        code_blocks = re.search(r"```(.*?)```", text, re.S)
        if code_blocks:
            return code_blocks.group(1).strip()

        # -------------------------------------
        # 3) Final Table of Contents 이후 추출
        # -------------------------------------
        ftoc_match = re.search(r"Final Table of Contents", text, re.IGNORECASE)
        if ftoc_match:# "Final Table of Contents" 이후 지점
            start_pos = ftoc_match.end()

            # 이후 문자열
            rest = text[start_pos:]

            # 첫 번째 줄바꿈을 찾아 그 다음 줄부터 반환
            newline_idx = rest.find("\n")
            if newline_idx != -1:
                rest = rest[newline_idx+1:]

            return rest.replace("```", "").strip()

        # -------------------------------------
        # 4) 위 조건이 모두 없으면 전체 반환
        # -------------------------------------
        return text.strip()

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
            _log.info("TOC 추출 시작...")

            # 원시 텍스트 추출
            raw_text = self._extract_raw_text_for_toc(document)

            # 사용자 정의 프롬프트 가져오기
            custom_system = self.enrichment_options.toc_system_prompt
            custom_user = self.enrichment_options.toc_user_prompt

            # AI로 목차 생성
            toc_content = self.prompt_manager.call_ai_model(
                category="toc_extraction",
                prompt_type="law_document",
                custom_system=custom_system,
                custom_user=custom_user,
                raw_text=raw_text
            )

            # 20250918, shkim, sliding window 방식으로 여러 조각을 받아서 결합하는 경우. 테스트 중.
            # pieces = self.prompt_manager.call_ai_model_windowed(
            #     category="toc_extraction",
            #     prompt_type="law_document",
            #     custom_system=custom_system,
            #     custom_user=custom_user,
            #     raw_text=raw_text
            # )

            # for i, p in enumerate(pieces):
            #     print(f"--- TOC piece {i} ---\n{p}\n")

            # toc_content = self.combine_windowed_toc(pieces)

            # print(f"--- Combined TOC ---\n{toc_content}\n")

            if toc_content:
                # 모든 SectionHeaderItem을 TextItem으로 변환
                self._convert_section_headers_to_text(document)

                toc_content = self.extract_content(toc_content)
                # print(f"--- only TOC ---\n{toc_content}\n")

                # 목차를 기반으로 SectionHeader 적용
                matched_count = self._apply_toc_to_law_document(document, toc_content)
                _log.info(f"TOC 추출 완료 - {matched_count}개 섹션 헤더 생성")
                return matched_count
            else:
                _log.warning("TOC 생성 실패")
                return 0

        except Exception as e:
            _log.error(f"TOC 추출 중 오류 발생: {str(e)}")
            return 0

    def apply_metadata_enrichment(self, document: DoclingDocument) -> bool:
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
            # 문서의 처음 2페이지에서 텍스트 추출
            temp_content = ""
            total_pages = len(document.pages)
            for page in range(1, min(3, total_pages + 1)):
                temp_content += document.export_to_markdown(page_no=page)

            metadata = self._extract_document_metadata_date(temp_content)
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

    def _apply_toc_to_law_document(self, document, toc_content: str, threshold: float = 0.5):
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
            if (isinstance(item, TextItem) or isinstance(item, ListItem))
            and (item.label == DocItemLabel.TEXT or item.label == DocItemLabel.LIST_ITEM or item.label == DocItemLabel.PAGE_HEADER)
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

                sim_full = difflib.SequenceMatcher(None, toc_full.lower(), text.lower()[:len(toc_full)]).ratio()
                sim_title = difflib.SequenceMatcher(None, toc_title.lower(), text.lower()[:len(toc_title)]).ratio()
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
                    # orig=original_item.orig,
                    orig=toc_title, # 짧은 제목을 orig에 저장
                    text=original_item.text,
                    formatting=original_item.formatting,
                    hyperlink=getattr(original_item, 'hyperlink', None),
                    level=target_level
                )
                document.texts[best_match_idx] = section_header
                converted_indices.add(best_match_idx)
                matched_count += 1

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

        except Exception as e:
            _log.error(f"메타데이터 추출 중 오류 발생: {str(e)}")
            return {"작성일": None, "작성자": []}


    def _extract_document_metadata_date(self, document_content):
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
                prompt_type="korean_financial_date",
                custom_system=custom_system,
                custom_user=custom_user,
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

        except Exception as e:
            _log.error(f"메타데이터 추출 중 오류 발생: {str(e)}")
            return {"작성일": None, "작성자": []}


# 간단한 함수형 API
def enrich_document(document: DoclingDocument, enrichment_options: DataEnrichmentOptions) -> DoclingDocument:
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
            metadata_extracted = enricher.apply_metadata_enrichment(enriched_doc)
        _log.info(f"Document enrichment 완료: TOC {toc_count}개, 메타데이터 {metadata_extracted}")

        return enriched_doc

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
