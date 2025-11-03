import json
import re
import os
from pathlib import Path
from typing import Dict, Optional, Union,List
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from fastapi import Request
from utils import assert_cancelled

from collections import namedtuple



# law_info 또는 admrule_info의 Type. function parameter
RuleInfo = namedtuple("RuleInfo", ["rule_id", "enforce_date", "enact_date", "is_effective"])

# 상하위법령
HierarchyLaws = namedtuple("HierarchyLaws", ["law_id", "law_num", "law_code", "law_type", "law_name", "parent_id"])

# 관련법령
ConnectedLaws = namedtuple("ConnectedLaws", ["law_id", "law_num", "law_code", "law_type", "law_name"])

class ArticleChapter(BaseModel):
    '''
        법령/행정규칙 조문의 (편)장절(관) 정보
    '''
    chapter_num: int = Field(1, description="조문의 장(chapter) 번호")
    chapter_title: str = Field("", description="조문의 장(chapter) 제목")
    section_num: int = Field(0, description="조문의 절(section) 번호")
    section_title: str = Field("", description="조문의 절(section) 제목")

    def extract_text(self, text:str):
        CHAPTERINFO = r"(제(\d+)장)\s*(.*?)(?:<|$)"
        SECTIONINFO = r"(제(\d+)절)\s*(.*?)(?:<|$)"
        chapter_match = re.search(CHAPTERINFO, text)
        if chapter_match:
            self.chapter_num=int(chapter_match.group(2))
            self.chapter_title=chapter_match.group(1) + " " + chapter_match.group(3).strip()

        section_match = re.search(SECTIONINFO, text)
        if section_match:
            self.section_num=int(section_match.group(2))
            self.section_title=section_match.group(1) + " " + section_match.group(3).strip()

        return self

class LawMetadata(BaseModel):
    law_id: str = Field(..., description="법령 ID")
    law_num: str = Field(..., description="법령 번호 (6자리 문자열)")
    announce_num: str = Field(..., description="공포번호")
    announce_date: str = Field(..., description="공포일자 (yyyymmdd 형식)")
    enforce_date: str = Field(..., description="시행일자 (yyyymmdd 또는 '00000000')")
    law_name: str = Field(..., description="법령명(한글)")
    law_short_name: Union[str, None] = Field(None, description="법령명 약칭")
    law_type: str = Field(..., description="법종구분명")
    law_field: str = Field(..., description="법 분야명")
    is_effective: int = Field(
        ..., title="현행여부", description="0: 현행, 1: 예정, -1:과거(연혁)"
    )
    hierarchy_laws: list[HierarchyLaws] = Field(
        [], description="상하위법 (법률, 시행령, 시행규칙, 행정규칙 정보)"
    )
    connected_laws: list[ConnectedLaws] = Field(
        [], description="참조법, 관련 타 법령 정보"
    )
    related_addenda_law: list[str] = Field(
        [], description="부칙 (공포일자 최신순으로 정렬된 부칙 ID 리스트)"
    )
    related_appendices_law: list[str] = Field([], description="별표 ID 목록")
    dept: Union[str, None] = Field(None, description="소관부처 (기관명 + 관련부서명)")
    enact_date: str = Field(..., description="제정일자 (첫 번째 부칙의 공포일자)")
    is_law: bool = Field(
        True, description="법령/행정규칙 구분 : True = 법령, False = 행정규칙"
    )


class LawArticleMetadata(BaseModel):
    article_id: str = Field(..., description="조문 ID (법령ID + 조문번호)")
    article_num: int = Field(..., description="조문번호")
    article_sub_num: int = Field(..., description="조문 가지번호")
    is_preamble: bool = Field(..., description="전문여부 (T: 전문, F: 조문)")
    article_chapter: ArticleChapter = Field(default_factory=ArticleChapter, description="조문의 장, 절 정보")
    article_title: str = Field(..., description="조문 제목")
    enforce_date: str = Field(
        ..., description="시행일자 (yyyymmdd 형식, 시행예정시 '00000000')"
    )
    announce_date: str = Field(
        ..., description="조문 최신 공포(개정)일자 (yyyymmdd 형식)"
    )
    law_id: str = Field(..., description="법령 ID")
    is_effective: int = Field(
        0, title="현행여부", description="0: 현행, 1: 예정, -1:과거(연혁)"
    )
    related_laws: list[str] = Field([], description="관련 법령명 리스트")
    related_appendices: list[str] = Field([], description="관련 별표 ID 리스트")
    related_addenda: list[str] = Field([], description="관련 부칙 ID 리스트")
    related_articles: list[str] = Field([], description="관련 법조항 ID 리스트")


class AppendixMetadata(BaseModel):
    appendix_id: str = Field(..., description="(법령 ID)(별표 번호(4자리) 별표 가지번호(2자리)")
    appendix_num : int = Field(..., description="별표 번호")
    appendix_sub_num: int = Field(..., description="별표 가지번호")
    appendix_seq_num: str = Field(
        ..., description="별표 시퀀스 번호 (고유값))"
    )
    appendix_type: str = Field(
        ..., description="별표 구분 (1: 별표, 2: 서식, 3: 별지, 4: 별도, 5: 부록)"
    )
    appendix_title: str = Field(..., description="별표 제목")
    appendix_link: str = Field(..., description="별표파일링크")
    announce_date: Union[str, None] = Field(
        None,
        description="최신 별표 개정일자 (yyyymmdd 형식, 삭제된 별표는 제목에 존재)",
    )
    enforce_date: str = Field(
        ..., description="별표 시행일자 (yyyymmdd 형식, 시행예정일은 '00000000')"
    )
    is_effective: int = Field(
        0, title="현행여부", description="0: 현행, 1: 예정, -1:과거(연혁)"
    )
    law_id: str = Field(..., description="법령 ID")
    related_articles: list[str] = Field([], description="관련 조문 ID 리스트")
    related_addenda : list[str] = Field([], description="관련 부칙 ID 리스트")



class AddendumMetadata(BaseModel):
    addendum_id: str = Field(..., description="부칙 ID (법령ID + 부칙 공포일자)")
    addendum_num: str = Field(..., description="부칙 공포번호")
    addendum_title: str = Field(..., description="부칙 제목")
    announce_date: str = Field(..., description="부칙 공포일자 (yyyymmdd 형식)")
    law_id: str = Field(..., description="법령 ID")
    related_laws: list[str] = Field([], description="관련 법령명, 주로 타법 개정의 법령명") #
    related_articles: list[str] = Field([], description="관련 조문 ID")
    related_appendices: list[str] = Field([], description="관련 별표 ID")

# 행정규칙 첨부파일
FileAttached = namedtuple("FileAttached", ["id", "filename", "filelink"])

class AdmRuleMetadata(BaseModel):
    admrule_id: str = Field(..., description="행정규칙 ID")
    admrule_num: str = Field(..., description="행정규칙 번호")
    announce_num: str = Field(..., description="행정규칙 발령번호")
    announce_date: str = Field(..., description="발령일자 (yyyymmdd 형식)")
    enforce_date: str = Field(..., description="시행일자")
    rule_name: str = Field(..., description="행정규칙명")
    rule_type: str = Field(..., description="행정규칙 종류 코드")
    article_form: bool = Field(
        ...,
        description="조문형식여부 (True: 조문번호 있음, False: 조문 번호 없을 가능성 있음)",
    )
    is_effective: int = Field(
        0, title="현행여부", description="0: 현행, 1: 예정, -1:과거(연혁)"
    )
    hierarchy_laws: list[HierarchyLaws] = Field(
        [], description="상하위법 (법률, 시행령, 시행규칙, 행정규칙 정보)"
    )
    connected_laws: list[ConnectedLaws] = Field(
        [], description="참조법, 관련 타 법령 정보"
    )
    related_addenda_admrule: list[str] = Field(
        [], description="부칙 (해당 법령의 부칙 키 리스트, 공포일자 최신순으로 정렬)"
    )
    related_appendices_admrule: list[str] = Field([], description="별표 (관련 별표 ID 리스트)")
    dept: Union[str, None] = Field(None, description="소관부처 (기관명 + 관련 부서명)")
    enact_date: str = Field(..., description="제정일자 (첫 번째 부칙의 공포일자)")
    file_attached: list[FileAttached] = Field(
        [], description="행정규칙 첨부파일 (ID, 파일명, 파일링크)"
    )
    is_law: bool = Field(
        False, description="법령/행정규칙 구분 : True = 법령, False = 행정규칙"
    )


class AdmRuleArticleMetadata(BaseModel):
    article_id: str = Field(..., description="조문 ID (행정규칙ID + 조문번호)")
    article_num: int = Field(
        ..., description="조문번호"
    )
    article_sub_num: int = Field(..., description="조문 가지번호")
    admrule_id: str = Field(..., description="행정규칙 ID")
    article_chapter: ArticleChapter = Field(default_factory=ArticleChapter, description="조문의 장, 절 정보")
    article_title: str = Field(..., description="조문 제목")
    is_effective: int = Field(
        ..., title="현행여부", description="0: 현행, 1: 예정, -1:과거(연혁)"
    )
    enforce_date: str = Field(
        ..., description="시행일자 (yyyymmdd 형식, 시행예정시 '00000000')"
    )
    announce_date: str = Field(
        ..., description="조문 최신 공포(개정)일자 (yyyymmdd 형식)"
    )
    is_preamble: bool = Field(..., description="전문여부 (T: 전문, F: 조문)")
    related_laws: list[str] = Field([], description="관련 법령 ID 리스트")
    related_articles: list[str] = Field([], description="관련 조문 ID 리스트")
    related_appendices: list[str] = Field([], description="관련 별표 ID 리스트")
    related_addenda: list[str] = Field([], description="관련 부칙 ID 리스트")


class ParserContent(BaseModel):
    metadata: Union[
        LawMetadata,
        LawArticleMetadata,
        AddendumMetadata,
        AdmRuleMetadata,
        AdmRuleArticleMetadata,
        AppendixMetadata,
    ] = Field(..., description="메타데이터")
    content: list[Union[dict, str]] = Field(
        ..., description="본문 텍스트 (law, appendix의 경우 빈 리스트일 수 있음)"
    )


class ParserResult(BaseModel):
    """
    최종 출력 형태
    """
    law: ParserContent = Field(..., description="법령 또는 행정규칙")
    article: list[ParserContent] = Field(..., description="조문")
    addendum: list[ParserContent] = Field(..., description="부칙")
    appendix: list[ParserContent] = Field(..., description="별표")




class DocumentChunk(BaseModel):
    # 공통 필드
    law_name: Optional[str] = None
    law_id: Optional[str] = None
    admrule_id: Optional[str] = None
    type: str  # "법령", "행정규칙", "조문", "부칙", "별표"
    text: str
    hierarchy_path: Optional[str] = None
    #parent: Optional[Dict[str, str]] = None
    reg_date: Optional[str] = None

    # 조문 관련
    article_id: Optional[str] = None
    article_num: Optional[int] = None
    article_sub_num: Optional[int] = None
    article_title: Optional[str] = None
    chapter_number: Optional[int] = None
    chapter_title: Optional[str] = None
    section_number: Optional[int] = None
    section_title: Optional[str] = None
    #is_preamble: Optional[bool] = None
    i_chunk_on_article: Optional[int] = None
    n_chunk_of_article: Optional[int] = None

    # 부칙 관련
    addendum_id: Optional[str] = None
    addendum_num: Optional[str] = None
    addendum_title: Optional[str] = None
    i_chunk_on_addendum: Optional[int] = None
    n_chunk_of_addendum: Optional[int] = None

    # 별표 관련
    appendix_id: Optional[str] = None
    appendix_num: Optional[int] = None
    appendix_sub_num: Optional[int] = None
    appendix_seq_num: Optional[str] = None
    appendix_title: Optional[str] = None
    file_link: Optional[str] = None
    appendix_type: Optional[str] = None

    # 일반 메타정보
    law_num: Optional[str] = None
    admrule_num: Optional[str] = None
    rule_name: Optional[str] = None
    law_name: Optional[str] = None
    law_short_name: Optional[str] = None
    law_type: Optional[str] = None
    law_field: Optional[str] = None
    rule_type: Optional[str] = None
    article_form: Optional[bool] = None
    is_effective: Optional[str] = None  # "현행", "예정", "연혁"
    announce_date: Optional[str] = None
    enforce_date: Optional[str] = None
    enact_date: Optional[str] = None
    dept: Optional[str] = None
    file_attached: Optional[List[List[str]]] = None # 행정규칙 첨부파일 (ID, 파일명, 파일링크)

    # 관계 필드
    related_laws: Optional[List[str]] = None
    related_articles: Optional[List[str]] = None
    related_addenda: Optional[List[str]] = None
    related_appendices: Optional[List[str]] = None
    related_addenda_law: Optional[List[str]] = None
    related_appendices_law: Optional[List[str]] = None
    related_addenda_admrule: Optional[List[str]] = None
    related_appendices_admrule: Optional[List[str]] = None
    hierarchy_laws: Optional[List[Dict[str, Optional[str]]]] = None
    connected_laws: Optional[List[Dict[str, Optional[str]]]] = None

class DocumentProcessor:
    def __init__(self):
        self.parser_result = None

    def load_json_as_model(self, json_file: str, model_class: type[BaseModel]) -> BaseModel:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return model_class.model_validate(data)

    def get_effective_status(self, is_effective: int) -> str:
        return {0: "현행", 1: "예정", -1: "연혁"}.get(is_effective, str(is_effective))


    def extract_parent_from_text(self,text: str) -> dict:
        parent = {}
        circled_number_map = {'①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5',
                            '⑥': '6', '⑦': '7', '⑧': '8', '⑨': '9', '⑩': '10'}
        lines = text.splitlines()
        LEVELS = ["조문", "항", "호", "목", "세목", "세세목"]
        for line in lines:
            detected = {}

            # 조문
            if m := re.match(r"^\s*제\d+조(의\d+)?", line):
                detected["조문"] = m.group()

            # 항
            if m := re.search(r"([①-⑩])", line):
                detected["항"] = circled_number_map[m.group(1)]

            # 호
            if m := re.match(r"^\s*(\d+)\.\s", line):
                detected["호"] = m.group(1)

            # 목
            if m := re.match(r"^\s*([가-힣])\.\s", line):
                detected["목"] = m.group(1)

            # 세목
            if m := re.match(r"^\s*(\d+)\)\s", line):
                detected["세목"] = m.group(1)

            # 세세목
            if m := re.match(r"^\s*([가-힣])\)\s", line):
                detected["세세목"] = m.group(1)

            # 계층 레벨 추적 및 상태 업데이트
            for level in LEVELS:
                if level in detected:
                    # 상위 level 갱신 + 하위 level 초기화
                    idx = LEVELS.index(level)
                    parent[level] = detected[level]
                    for lower in LEVELS[idx+1:]:
                        parent.pop(lower, None)
                    break  # 한 줄에서 하나만 반영

        return {k: v for k, v in parent.items() if k != "조문"}


    def update_hierarchy_memory(self, prev: Dict[str, str], new_detected: Dict[str, str]) -> Dict[str, str]:
        levels = ["항", "호", "목", "세목", "세세목"]
        if "reset" in new_detected:
            return {}

        updated = prev.copy()
        for level in levels:
            if level in new_detected:
                updated[level] = new_detected[level]
                # 하위 항목 제거
                for lower in levels[levels.index(level) + 1:]:
                    updated.pop(lower, None)
        return updated

    def generate_hierarchy_path(self, chapter_title, section_num, section_title, article_num, article_title, parent):
        parts = []
        if chapter_title:
            parts.append(f"{chapter_title}".strip())
        if section_num:
            parts.append(f"{section_num} {section_title}".strip())
        if article_title:
            parts.append(f"제{article_num}조 {article_title}".strip())
        for level in ["항", "호", "목", "세목", "세세목"]:
            if parent and level in parent:
                parts.append(f"{level} {parent[level]}")
        return "/".join(parts)


    def create_hierarchical_text(self, name: str, chapter_title: str, section_title: str, article_title: str, content: str, parent: Dict = None) -> str:
        parts = [p for p in [name, chapter_title, section_title, article_title] if p]
        if parent:
            for level in ["항", "호", "목", "세목", "세세목"]:
                if level in parent:
                    parts.append(f"{level} {parent[level]}")
        return " > ".join(parts) + "\n" + content

    def split_addendum_by_article(self, content_lines: List[str]) -> List[Dict[str, Union[str, List[str]]]]:
        articles = []
        current = None
        for line in content_lines:
            if re.match(r"제\d+조", line):  # 조문 시작
                if current:
                    articles.append(current)
                current = {"title": line, "body": []}
            elif current:
                current["body"].append(line)
        if current:
            articles.append(current)
        return articles

    def process_law_document(self, input_path: Path, output_dir: Path):
        match = re.search(r"Document-(\d+)", input_path.name)
        if not match:
            raise ValueError("Invalid file name format. Expected 'Docunebt-<id>_...'")
        # id_number = match.group(1) # output 파일명에 쓰려고 뽑음

        self.parser_result :ParserResult= self.load_json_as_model(str(input_path), ParserResult)
        law_metadata = self.parser_result.law.metadata
        documents = []

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len, separators=["\n\n", "\n", " ", ""])

        common = DocumentChunk.model_fields.keys()
        base_fields = {key: None for key in common}

        # 메타데이터 공통 처리
        base = base_fields.copy()
        if isinstance(law_metadata, LawMetadata): # lawmetadata 클래스의 인스턴스면 법령이라고 판단
            base.update({
                "law_name": law_metadata.law_name,
                "law_id": law_metadata.law_id,
                "law_num": law_metadata.law_num,
                "announce_num": law_metadata.announce_num,
                "announce_date": law_metadata.announce_date,
                "enforce_date": law_metadata.enforce_date,
                "law_short_name": law_metadata.law_short_name,
                "law_type": law_metadata.law_type,
                "law_field": law_metadata.law_field,
                "type": "법령",
                "is_effective": self.get_effective_status(law_metadata.is_effective),
                "hierarchy_laws": [h._asdict() for h in law_metadata.hierarchy_laws],
                "connected_laws": [h._asdict() for h in law_metadata.connected_laws],
                "related_appendices_law": law_metadata.related_appendices_law,
                "related_addenda_law": law_metadata.related_addenda_law,
                "dept": law_metadata.dept,
                "enact_date": law_metadata.enact_date,
                "text":law_metadata.law_name,
                "reg_date": datetime.now().isoformat(timespec='seconds') + 'Z'
            })
        elif isinstance(law_metadata, AdmRuleMetadata):# AdmRuleMetadata 클래스의 인스턴스면 행정규칙이라고 판단
            base.update({
                "admrule_id": law_metadata.admrule_id,
                "admrule_num": law_metadata.admrule_num,
                "announce_num": law_metadata.announce_num,
                "announce_date": law_metadata.announce_date,
                "enforce_date": law_metadata.enforce_date,
                "rule_name": law_metadata.rule_name,
                "rule_type": law_metadata.rule_type,
                "article_form": law_metadata.article_form,
                "type": "행정규칙",
                "is_effective": self.get_effective_status(law_metadata.is_effective),
                "hierarchy_laws": [h._asdict() for h in law_metadata.hierarchy_laws],
                "connected_laws": [h._asdict() for h in law_metadata.connected_laws],
                "related_appendices_admrule": law_metadata.related_appendices_admrule,
                "related_addenda_admrule": law_metadata.related_addenda_admrule,
                "dept": law_metadata.dept,
                "enact_date": law_metadata.enact_date,
                "file_attached": law_metadata.file_attached,
                "text": law_metadata.rule_name,
            })
        documents.append(DocumentChunk(**base).model_dump())



        name = getattr(law_metadata, "law_name", None) or getattr(law_metadata, "rule_name", None)

        for article in self.parser_result.article:
            meta = article.metadata
            content = "\n".join(article.content)
            is_law = isinstance(meta, LawArticleMetadata)
            split_texts = splitter.split_text(content)


            # 바뀐코드(parent 뒤로 미루기기)
            parent_prev = {}
            parent_curr = {}

            for i, chunk_text in enumerate(split_texts):
                # 현재 청크의 parent는 이전 값 사용
                parent = parent_prev.copy()

                chunk_data = base.copy()
                chunk_data.update({
                    "article_title": meta.article_title,
                    "type": "조문",
                    "law_id": meta.law_id if is_law else None,
                    "admrule_id": meta.admrule_id if not is_law else None,
                    "article_id": meta.article_id,
                    "article_num": meta.article_num,
                    "article_sub_num": meta.article_sub_num,
                    "chapter_number": meta.article_chapter.chapter_num,
                    "chapter_title": meta.article_chapter.chapter_title,
                    "section_number": meta.article_chapter.section_num,
                    "section_title": meta.article_chapter.section_title,
                    "text": self.create_hierarchical_text(
                        name,
                        meta.article_chapter.chapter_title,
                        meta.article_chapter.section_title,
                        # meta.article_title,
                        f"제{meta.article_num}조 {meta.article_title}",
                        chunk_text,
                        parent
                    ),
                    "hierarchy_path": self.generate_hierarchy_path(
                        # meta.article_chapter.chapter_num,
                        meta.article_chapter.chapter_title,
                        meta.article_chapter.section_num,
                        meta.article_chapter.section_title,
                        meta.article_num,
                        meta.article_title,
                        parent
                    ),
                    "parent": parent,
                    "i_chunk_on_article": i,
                    "n_chunk_of_article": len(split_texts),
                    "related_laws": meta.related_laws,
                    "related_addenda": meta.related_addenda,
                    "related_appendices": meta.related_appendices,
                    "related_articles": getattr(meta, "related_articles", None),
                    "announce_date": meta.announce_date,
                    "enforce_date": meta.enforce_date,
                    "is_effective": self.get_effective_status(meta.is_effective),
                })
                documents.append(DocumentChunk(**chunk_data).model_dump())

                # 다음 청크를 위한 parent_curr 추출 및 업데이트
                detected = self.extract_parent_from_text(chunk_text)
                parent_curr = self.update_hierarchy_memory(parent_prev, detected)
                parent_prev = parent_curr.copy()


        # 일반 청크 코드
        # for addendum in self.parser_result.addendum:
        #     meta = addendum.metadata
        #     content = "\n".join(addendum.content)
        #     split_texts = splitter.split_text(content)
        #     for i, chunk in enumerate(split_texts):
        #         d = base.copy()
        #         d.update({
        #             "type": "부칙",
        #             "text": self.create_hierarchical_text(name, None, None, "부칙", chunk),
        #             "hierarchy_path": f"부칙/{meta.addendum_title}",
        #             "addendum_id": meta.addendum_id,
        #             "addendum_num": meta.addendum_num,
        #             "addendum_title": meta.addendum_title,
        #             "law_id": meta.law_id,
        #             "announce_date": meta.announce_date,
        #             "related_laws": meta.related_laws,
        #             "related_articles": meta.related_articles,
        #             "i_chunk_on_addendum": i,
        #             "n_chunk_of_addendum": len(split_texts)
        #         })
        #         documents.append(DocumentChunk(**d).model_dump())

        # tab 처리된 부칙 처리가 가능한 코드
        for addendum in self.parser_result.addendum:
            meta = addendum.metadata
            content_lines = addendum.content
            article_blocks = self.split_addendum_by_article(content_lines)  # 조문으로 분리

            # 이 부칙에서 생성될 모든 청크를 미리 저장할 리스트
            addendum_chunks = []
            global_chunk_idx = 0  # 부칙 내 청크 인덱스

            for article in article_blocks:
                raw_text = "\n".join([article["title"]] + article["body"]) # title은 path로 넣으면 됨
                split_chunks = splitter.split_text(raw_text)

                parent_prev = {}
                for chunk in split_chunks:
                    parent = parent_prev.copy()

                    d = base.copy()
                    d.update({
                        "type": "부칙",
                        "text": self.create_hierarchical_text(
                            name,
                            chapter_title=None,
                            section_title=None,
                            article_title=meta.addendum_title, # 부칙 첫 줄
                            content=chunk,
                            parent=parent
                        ),
                        "hierarchy_path": f"{meta.addendum_title}/{article['title'].split('(')[0].strip()}",
                        "parent": parent,
                        "addendum_id": meta.addendum_id,
                        "addendum_num": meta.addendum_num,
                        "addendum_title": meta.addendum_title,
                        "law_id": meta.law_id,
                        "announce_date": meta.announce_date,
                        "related_laws": meta.related_laws,
                        "related_articles": meta.related_articles,
                        "related_appendices": meta.related_appendices,
                        "i_chunk_on_addendum": global_chunk_idx,
                        # n_chunk_of_addendum은 아래에서 일괄 할당
                    })
                    addendum_chunks.append(DocumentChunk(**d).model_dump())
                    global_chunk_idx += 1

                    detected = self.extract_parent_from_text(chunk)
                    parent_prev = self.update_hierarchy_memory(parent_prev, detected)

            for d in addendum_chunks:
                d["n_chunk_of_addendum"] = global_chunk_idx

            documents.extend(addendum_chunks)


        for appendix in self.parser_result.appendix:
            meta = appendix.metadata
            d = base.copy()
            d.update({
                "type": "별표",
                "text": self.create_hierarchical_text(name, None, None, f"별표 {meta.appendix_num} {meta.appendix_title}", ""),
                #"hierarchy_path": f"별표/{meta.appendix_id} {meta.appendix_title}",
                "appendix_id": meta.appendix_id,
                "appendix_num": meta.appendix_num,
                "appendix_sub_num": meta.appendix_sub_num,
                "appendix_seq_num": meta.appendix_seq_num,
                "appendix_title": meta.appendix_title,
                "appendix_type": meta.appendix_type,
                "law_id": meta.law_id,
                "announce_date": meta.announce_date,
                "enforce_date": meta.enforce_date,
                "is_effective": self.get_effective_status(meta.is_effective),
                "file_link": meta.appendix_link,
                "related_articles": meta.related_articles,
                "related_addenda": meta.related_addenda,
            })
            documents.append(DocumentChunk(**d).model_dump())
        def sort_fields_with_none_last(obj: dict, field_order: list[str]) -> dict:
            return dict(sorted(obj.items(), key=lambda item: (
                item[1] is None or item[1] == [],
                field_order.index(item[0]) if item[0] in field_order else float('inf')
            )))

        # field_order = list(DocumentChunk.model_fields.keys())

        # output_dir.mkdir(parents=True, exist_ok=True)
        # output_file = output_dir / f"{id_number}_now_chunk.json"
        # with open(output_file, "w", encoding="utf-8") as f:
        #     sorted_documents = [sort_fields_with_none_last(d, field_order) for d in documents]
        #     json.dump(sorted_documents, f, ensure_ascii=False, indent=2)


        return documents

    async def __call__(self, request: Request, file_path: str, **kwargs: dict):
        output_path, output_file = os.path.split(file_path)
        vectors = self.process_law_document(Path(file_path), output_path)
        await assert_cancelled(request)

        return vectors
