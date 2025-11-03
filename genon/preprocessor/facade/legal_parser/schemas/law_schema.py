import re
from collections import namedtuple
from typing import Union
from pydantic import BaseModel, Field

# 법령 또는 행정규칙의 기본 정보 (ID, 시행일, 제정일, 현행여부)를 담는 구조
# 함수 파라미터의 타입으로 사용
RuleInfo = namedtuple("RuleInfo", ["rule_id", "enforce_date", "enact_date", "is_effective"])

# 상하위 관계에 있는 법령 정보 구조
HierarchyLaws = namedtuple("HierarchyLaws", ["law_id", "law_num", "law_code", "law_type", "law_name", "parent_id"])

# 참조 관계에 있는 관련 법령 정보 구조
ConnectedLaws = namedtuple("ConnectedLaws", ["law_id", "law_num", "law_code", "law_type", "law_name"])

class ArticleChapter(BaseModel):
    """
    법령 또는 행정규칙 조문 내의 (편)장절(관) 정보를 담는 모델입니다.
    조문 텍스트에서 장 및 절 정보를 추출할 수 있는 기능을 포함합니다.
    """
    chapter_num: int = Field(1, description="조문의 장(chapter) 번호")
    chapter_title: str = Field("", description="조문의 장(chapter) 제목")
    section_num: int = Field(0, description="조문의 절(section) 번호")
    section_title: str = Field("", description="조문의 절(section) 제목")

    def extract_text(self, text: str):
        """
        주어진 텍스트에서 '제X장', '제X절' 형식의 장/절 정보를 추출하여 필드에 저장합니다.
        """
        CHAPTERINFO = r"(제(\d+)장)\s*(.*?)(?:<|$)"
        SECTIONINFO = r"(제(\d+)절)\s*(.*?)(?:<|$)"
        
        chapter_match = re.search(CHAPTERINFO, text)
        if chapter_match:
            self.chapter_num = int(chapter_match.group(2))
            self.chapter_title = chapter_match.group(1) + " " + chapter_match.group(3).strip()
        
        section_match = re.search(SECTIONINFO, text)
        if section_match:
            self.section_num = int(section_match.group(2))
            self.section_title = section_match.group(1) + " " + section_match.group(3).strip()

        return self

class LawMetadata(BaseModel):
    """
    법령 기본 메타데이터를 표현하는 모델입니다.
    """
    law_id: str = Field(..., description="법령 ID")
    law_num: str = Field(..., description="법령 번호 (6자리 문자열)")
    announce_num: str = Field(..., description="공포번호")
    announce_date: str = Field(..., description="공포일자 (yyyymmdd 형식)")
    enforce_date: str = Field(..., description="시행일자 (yyyymmdd 또는 '00000000')")
    law_name: str = Field(..., description="법령명(한글)")
    law_short_name: Union[str, None] = Field(None, description="법령명 약칭")
    law_type: str = Field(..., description="법종구분명")
    law_field: str = Field(..., description="법 분야명")
    is_effective: int = Field(..., title="현행여부", description="0: 현행, 1: 예정, -1:과거(연혁)")
    hierarchy_laws: list[HierarchyLaws] = Field([], description="상하위법 정보")
    connected_laws: list[ConnectedLaws] = Field([], description="관련 법령 정보")
    related_addenda_law: list[str] = Field([], description="부칙 ID 리스트 (최신순)")
    related_appendices_law: list[str] = Field([], description="별표 ID 목록")
    dept: Union[str, None] = Field(None, description="소관부처 정보")
    enact_date: str = Field(..., description="제정일자 (첫 부칙 공포일자)")
    is_law: bool = Field(True, description="법령 여부 (True=법령, False=행정규칙)")

class LawArticleMetadata(BaseModel):
    """
    법령의 조문 메타데이터를 담는 모델입니다.
    """
    article_id: str = Field(..., description="조문 ID (법령ID + 조문번호)")
    article_num: int = Field(..., description="조문번호")
    article_sub_num: int = Field(..., description="조문 가지번호")
    is_preamble: bool = Field(..., description="전문 여부")
    article_chapter: ArticleChapter = Field(default_factory=ArticleChapter, description="장, 절 정보")
    article_title: str = Field(..., description="조문 제목")
    enforce_date: str = Field(..., description="시행일자")
    announce_date: str = Field(..., description="공포일자")
    law_id: str = Field(..., description="법령 ID")
    is_effective: int = Field(0, title="현행여부", description="0: 현행, 1: 예정, -1:과거(연혁)")
    related_appendices: list[str] = Field([], description="관련 별표 ID 리스트")
    related_addenda: list[str] = Field([], description="관련 부칙 ID 리스트")

class AppendixMetadata(BaseModel):
    """
    법령의 별표 메타데이터를 담는 모델입니다.
    """
    appendix_id: str = Field(..., description="별표 ID")
    appendix_num: int = Field(..., description="별표 번호")
    appendix_sub_num: int = Field(..., description="별표 가지번호")
    appendix_seq_num: str = Field(..., description="별표 시퀀스 번호")
    appendix_type: str = Field(..., description="별표 구분 (1: 별표, 2: 서식 등)")
    appendix_title: str = Field(..., description="별표 제목")
    appendix_link: str = Field(..., description="별표 파일 링크")
    announce_date: Union[str, None] = Field(None, description="최신 별표 개정일자")
    enforce_date: str = Field(..., description="별표 시행일자")
    is_effective: int = Field(0, title="현행여부", description="0: 현행, 1: 예정, -1:과거")
    law_id: str = Field(..., description="법령 ID")
    related_articles: list[str] = Field([], description="관련 조문 ID 리스트")
    related_addenda: list[str] = Field([], description="관련 부칙 ID 리스트")

class AddendumMetadata(BaseModel):
    """
    부칙의 메타데이터를 표현하는 모델입니다.
    """
    addendum_id: str = Field(..., description="부칙 ID (법령ID + 부칙 공포일자)")
    addendum_num: str = Field(..., description="부칙 공포번호")
    addendum_title: str = Field(..., description="부칙 제목")
    announce_date: str = Field(..., description="공포일자")
    law_id: str = Field(..., description="법령 ID")
    related_laws: list[str] = Field([], description="관련 법령명 리스트")
    related_articles: list[str] = Field([], description="관련 조문 ID 리스트")
    related_appendices: list[str] = Field([], description="관련 별표 ID 리스트")
    is_exit: bool = Field(False, description="최초 조문 개정일 이전 제정 여부")

# 행정규칙 첨부파일 정보 구조
FileAttached = namedtuple("FileAttached", ["id", "filename", "filelink"])

class AdmRuleMetadata(BaseModel):
    """
    행정규칙의 메타데이터를 표현하는 모델입니다.
    """
    admrule_id: str = Field(..., description="행정규칙 ID")
    admrule_num: str = Field(..., description="행정규칙 번호")
    announce_num: str = Field(..., description="발령번호")
    announce_date: str = Field(..., description="발령일자")
    enforce_date: str = Field(..., description="시행일자")
    rule_name: str = Field(..., description="행정규칙명")
    rule_type: str = Field(..., description="행정규칙 종류 코드")
    article_form: bool = Field(..., description="조문 형식 여부")
    is_effective: int = Field(0, title="현행여부", description="0: 현행, 1: 예정, -1:과거")
    hierarchy_laws: list[HierarchyLaws] = Field([], description="상하위법 정보")
    connected_laws: list[ConnectedLaws] = Field([], description="관련 법령 정보")
    related_addenda_admrule: list[str] = Field([], description="부칙 리스트")
    related_appendices_admrule: list[str] = Field([], description="별표 리스트")
    dept: Union[str, None] = Field(None, description="소관부처")
    enact_date: str = Field(..., description="제정일자")
    file_attached: list[FileAttached] = Field([], description="첨부파일 리스트")
    is_law: bool = Field(False, description="법령 여부 (False = 행정규칙)")

class AdmRuleArticleMetadata(BaseModel):
    """
    행정규칙의 조문 메타데이터를 표현하는 모델입니다.
    """
    article_id: str = Field(..., description="조문 ID (행정규칙ID + 조문번호)")
    article_num: int = Field(..., description="조문번호")
    article_sub_num: int = Field(..., description="조문 가지번호")
    admrule_id: str = Field(..., description="행정규칙 ID")
    article_chapter: ArticleChapter = Field(default_factory=ArticleChapter, description="장, 절 정보")
    article_title: str = Field(..., description="조문 제목")
    is_effective: int = Field(..., title="현행여부", description="0: 현행, 1: 예정, -1:과거")
    enforce_date: str = Field(..., description="시행일자")
    announce_date: str = Field(..., description="공포일자")
    is_preamble: bool = Field(..., description="전문 여부")
    related_appendices: list[str] = Field([], description="관련 별표 리스트")
    related_addenda: list[str] = Field([], description="관련 부칙 리스트")
