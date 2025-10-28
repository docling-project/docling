from typing import Union

from pydantic import BaseModel, Field

from schemas.law_schema import (
    AddendumMetadata,
    AdmRuleArticleMetadata,
    AdmRuleMetadata,
    AppendixMetadata,
    LawArticleMetadata,
    LawMetadata,
)
from schemas.vdb_schema import VDBResponse


class ParserRequest(BaseModel):
    """
    파싱 요청 데이터를 나타내는 스키마입니다.

    속성:
        law_ids (list[str]): 파싱할 법령 ID 목록.
        admrule_ids (list[str]): 파싱할 행정규칙 ID 목록.
    """
    law_ids: list[str] = []
    admrule_ids: list[str] = []


class ParserContent(BaseModel):
    """
    파싱된 콘텐츠 데이터를 나타내는 스키마입니다.

    속성:
        metadata (Union[LawMetadata, LawArticleMetadata, AddendumMetadata, AdmRuleMetadata, AdmRuleArticleMetadata, AppendixMetadata]):
            콘텐츠에 대한 메타데이터.
        content (list[Union[dict, str]]): 본문 텍스트 (법령 또는 별표의 경우 빈 리스트일 수 있음).
    """
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
    파싱 결과 데이터를 나타내는 스키마입니다.

    속성:
        law (ParserContent): 법령 또는 행정규칙 데이터.
        article (list[ParserContent]): 조문 데이터.
        addendum (list[ParserContent]): 부칙 데이터.
        appendix (list[ParserContent]): 별표 데이터.
    """
    law: ParserContent = Field(..., description="법령 또는 행정규칙")
    article: list[ParserContent] = Field(..., description="조문", default_factory=list)
    addendum: list[ParserContent] = Field(..., description="부칙", default_factory=list)
    appendix: list[ParserContent] = Field(..., description="별표", default_factory=list)


class ParserResponse(BaseModel):
    """
    파싱 작업의 응답 데이터를 나타내는 스키마입니다.

    속성:
        total_count (int): 총 처리된 ID 수.
        seen_count (int): 처리된 ID 수.
        unseen_count (int): 처리되지 않은 ID 수.
        success_count (int): 성공적으로 처리된 ID 수.
        fail_count (int): 실패한 ID 수.
        seen_ids (dict[str, set[str]]): 처리된 ID 목록 (법령 및 행정규칙).
        unseen_ids (dict[str, set[str]]): 처리되지 않은 ID 목록 (법령 및 행정규칙).
        fail_ids (set[str]): 실패한 ID 목록.
    """
    total_count: int = 0
    seen_count: int = 0
    unseen_count: int = 0
    success_count: int = 0
    fail_count: int = 0 
    seen_ids: dict[str, set[str]] = {}        # {"law": [...], "admrule": [...]}
    unseen_ids: dict[str, set[str]] = {}     # {"law": [...], "admrule": [...]}
    fail_ids: set[str] = set()     

    def model_post_init(self, __context):
        """
        초기화 후 처리되지 않은 ID 수를 설정합니다.
        """
        self.unseen_count = self.total_count

    def increment_total(self):
        """총 처리된 ID 수를 증가시킵니다."""
        self.total_count += 1
        
    def increment_success(self):
        """성공적으로 처리된 ID 수를 증가시킵니다."""
        self.success_count += 1

    def increment_fail(self, id: str):
        """
        실패한 ID 수를 증가시키고, 실패한 ID를 추가합니다.

        Args:
            id (str): 실패한 ID.
        """
        self.fail_count += 1
        self.fail_ids.add(id)


class PipelineResponse(BaseModel):
    """
    파이프라인 작업의 응답 데이터를 나타내는 스키마입니다.

    속성:
        parser (ParserResponse): 파싱 작업 결과.
        vdb (VDBResponse): VDB 벡터화 작업 결과.
    """
    parser: ParserResponse
    vdb: VDBResponse = VDBResponse()