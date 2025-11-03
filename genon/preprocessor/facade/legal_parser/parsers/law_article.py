from commons.regex_handler import RegexProcessor
from commons.type_converter import TypeConverter
from commons.utils import format_date
from parsers.extractor import (
    extract_date_to_yyyymmdd,
    extract_related_appendices,
    get_latest_date,
)
from schemas.law_schema import (
    ArticleChapter,
    LawArticleMetadata,
    RuleInfo,
)
from schemas.schema import ParserContent

type_converter = TypeConverter()
regex_processor = RegexProcessor()

# ======================= 개정일자 추출 ===========================
def _extract_latest_announce(data:dict, enact_date: str) -> str:
    """조문 내용, 조문 참고자료, 항 내용, 호 내용에서 가장 최신의 개정 날짜를 추출하여 내용과 함께 반환합니다."""
    amendment_dates = _extract_amendment_dates(data)
    return get_latest_date(amendment_dates, enact_date)

def _extract_amendment_dates(data: dict) -> list[str]:
    """법령 데이터에서 개정일자를 추출하는 함수"""

    dates = []

    dates.extend(_extract_dates_from_article_content(data))
    dates.extend(_extract_dates_from_reference(data))
    dates.extend(_extract_dates_from_paragraphs(data))

def _extract_dates_from_article_content(data: dict) -> list[str]:
    """조문내용에서 개정일을 추출하는 함수"""
    dates = []
    if "조문내용" not in data or not data["조문내용"]:
        return dates
    
    raw_content = data["조문내용"]
    text = type_converter.converter(raw_content, str)
    dates.extend(extract_date_to_yyyymmdd(text))
    return dates


def _extract_dates_from_reference(data: dict) -> list[str]:
    """조문참고자료에서 개정일을 추출하는 함수"""
    dates = []
    reference_data = data.get("조문참고자료")
    
    if not reference_data:
        return dates
    
    # 문자열인 경우
    if isinstance(reference_data, str):
        dates.extend(extract_date_to_yyyymmdd(reference_data))
    
    # 2차원 리스트인 경우
    elif isinstance(reference_data, list):
        items = type_converter.converter(reference_data, list[str])
        for item in items:
            dates.extend(extract_date_to_yyyymmdd(item))
    
    return dates

def _extract_dates_from_paragraphs(data: dict) -> list[str]:
    """항 내용에서 개정일을 추출하는 함수"""
    dates = []
    paragraph = data.get("항")
    if not paragraph:
        return dates
    
    # 항이 리스트인 경우
    if isinstance(paragraph, dict) and "호" in paragraph:
        dates.extend(_extract_dates_from_subparagraphs(paragraph["호"]))
    # 항이 dict이고 호가 있는 경우
    elif type_converter.validator(paragraph, list[dict]):
        for item in paragraph:
            # 항제개정일자문자열이 있는 경우 우선 처리
            if "항제개정일자문자열" in item:
                dates.extend(extract_date_to_yyyymmdd(item["항제개정일자문자열"]))
                return dates  # 명시적인 개정일이 있으면 바로 반환
            
            # 항내용에서 추출
            if "항내용" in item:
                text = _extract_paragraph_content(item)
                dates.extend(extract_date_to_yyyymmdd(text))
    return dates

def _extract_dates_from_subparagraphs(subparagraphs: list[dict]) -> list[str]:
    """호 내용에서 개정일을 추출하는 함수"""
    dates = []
    for item in subparagraphs:
        if "호내용" in item:
            text = _extract_subparagraph_content(item)
            dates.extend(extract_date_to_yyyymmdd(text, True))
    return dates

# ======================= 조문 내용[조, 항, 호, 목] 추출 ===========================
def _stringify_article_content(data: dict) -> list[str]:
    """법령 조문 데이터를 문자열 리스트로 변환하는 함수"""
    content = []
    
    # 조문 내용 추가
    article_content = _extract_article_content(data)
    if article_content:
        content.append(article_content)
    
    # 항 내용 추가
    if data.get("항"):
        paragraphs = type_converter.converter(data.get("항"), list[dict])
        paragraph_contents = _process_paragraphs(paragraphs)
        content.extend(paragraph_contents)

    return content

def _extract_article_content(data: dict) -> str:
    """조문내용을 추출하는 함수"""
    if not data.get("조문내용"):
        return ""
    text = type_converter.converter(data.get("조문내용"), str)
    return text.strip()

def _process_paragraphs(paragraphs: list[dict]) -> list[str]:
    """항 내용을 처리하는 함수"""
    content = []
    
    for paragraph in paragraphs:
        # 항내용 추가
        paragraph_content = _extract_paragraph_content(paragraph)
        if paragraph_content:
            content.append(paragraph_content)
        
        # 호 내용 추가
        if paragraph.get("호"):
            subparagraph = type_converter.converter(paragraph.get("호"), list[dict])
            subparagraph_contents = _process_subparagraphs(subparagraph)
            content.extend(subparagraph_contents)
    
    return content

def _extract_paragraph_content(paragraph: dict) -> str:
    """항내용을 추출하는 함수"""  
    if not paragraph.get("항내용"):
        return ""  
    text:str = type_converter.converter(paragraph.get("항내용"), str)
    return text.strip()

def _process_subparagraphs(subparagraphs: list[dict]) -> list[str]:
    """호 내용을 처리하는 함수"""
    content = []
    
    for subparagraph in subparagraphs:
        # 호내용 추가
        subparagraph_content = _extract_subparagraph_content(subparagraph)
        if subparagraph_content:
            content.append(subparagraph_content)
        
        # 목 내용 추가
        if subparagraph.get("목"):
            items = type_converter.converter(subparagraph.get("목"), list[dict])
            item_contents = _process_items(items)
            content.extend(item_contents)
    
    return content

def _extract_subparagraph_content(subparagraph: dict) -> str:
    """호내용을 추출하는 함수"""
    if not subparagraph["호내용"]:
        return ""
    
    text = type_converter.converter(subparagraph["호내용"], str)
    return text.strip()

def _process_items(items: list[dict]) -> list[str]:
    """목 내용을 처리하는 함수"""
    content = []
    
    for item in items:
        if not item.get("목내용") :
            continue
        
        texts = type_converter.converter(item["목내용"], list[str], use_strip=True, use_default=True)
        content.extend(texts)
    
    return content

# =======================삭제 및 전문 메타데이터 처리===========================
def _extract_deleted_article_date(article: list[str], default_date: str) -> str:
    """삭제된 조문의 날짜를 추출하는 함수"""
    if not article:
        return default_date
    
    # 리스트를 문자열로 변환
    content_str = ' '.join(article) if isinstance(article, list) else article
    
    # 삭제 날짜 추출 (예: <2023. 5. 16.> 형식)
    announce_date_match = regex_processor.search("CHEVRON_DATE", content_str)
    
    if announce_date_match:
        year, month, day = announce_date_match.groups()
        return format_date(year, month, day)
    
    return default_date


def _process_preamble(item: dict, article_chapter: ArticleChapter) -> tuple[str, int, ArticleChapter]:
    """전문을 처리하는 함수"""
    content = item.get("조문내용")
    content = type_converter.converter(content, str)
    
    article_chapter.extract_text(content)
    updated_chapter = ArticleChapter(
        chapter_num=article_chapter.chapter_num,
        chapter_title=article_chapter.chapter_title,
        section_num=article_chapter.section_num,
        section_title=article_chapter.section_title,
    )
    
    article_num = f"{article_chapter.chapter_num}"
    article_sub_num = 0
    
    return article_num, article_sub_num, updated_chapter

# =========================== 법령 조문 파싱 ====================================== 

def process_article_unit(
    item: dict, 
    law_id: str, 
    enact_date: str, 
    is_effective: int, 
    article_chapter: ArticleChapter, 
    current_chapter: ArticleChapter = None
) -> ParserContent:
    """
    하나의 조문을 처리하여 ParserContent 객체를 생성하는 함수

    Args:
        item (dict): 조문 단위 정보
        law_id (str): 법령 ID
        enact_date (str): 법령 제정일자
        is_effective (int): 현행 여부
        article_chapter (ArticleChapter): 조문 챕터 정보
        current_chapter (ArticleChapter, optional): 현재 챕터 정보

    Returns:
        ParserContent: 파싱된 조문 메타데이터와 내용을 담은 객체
    """
    # 기본 정보 추출
    article_num = item.get("조문번호")
    article_sub_num = item.get("조문가지번호") or 1
    article_title = item.get("조문제목", "")  # 삭제의 경우 ""
    enforce_date = item.get("조문시행일자")
    is_preamble = True if item.get("조문여부") == "전문" else False
    
    # 전문(조문의 머리말)인 경우 별도 처리
    if is_preamble:
        article_num, article_sub_num, updated_chapter = _process_preamble(item, article_chapter)
        current_chapter = updated_chapter
    
    # 조문 ID 생성 (법령ID + 조문번호 + 가지번호)
    article_id = f"{law_id}{int(article_num):04d}{int(article_sub_num):03d}"
    
    # 가장 최근의 공포일(개정일) 추출
    announce_date = _extract_latest_announce(item, enact_date)
    # 조문 내용 추출 및 문자열화
    article_content = _stringify_article_content(item)

    # 삭제된 조문 처리
    if "삭제" in article_title or any("삭제" in content for content in article_content if isinstance(content, str)):
        # 삭제된 조문에서 날짜 추출 시도
        deleted_date = _extract_deleted_article_date(article_content, announce_date)
        if deleted_date:
            announce_date = deleted_date
            enforce_date = deleted_date
        
        # 제목이 없는 경우 "삭제"로 설정
        if not article_title:
            article_title = "삭제"
    
    # 조문 내 인용된 별표(appendix) ID 추출
    related_appendices = extract_related_appendices(law_id, article_content)
    
    # 조문 메타데이터 객체 생성
    article_meta = LawArticleMetadata(
        article_id=article_id,
        article_num=article_num,
        article_sub_num=article_sub_num,
        is_preamble=is_preamble,
        article_title=article_title,
        article_chapter=current_chapter or article_chapter, 
        enforce_date=enforce_date,
        announce_date=announce_date,
        law_id=law_id,
        is_effective=is_effective,
        related_appendices=related_appendices,
        related_addenda=[],
    )
    
    # 최종 ParserContent 객체 반환
    return ParserContent(metadata=article_meta, content=article_content)


def parse_law_article_info(law_info: RuleInfo, article_data: dict) -> list[ParserContent]:
    """
    법령 조문 정보를 파싱하여 ParserContent 객체 리스트로 반환하는 함수

    Args:
        law_info (RuleInfo): 법령 기본 정보
        article_data (dict): 조문 데이터

    Returns:
        list[ParserContent]: 파싱된 조문 객체 리스트
    """
    # 조문 데이터가 없거나 "조문단위" 키가 없으면 빈 리스트 반환
    if not article_data:
        return []
    if not article_data.get("조문단위"):
        return []  
    
    # 조문단위 데이터를 리스트[dict]로 변환
    article_units = type_converter.converter(article_data.get("조문단위"), list[dict])
    if not article_units:
        return []
    
    article_list = []
    law_id = law_info.rule_id
    enact_date = law_info.enact_date
    is_effective = law_info.is_effective
    
    # 챕터 정보 객체 생성
    article_chapter = ArticleChapter()
    current_chapter = None
    
    # 각 조문 단위별로 파싱 수행
    for item in article_units:
        article_result = process_article_unit(
            item, 
            law_id, 
            enact_date, 
            is_effective, 
            article_chapter, 
            current_chapter
        )
        
        if article_result:
            article_list.append(article_result)
            # 전문(머리말)인 경우, current_chapter를 업데이트하여 이후 조문에 반영
            if article_result.metadata.is_preamble:
                current_chapter = article_result.metadata.article_chapter
    
    return article_list
