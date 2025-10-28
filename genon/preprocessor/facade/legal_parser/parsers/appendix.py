from commons.regex_handler import RegexProcessor
from commons.type_converter import TypeConverter
from commons.utils import replace_strip
from parsers.extractor import (
    extract_article_num,
    extract_date_to_yyyymmdd,
    get_latest_date,
)
from schemas.law_schema import AppendixMetadata, RuleInfo
from schemas.schema import ParserContent

type_converter = TypeConverter()
regex_processor = RegexProcessor()

def parse_appendix_info(rule_info: RuleInfo, appendix_data: dict, is_admrule: bool = True) -> list[ParserContent]:
    """
    법령 또는 행정규칙의 별표 정보를 파싱하여 구조화된 콘텐츠 리스트를 반환합니다.

    Args:
        rule_info (RuleInfo): 법령 또는 행정규칙에 대한 기본 정보
        appendix_data (dict): 별표(raw JSON 형식) 데이터
        is_admrule (bool, optional): 행정규칙 여부를 나타내는 플래그. 기본값은 True

    Returns:
        list[ParserContent]: 파싱된 별표 콘텐츠와 메타데이터가 포함된 ParserContent 객체의 리스트
    """
    # 별표 데이터가 없으면 빈 리스트 반환
    if not appendix_data:
        return []
    
    # 별표단위 데이터를 리스트[dict]로 변환
    appendix_units = type_converter.converter(appendix_data.get("별표단위"), list[dict])
    
    # 행정규칙 여부에 따라 각각의 파서 함수 호출
    if is_admrule:
        return parse_administrative_rule_appendix(rule_info, appendix_units)
    else:
        return parse_law_appendix(rule_info, appendix_units)

def parse_administrative_rule_appendix(rule_info: RuleInfo, appendix_units: list[dict]) -> list[ParserContent]:
    """
    행정규칙의 별표 정보를 파싱합니다.

    Args:
        rule_info (RuleInfo): 행정규칙에 대한 기본 정보
        appendix_units (list[dict]): 별표 단위 목록

    Returns:
        list[ParserContent]: 파싱된 행정규칙 별표 콘텐츠 리스트
    """
    appendix_list = []
    
    for item in appendix_units:
        # 행정규칙 별표 메타데이터 추출
        metadata = _extract_admrule_appendix_metadata(item, rule_info)
        
        # 별표 내용 추출 및 전처리
        content = _extract_appendix_content(item)
        
        # 별표 내용에서 날짜 정보 추출 및 처리
        date_info = _process_admrule_date(content, rule_info)
        
        # 관련 조문 추출 (행정규칙은 content[0]에서 추출)
        articles = extract_article_num(rule_info.rule_id, content[0] if content else "", lst=True)
        
        # 최종 메타데이터 및 결과 객체 생성
        appendix_metadata = _create_appendix_metadata(metadata, date_info, articles, rule_info)
        appendix_result = ParserContent(content=content, metadata=appendix_metadata)
        
        appendix_list.append(appendix_result)
    
    return appendix_list

def parse_law_appendix(rule_info: RuleInfo, appendix_units: list[dict]) -> list[ParserContent]:
    """
    법령의 별표 정보를 파싱합니다.

    Args:
        rule_info (RuleInfo): 법령에 대한 기본 정보
        appendix_units (list[dict]): 별표 단위 목록

    Returns:
        list[ParserContent]: 파싱된 법령 별표 콘텐츠 리스트
    """
    appendix_list = []
    
    for item in appendix_units:
        # 법령 별표 메타데이터 추출
        metadata = _extract_law_appendix_metadata(item, rule_info)
        
        # 별표 내용 추출 및 전처리
        content = _extract_appendix_content(item)
        
        # 법령 별표의 날짜 정보 처리 (법령은 rule_info의 날짜 정보 사용)
        date_info = {
            "announce_date": rule_info.enact_date,
            "enforce_date": rule_info.enforce_date
        }
        
        # 관련 조문 추출 (법령은 별표 제목에서 추출)
        articles = extract_article_num(rule_info.rule_id, metadata["appendix_title"], lst=True)
        
        # 최종 메타데이터 및 결과 객체 생성
        appendix_metadata = _create_appendix_metadata(metadata, date_info, articles, rule_info)
        appendix_result = ParserContent(content=content, metadata=appendix_metadata)
        
        appendix_list.append(appendix_result)
    
    return appendix_list

def _extract_admrule_appendix_metadata(item: dict, rule_info: RuleInfo) -> dict:
    """
    행정규칙 별표 항목에서 메타데이터를 추출합니다.

    Args:
        item (dict): 별표 단위 정보
        rule_info (RuleInfo): 행정규칙 기본 정보

    Returns:
        dict: 별표 메타데이터 딕셔너리
    """
    rule_id = rule_info.rule_id
    appendix_type = item.get("별표구분", "")
    
    # 행정규칙은 별표 구분에 따라 E(별표) 또는 F(서식) 타입 지정
    type_sign = "E" if appendix_type == "별표" else "F"
    appendix_id = f"{rule_id}{type_sign}"
    
    appendix_num = int(item.get("별표번호"))
    appendix_sub_num = int(item.get("별표가지번호", 0))
    
    # 파일 링크에서 flSeq 값을 추출하여 appendix_seq_num으로 사용
    file_link = item.get("별표서식파일링크")
    appendix_seq_num = file_link.split("flSeq=")[-1]
    appendix_link = f"https://www.law.go.kr{file_link}"
    appendix_title = item.get("별표제목", "")
    
    return {
        "appendix_id": appendix_id,
        "appendix_num": appendix_num,
        "appendix_sub_num": appendix_sub_num,
        "appendix_seq_num": appendix_seq_num,
        "appendix_type": appendix_type,
        "appendix_title": appendix_title,
        "appendix_link": appendix_link
    }

def _extract_law_appendix_metadata(item: dict, rule_info: RuleInfo) -> dict:
    """
    법령 별표 항목에서 메타데이터를 추출합니다.

    Args:
        item (dict): 별표 단위 정보
        rule_info (RuleInfo): 법령 기본 정보

    Returns:
        dict: 별표 메타데이터 딕셔너리
    """
    rule_id = rule_info.rule_id
    appendix_type = item.get("별표구분", "")
    appendix_key = item.get("별표키")
    
    # 법령은 별표키를 appendix_id에 사용
    appendix_id = f"{rule_id}{appendix_key}"
    
    appendix_num = int(item.get("별표번호"))
    appendix_sub_num = int(item.get("별표가지번호", 0))
    
    # 파일 링크에서 flSeq 값을 추출하여 appendix_seq_num으로 사용
    file_link = item.get("별표서식파일링크")
    appendix_seq_num = file_link.split("flSeq=")[-1]
    appendix_link = f"https://www.law.go.kr{file_link}"
    appendix_title = item.get("별표제목", "")
    
    return {
        "appendix_id": appendix_id,
        "appendix_num": appendix_num,
        "appendix_sub_num": appendix_sub_num,
        "appendix_seq_num": appendix_seq_num,
        "appendix_type": appendix_type,
        "appendix_title": appendix_title,
        "appendix_link": appendix_link
    }

def _extract_appendix_content(item: dict) -> list[str]:
    """
    별표 항목에서 내용을 추출하고 정제합니다.

    Args:
        item (dict): 별표 단위 정보

    Returns:
        list[str]: 정제된 별표 내용 리스트
    """
    # 별표내용을 리스트[str]로 변환 (기본값 사용)
    appendix_content = type_converter.converter(item.get("별표내용"), list[str], use_default=True)
    # 공백 및 특수문자 정제
    return replace_strip(appendix_content)

def _process_admrule_date(content: list[str], rule_info: RuleInfo) -> dict:
    """
    행정규칙 별표 내용에서 날짜 정보를 추출하고 처리합니다.

    Args:
        content (list[str]): 별표 내용 리스트
        rule_info (RuleInfo): 행정규칙 기본 정보

    Returns:
        dict: announce_date, enforce_date 정보를 담은 딕셔너리
    """
    enact_date = rule_info.enact_date
    matched_date = None
    
    # 별표 내용이 있을 경우 첫 번째 항목에서 날짜 추출 시도
    if content:
        matched_date = _extract_date_from_admrule(content[0])
    
    if matched_date:
        # 추출된 날짜를 yyyymmdd 형식으로 변환
        format_dates = extract_date_to_yyyymmdd(matched_date)
        # 가장 최근 날짜를 announce_date로 사용
        announce_date = get_latest_date(format_dates, enact_date)
        enforce_date = announce_date  # 시행일 = 개정일 또는 법령 시행일
    else:
        # 날짜 정보가 없으면 rule_info의 날짜 사용
        announce_date = enact_date
        enforce_date = rule_info.enforce_date
    
    return {
        "announce_date": announce_date,
        "enforce_date": enforce_date
    }

def _extract_date_from_admrule(content: str) -> str:
    """
    행정규칙 내용에서 날짜 정보를 추출합니다.

    Args:
        content (str): 별표 내용 문자열

    Returns:
        str: 추출된 날짜 문자열 (없으면 None)
    """
    # AMEND_DATE, BLANKET_DATE 패턴을 가져와서 새로운 패턴으로 추가
    AMEND_DATE = regex_processor.get_pattern("AMEND_DATE")
    BLANKET_DATE = regex_processor.get_pattern("BLANKET_DATE")
    regex_processor.add_pattern("ADR_APPENDIX_DATE", rf"{AMEND_DATE}|{BLANKET_DATE}")
    
    # ADR_APPENDIX_DATE 패턴으로 먼저 검색
    amend_date_matches = regex_processor.search("ADR_APPENDIX_DATE", content)
    if amend_date_matches:
        return amend_date_matches.group(1) or amend_date_matches.group(0)
    
    # 없으면 AMEND_DATE 패턴으로 재검색
    amend_date_matches = regex_processor.search("AMEND_DATE", content)
    return amend_date_matches.group(0) if amend_date_matches else None

def _create_appendix_metadata(metadata: dict, date_info: dict, articles: list, rule_info: RuleInfo) -> AppendixMetadata:
    """
    최종 AppendixMetadata 객체를 생성합니다.

    Args:
        metadata (dict): 별표 메타데이터 딕셔너리
        date_info (dict): 날짜 정보 딕셔너리
        articles (list): 관련 조문 리스트
        rule_info (RuleInfo): 법령/행정규칙 기본 정보

    Returns:
        AppendixMetadata: 별표 메타데이터 객체
    """
    return AppendixMetadata(
        appendix_id=metadata["appendix_id"],
        appendix_num=metadata["appendix_num"],
        appendix_sub_num=metadata["appendix_sub_num"],
        appendix_seq_num=metadata["appendix_seq_num"],
        appendix_type=metadata["appendix_type"],
        appendix_title=metadata["appendix_title"],
        appendix_link=metadata["appendix_link"],
        announce_date=date_info["announce_date"],
        enforce_date=date_info["enforce_date"],
        is_effective=rule_info.is_effective,
        law_id=rule_info.rule_id,
        related_articles=articles
    )
