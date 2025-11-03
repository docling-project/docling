from typing import Union

from ..commons.regex_handler import RegexProcessor
from ..commons.utils import format_date

regex_processor = RegexProcessor()

def extract_addenda_id(
    rule_id: int, addenda_list: list[dict]
) -> tuple[list[str], str]:
    """
    법령/행정규칙 메타데이터에 필요한 부칙 ID를 추출하는 함수

    Args:
        rule_id (int): 법령 또는 행정규칙 ID
        addenda_list (list[dict]): 부칙 데이터 리스트

    Returns:
        tuple[list[str], str]: (부칙 ID 리스트, 첫 부칙의 공포일자)
    """
    if not addenda_list:
        # 부칙 데이터가 없으면 빈 리스트와 기본 날짜 반환
        return [], "00000000"
    
    def extractor(addenda_list: list[dict]):
        """
        내부 함수: 부칙 리스트에서 부칙 ID와 첫 시행일자 추출
        """
        res = []
        for item in addenda_list:
            announce_date = item.get("부칙공포일자")
            # 부칙 공포일자가 문자열인 경우(단일 부칙)
            if isinstance(announce_date, str):
                res.append(f"{rule_id}{announce_date}")
            # 부칙 공포일자가 리스트인 경우(복수 부칙)
            elif isinstance(announce_date, list):
                for date in announce_date:
                    res.append(f"{rule_id}{date}")
            enact_date = res[0][-8:]  # 첫 부칙의 날짜(마지막 8자리)
        return res, enact_date
    
    addenda, enact_date = extractor(addenda_list)
    return addenda, enact_date

def extract_appendix_id(rule_id: str, appendix_units: list[dict]) -> list[str]:
    """
    법령/행정규칙 별표 ID를 추출하는 함수

    Args:
        rule_id (str): 법령 또는 행정규칙 ID
        appendix_units (list[dict]): 별표 단위 데이터 리스트

    Returns:
        list[str]: 별표 ID 리스트
    """
    if not appendix_units:
        return []
    
    appendices = []
    for item in appendix_units:
        appendix_key = item.get('별표키', "")
        # rule_id가 10자 이상이면 행정규칙으로 간주하여 타입 부호 추가
        if len(rule_id) > 10:
            type_sign = "E" if item.get('별표구분') == "별표" else "F"
            appendix_key += type_sign
        appendices.append(f'{rule_id}{appendix_key}')
    return appendices

def extract_date_to_yyyymmdd(text: str, date_korean: bool = False) -> list[str]:
    """
    문자열에서 YYYY.MM.DD 또는 YYYY년 MM월 DD일 형식의 날짜를 추출하여 YYYYMMDD로 변환

    Args:
        text (str): 날짜가 포함된 문자열
        date_korean (bool, optional): 한글 날짜 패턴도 사용할지 여부

    Returns:
        list[str]: 변환된 날짜 문자열 리스트(YYYYMMDD)
    """
    # DATE 패턴(숫자형)으로 먼저 추출
    date_list = regex_processor.findall("DATE", text)
    # 없으면 한글 날짜 패턴 사용
    if not date_list and date_korean:
        date_list = regex_processor.findall("DATE_KOR", text)
    # 추출된 날짜를 YYYYMMDD로 변환
    return [format_date(year, month, day) for year, month, day in date_list]

def get_latest_date(dates: list[str], enact_date: str) -> str:
    """
    날짜 리스트에서 가장 최신 날짜를 반환 (없으면 enact_date 반환)

    Args:
        dates (list[str]): 날짜 문자열 리스트
        enact_date (str): 기본 제정일자

    Returns:
        str: 가장 최신 날짜
    """
    return max(dates) if dates else enact_date

def extract_article_num(law_id: str, text: str, lst=False) -> Union[tuple[str, int, int], list[str]]:
    """
    텍스트에서 조문 번호를 추출합니다.
    "(제 xx조의 xx~~)" 또는 "(제 xx조)" 패턴을 찾아 조문 ID 리스트를 반환합니다.
    조문 번호가 없으면 []을 반환합니다.
    현재 행정규칙, 별표 데이터에서 사용

    Args:
        law_id (str): 법령 ID
        text (str): 조문 번호가 포함된 텍스트
        lst (bool): 리스트로 반환할지 여부

    Returns:
        Union[tuple[str, int, int], list[str]]: (조문ID, 본조번호, 가지번호) 또는 조문ID 리스트
    """
    article_ids = []
    main_num = 1
    sub_num = 1

    match = regex_processor.search("ARTICLE_NUM", text)
    if match:
        main_num = int(match.group(1))  # 본조 번호
        sub_num = int(match.group(2)) if match.group(2) else sub_num  # '의' 조문 번호(없으면 1)
    article_id = f"{law_id}{main_num:04d}{sub_num:03d}"
    if not lst:
        return article_id, main_num, sub_num
    else:
        article_ids.append(article_id)
        return article_ids

def extract_related_appendices(law_id: str, article_content: Union[str, list[str]]) -> list[str]:
    """
    조문 내용에서 인용된 별표를 검색하고 별표 ID 리스트를 생성하는 함수

    Args:
        law_id (str): 법령 ID
        article_content (Union[str, list[str]]): 조문 내용(문자열 또는 문자열 리스트)

    Returns:
        list[str]: 인용된 별표 ID 리스트
    """
    # 리스트면 문자열로 합침
    content_text = " ".join(article_content) if isinstance(article_content, list) else article_content
    related_appendices = set()
    
    # 별표 인용 검색 및 ID 생성
    for match in regex_processor.findall("APPENDIX_REF", content_text):
        appendix_num = int(match[0])
        appendix_sub_num = int(match[1]) if match[1] else 0
        appendix_id = f"{law_id}{appendix_num:04d}{appendix_sub_num:02d}E"
        related_appendices.add(appendix_id)

    # 별지 인용 검색 및 ID 생성
    for match in regex_processor.findall("APPENDIX_FORM_REF", content_text):
        appendix_num = int(match[0])
        appendix_sub_num = int(match[1]) if match[1] else 0
        appendix_id = f"{law_id}{appendix_num:04d}{appendix_sub_num:02d}F"
        related_appendices.add(appendix_id)

    return list(related_appendices)
