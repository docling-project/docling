from commons.regex_handler import RegexProcessor
from commons.type_converter import TypeConverter
from commons.utils import format_date
from parsers.extractor import extract_related_appendices
from schemas.law_schema import AddendumMetadata
from schemas.schema import ParserContent

# 초기화
regex_processor = RegexProcessor()
type_converter = TypeConverter()

# ========================== 법령 부칙 처리 ============================================

def parse_law_addendum_info(law_id: str, addendum_data: dict) -> list[ParserContent]:
    """
    법령 부칙을 파싱하여 구조화된 콘텐츠로 반환합니다.

    주어진 법령 ID와 부칙 데이터를 사용하여 부칙 제목, 번호, 공포일자 및 본문을 추출하고,
    이를 구조화된 `ParserContent` 객체 리스트로 반환합니다.

    Args:
        law_id (str): 법령의 고유 식별자.
        addendum_data (dict): 부칙 데이터를 포함하는 딕셔너리.

    Returns:
        list[ParserContent]: 파싱된 법령 부칙의 콘텐츠 리스트.
    """
    # addendum_data가 dict 타입이 아니면 빈 리스트 반환
    if not type_converter.validator(addendum_data, dict):
        return []
    
    addendum_list = []
    # 부칙단위가 없으면 빈 리스트 반환
    addendum_units = addendum_data.get("부칙단위")
    
    if not addendum_units:
        return []
    
    # 부칙단위 리스트를 타입 변환
    addendum_units = type_converter.converter(addendum_units, list[dict], use_default=True)
    
    # 부칙 항목 처리
    for item in addendum_units:
        title, number, announce_date = _extract_law_addendum_info(item)
        content = type_converter.converter(item.get("부칙내용", []), list[str])
        addendum_content = _split_law_addendum_content(content)
        
        # 메타데이터 생성
        metadata = _create_addendum_metadata(
            law_id, title, number, announce_date, addendum_content
        )
        
        # ParserContent 객체 생성
        addendum_result = ParserContent(
            metadata=metadata, 
            content=addendum_content
        )
        addendum_list.append(addendum_result)
    
    return addendum_list

def _extract_law_addendum_info(item: dict) -> tuple[str, str, str]:
    """
    법령 부칙에서 제목, 번호, 공포일자를 추출하는 함수.
    
    부칙 항목에서 제목, 번호, 공포일자를 추출하여 반환합니다.
    
    Args:
        item (dict): 부칙 항목 데이터.
    
    Returns:
        tuple[str, str, str]: 부칙 제목, 번호, 공포일자.
    """
    title = number = announce_date = ""
    
    # item이 dict 타입이 아니면 기본값 반환
    if not type_converter.validator(item, dict):
        return title, number, announce_date
    
    # 부칙내용에서 제목 추출
    if item.get("부칙내용") and isinstance(item.get("부칙내용"), list) and item.get("부칙내용")[0]:
        title = item.get("부칙내용")[0][0].lstrip() if isinstance(item.get("부칙내용")[0], list) else ""
    
    # 부칙 번호와 공포일자 추출
    number = item.get("부칙공포번호", "")
    announce_date = item.get("부칙공포일자", "")
    
    return title, number, announce_date

def _split_law_addendum_content(text_data: list[str]) -> list[str]:
    """
    법령 부칙 본문을 조문 단위로 나누는 함수.
    
    주어진 본문을 조문 단위로 나누고, 제목, 조문, 본문을 분리하여 리스트로 반환합니다.
    
    Args:
        text_data (list[str]): 부칙 본문 텍스트 리스트.
    
    Returns:
        list[str]: 조문 단위로 나누어진 부칙 본문 리스트.
    """
    if not text_data or not isinstance(text_data, list):
        return []

    contents = []
    buffer = ""
    
    for line in text_data:
        raw_line = line.rstrip()
        line = line.strip()
        
        # 부칙 제목 줄 처리
        if regex_processor.match("ADDENDUM_TITLE", raw_line):
            if buffer:
                contents.append(buffer.strip())
                buffer = ""
            contents.append(line)
            continue
        
        # 들여쓰기 처리
        if raw_line.startswith("  "):
            buffer += " " + line
            continue
        
        # 조문 시작 (제1조, 제2조 등)
        if regex_processor.match("ADDENDUM_ARTICLE", line):
            if buffer:
                contents.append(buffer.strip())
            buffer = line
            continue
        else:
            buffer += " " + line if buffer else line
    
    # 마지막에 남은 buffer 처리
    if buffer:
        contents.append(buffer.strip())
    
    return [c for c in contents if c.strip()]

# ========================== 행정규칙 부칙 처리 ============================================

def parse_admrule_addendum_info(law_id: str, addendum_data: dict) -> list[ParserContent]:
    """
    행정규칙 부칙을 파싱하여 구조화된 콘텐츠로 반환합니다.

    주어진 법령 ID와 행정규칙 부칙 데이터를 사용하여 부칙 제목, 번호, 공포일자 및 본문을 추출하고,
    이를 구조화된 `ParserContent` 객체 리스트로 반환합니다.

    Args:
        law_id (str): 법령의 고유 식별자.
        addendum_data (dict): 행정규칙 부칙 데이터를 포함하는 딕셔너리.
    
    Returns:
        list[ParserContent]: 파싱된 행정규칙 부칙의 콘텐츠 리스트.
    """
    if not type_converter.validator(addendum_data, dict):
        return []
    
    addendum_list = []
    addendum_units = addendum_data.get("부칙내용")
    
    if not addendum_units:
        return []
    
    addendum_units = type_converter.converter(addendum_units, list[str], use_default=True)
    
    for item in addendum_units:
        title, number, announce_date = _extract_admrule_addendum_info(item)
        addendum_content = _split_admrule_addendum_content(title, item)
        
        metadata = _create_addendum_metadata(
            law_id, title, number, announce_date, addendum_content
        )
        
        addendum_result = ParserContent(
            metadata=metadata, 
            content=addendum_content
        )
        addendum_list.append(addendum_result)
    
    return addendum_list

def _extract_admrule_addendum_info(item: str) -> tuple[str, str, str]:
    """
    행정규칙 부칙에서 제목, 번호, 공포일자를 추출하는 함수.
    
    행정규칙 부칙 항목에서 제목, 번호, 공포일자를 추출하여 반환합니다.
    
    Args:
        item (str): 행정규칙 부칙 항목 데이터 (문자열).
    
    Returns:
        tuple[str, str, str]: 부칙 제목, 번호, 공포일자.
    """
    title = number = announce_date = ""
    
    if not type_converter.validator(item, str):
        return title, number, announce_date
    
    item = item.strip()
    title_match = regex_processor.search("ADDENDUM_TITLE", item)
    title = title_match.group(0) if title_match else ""
    
    number_match = regex_processor.search("ADDENDUM_NUM", item)
    number = number_match.group(1) if number_match else ""
    
    date_match = regex_processor.search("DATE", item)
    if date_match:
        year, month, day = date_match.groups()
        announce_date = format_date(year, month, day)
    
    return title, number, announce_date

def _split_admrule_addendum_content(title: str, text: str) -> list[str]:
    """
    행정규칙 부칙 본문을 조문 단위로 나누는 함수.

    주어진 본문을 조문 단위로 나누고, 제목과 함께 반환합니다.
    
    Args:
        title (str): 부칙 제목.
        text (str): 부칙 본문 텍스트.
    
    Returns:
        list[str]: 조문 단위로 나누어진 부칙 본문 리스트.
    """
    contents = []
    
    if not type_converter.validator(text, str):
        return contents
    
    line = text.strip()
    
    if title and title in line:
        contents.append(title)
        line = regex_processor.sub("ADDENDUM_TITLE", "", line).strip()
    
    parts = regex_processor.split("ADDENDUM_ARTICLE", line)
    if len(parts) == 1:
        if parts[0]:
            contents.append(parts[0].strip())
    else:
        for i in range(1, len(parts), 2):
            article = parts[i]
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            contents.append(f"{article} {content}")
    
    return [c for c in contents if c.strip()]

def _create_addendum_metadata(
    law_id: str,
    title: str,
    number: str,
    announce_date: str,
    content: list[str]
) -> AddendumMetadata:
    """
    부칙 메타데이터를 생성하는 함수.

    주어진 법령 ID, 부칙 제목, 번호, 공포일자 및 본문을 사용하여 부칙 메타데이터를 생성하고 반환합니다.
    
    Args:
        law_id (str): 법령 ID.
        title (str): 부칙 제목.
        number (str): 부칙 번호.
        announce_date (str): 부칙 공포일자.
        content (list[str]): 부칙 본문 내용.
    
    Returns:
        AddendumMetadata: 부칙 메타데이터 객체.
    """
    related_laws = [match.group(1) for match in regex_processor.finditer("BLANKET", title)]
    related_appendices = extract_related_appendices(law_id, content)
    
    return AddendumMetadata(
        addendum_id=f"{law_id}{announce_date}",
        addendum_num=number,
        addendum_title=title,
        announce_date=announce_date,
        law_id=law_id,
        related_laws=related_laws,
        related_articles=[],
        related_appendices=related_appendices,
        is_exit=False
    )

def parse_addendum_info(law_id: str, addendum_data: dict, is_admrule: bool = True) -> list[ParserContent]:
    """
    메인 함수 - 타입에 따라 적절한 파서 호출

    주어진 법령 ID와 부칙 데이터를 사용하여 법령 부칙 또는 행정규칙 부칙을 파싱하고,
    파싱된 내용을 구조화된 `ParserContent` 객체 리스트로 반환합니다. 
    `is_admrule` 값에 따라 법령 부칙 또는 행정규칙 부칙을 파싱하는 다른 함수를 호출합니다.

    Args:
        law_id (str): 법령의 고유 식별자.
        addendum_data (dict): 부칙 데이터를 포함하는 딕셔너리 (법령 부칙 또는 행정규칙 부칙).
        is_admrule (bool, optional): `True`이면 행정규칙 부칙을 파싱하고, `False`이면 법령 부칙을 파싱합니다. 기본값은 `True`.

    Returns:
        list[ParserContent]: 파싱된 법령 부칙 또는 행정규칙 부칙의 콘텐츠 리스트.
    """
    if is_admrule:
        return parse_admrule_addendum_info(law_id, addendum_data)
    else:
        return parse_law_addendum_info(law_id, addendum_data)
