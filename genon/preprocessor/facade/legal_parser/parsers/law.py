from datetime import datetime

from commons.constants import LAWFIELD
from commons.type_converter import TypeConverter
from parsers.extractor import (
    extract_addenda_id,
    extract_appendix_id,
)
from schemas.law_schema import LawMetadata
from schemas.schema import ParserContent

type_converter = TypeConverter()

def _extract_department_info(office: dict) -> str:
    """소관부처 정보를 추출하는 함수"""
    if type_converter.validator(office, dict):
        return f"{office['소관부처명']} {office['부서명']}"
    elif type_converter.validator(office, list[dict]):
        return f"{office[0]['소관부처명']}"
    else :
        return ""

def _extract_law_field(law: dict) -> str:
    """법 분야명을 추출하는 함수"""
    return LAWFIELD.get(int(law.get("편장절관", "00")[:2]))

def _extract_addenda_info(law_id: str, law_data: dict) -> tuple[list[str], str]:
    """부칙 정보를 추출하는 함수"""
    addenda = []
    enact_date = "00000000"
    addenda_data = law_data.get("부칙")
    
    if addenda_data and addenda_data.get("부칙단위"):
        addenda_units = type_converter.converter(addenda_data.get("부칙단위"), list[dict])
        addenda, enact_date = extract_addenda_id(law_id, addenda_units)
    
    return addenda, enact_date

def _extract_appendix_info(law_id: str, law_data: dict) -> list[str]:
    """별표 정보를 추출하는 함수"""
    appendices = []
    appendix_data = law_data.get("별표")
    
    if appendix_data and appendix_data.get("별표단위"):
        appendix_units = type_converter.converter(appendix_data.get("별표단위"), list[dict])
        appendices = extract_appendix_id(law_id, appendix_units)
    
    return appendices

def _extract_is_effective_info(enforce_date:str) -> int:
    "시행일자를 기준으로 현재 시행 예정인지(1) 혹은 현행(0)인지 추출하는 함수"
    today = datetime.now().strftime("%Y%m%d")
    return 1 if enforce_date > today else 0

def _create_law_metadata(
    law_id: str,
    law: dict,
    law_field: str,
    hierarchy_laws: list,
    connected_laws: list,
    addenda: list,
    appendices: list,
    dept: str,
    enact_date: str,
    is_effective: int
) -> LawMetadata:
    return LawMetadata(
        law_id=law_id,
        law_num=law.get("법령ID"),
        announce_num=law.get("공포번호"),
        announce_date=law.get("공포일자"),
        enforce_date=law.get("시행일자"),
        law_name=law.get("법령명_한글"),
        law_short_name=law.get("법령명약칭"),
        law_type=law.get("법종구분", {}).get("content", ""),
        law_field=law_field,
        is_effective=is_effective, 
        hierarchy_laws=hierarchy_laws,
        connected_laws=connected_laws,  
        related_addenda_law=addenda,  
        related_appendices_law=appendices,  
        dept=dept if dept else None,
        enact_date=enact_date,
    )
    

# 법령본문 조회 -> 법령
def parse_law_info(law_id: str, law_data: dict, hierarchy_laws, connected_laws) -> ParserContent:
    law:dict = law_data.get('기본정보')

    # 소관부처 : 소관부처명 + 연락부서 부서명
    office = law.get("연락부서", {}).get("부서단위")
    dept = _extract_department_info(office)

    # 현행 여부
    is_effective = _extract_is_effective_info(law.get("시행일자"))

    ## 법 분야명
    law_field = _extract_law_field(law)
    
    ## 부칙 ID 리스트
    addenda, enact_date = _extract_addenda_info(law_id, law_data)

    ## 별표 ID 리스트
    appendices = _extract_appendix_info(law_id, law_data)

    metadata = _create_law_metadata(
        law_id=law_id,
        law=law,
        law_field=law_field,
        hierarchy_laws=hierarchy_laws,
        connected_laws=connected_laws,
        addenda=addenda,
        appendices=appendices,
        dept=dept,
        enact_date=enact_date,
        is_effective=is_effective
    )

    return ParserContent(metadata=metadata, content=[])