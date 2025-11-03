from commons.regex_handler import RegexProcessor
from commons.type_converter import TypeConverter
from parsers.extractor import (
    extract_addenda_id,
    extract_appendix_id,
)
from schemas.law_schema import AdmRuleMetadata, FileAttached
from schemas.schema import ParserContent

type_converter = TypeConverter()
regex_processor = RegexProcessor()


def parse_admrule_info(admrule_id: str, admrule_data: dict, hierarchy_laws, connected_laws) -> ParserContent:
    """
    행정규칙 정보를 파싱하여 ParserContent 객체로 반환하는 함수

    Args:
        admrule_id (str): 행정규칙 ID
        admrule_data (dict): 행정규칙 데이터 딕셔너리
        hierarchy_laws: 상위법 정보
        connected_laws: 연계법 정보

    Returns:
        ParserContent: 메타데이터와 내용이 포함된 ParserContent 객체
    """
    # 행정규칙 기본 정보 추출
    basic_info = _extract_basic_info(admrule_data)
    
    # 부칙 정보 및 시행일자 추출
    addenda, enact_date = _extract_addenda_info(admrule_id, admrule_data)
    
    # 별표(부속서류) 정보 추출
    appendices = _extract_appendix_info(admrule_id, admrule_data)
    
    # 첨부파일 정보 추출
    file_attached = _extract_file_attachments(admrule_data)
    
    # 메타데이터 객체 생성
    metadata = _create_admrule_metadata(
        admrule_id=admrule_id,
        basic_info=basic_info,
        hierarchy_laws=hierarchy_laws,
        connected_laws=connected_laws,
        addenda=addenda,
        appendices=appendices,
        enact_date=enact_date,
        file_attached=file_attached
    )
    
    # ParserContent 객체 반환 (content는 비어 있음)
    return ParserContent(metadata=metadata, content=[])

def _extract_basic_info(admrule_data: dict) -> dict:
    """
    행정규칙 기본 정보를 admrule_data에서 추출하는 함수

    Args:
        admrule_data (dict): 행정규칙 데이터 딕셔너리

    Returns:
        dict: 메타데이터 기본 정보가 담긴 딕셔너리
    """
    # admrule_data에서 '행정규칙기본정보' 키의 값을 가져옴, 없으면 빈 dict
    basic_info = admrule_data.get("행정규칙기본정보", {})
    
    # 각 필드별로 값 추출, 없는 경우 기본값 적용
    return {
        "admrule_num": basic_info.get("행정규칙ID", ""),
        "announce_num": basic_info.get("발령번호", ""),
        "announce_date": basic_info.get("발령일자", ""),
        "enforce_date": basic_info.get("시행일자", ""),
        "rule_name": basic_info.get("행정규칙명", ""),
        "rule_type": basic_info.get("행정규칙종류", ""),
        "article_form": True if basic_info.get("조문형식여부") == "Y" else False,
        "is_effective": 0 if basic_info.get("현행여부") == "Y" else -1,
        "dept": basic_info.get("담당부서기관명", "")
    }

def _extract_addenda_info(admrule_id: str, admrule_data: dict) -> tuple[list, str]:
    """
    부칙 정보를 admrule_data에서 추출하는 함수

    Args:
        admrule_id (str): 행정규칙 ID
        admrule_data (dict): 행정규칙 데이터 딕셔너리

    Returns:
        tuple[list, str]: (부칙 목록, 시행일자)
    """
    addenda = []
    enact_date = "00000000"  # 기본 시행일자

    # admrule_data에 '부칙' 정보가 있을 경우 처리
    if admrule_data.get("부칙"):
        try:
            # 부칙 데이터를 list[dict]로 변환
            addenda_data = type_converter.converter(admrule_data.get("부칙"), list[dict])
            # 부칙 ID 및 시행일자 추출
            addenda, enact_date = extract_addenda_id(admrule_id, addenda_data)
        except Exception:
            # 변환 실패 시 기본값 유지
            pass
    
    return addenda, enact_date

def _extract_appendix_info(admrule_id: str, admrule_data: dict) -> list:
    """
    별표(부속서류) 정보를 admrule_data에서 추출하는 함수

    Args:
        admrule_id (str): 행정규칙 ID
        admrule_data (dict): 행정규칙 데이터 딕셔너리

    Returns:
        list: 별표 정보 목록
    """
    appendices = []
    appendix_data = admrule_data.get("별표")
    
    # '별표' 정보가 있고, 그 안에 '별표단위'가 있을 경우 처리
    if appendix_data and appendix_data.get("별표단위"):
        try:
            # 별표단위 데이터를 list[dict]로 변환
            appendix_units = type_converter.converter(appendix_data.get("별표단위"), list[dict])
            # 별표 ID 추출
            appendices = extract_appendix_id(admrule_id, appendix_units)
        except Exception:
            # 변환 실패 시 기본값 유지
            pass
    
    return appendices

def _extract_file_attachments(admrule_data: dict) -> list[FileAttached]:
    """
    첨부파일 정보를 admrule_data에서 추출하는 함수

    Args:
        admrule_data (dict): 행정규칙 데이터 딕셔너리

    Returns:
        list[FileAttached]: 첨부파일 객체 리스트
    """
    file_attached = []
    attachments = admrule_data.get("첨부파일")
    
    # 첨부파일 정보가 dict 형태로 유효할 경우 처리
    if type_converter.validator(attachments, dict):
        # 첨부파일 링크와 이름을 쌍으로 묶어서 FileAttached 객체 생성
        file_attached = [
            FileAttached(
                id=link.split("flSeq=")[-1],  # 링크에서 파일 ID 추출
                filename=name,
                filelink=link
            )
            for link, name in zip(attachments.get("첨부파일링크", []), attachments.get("첨부파일명", []))
        ]
    
    return file_attached

def _create_admrule_metadata(
    admrule_id: str,
    basic_info: dict,
    hierarchy_laws: list,
    connected_laws: list,
    addenda: list,
    appendices: list,
    enact_date: str,
    file_attached: list
) -> AdmRuleMetadata:
    """
    행정규칙 메타데이터 객체를 생성하는 함수

    Args:
        admrule_id (str): 행정규칙 ID
        basic_info (dict): 기본 정보 딕셔너리
        hierarchy_laws (list): 상위법 목록
        connected_laws (list): 연계법 목록
        addenda (list): 부칙 목록
        appendices (list): 별표 목록
        enact_date (str): 시행일자
        file_attached (list): 첨부파일 목록

    Returns:
        AdmRuleMetadata: 행정규칙 메타데이터 객체
    """
    # AdmRuleMetadata 객체 생성 및 반환
    return AdmRuleMetadata(
        admrule_id=admrule_id,
        admrule_num=basic_info["admrule_num"],
        announce_num=basic_info["announce_num"],
        announce_date=basic_info["announce_date"],
        enforce_date=basic_info["enforce_date"],
        rule_name=basic_info["rule_name"],
        rule_type=basic_info["rule_type"],
        article_form=basic_info["article_form"],
        is_effective=basic_info["is_effective"],
        hierarchy_laws=hierarchy_laws,
        connected_laws=connected_laws,
        related_addenda_admrule=addenda,
        related_appendices_admrule=appendices,
        dept=basic_info["dept"],
        enact_date=enact_date,
        file_attached=file_attached,
    )
