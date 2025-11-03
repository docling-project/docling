import unicodedata
from datetime import datetime, timedelta
from typing import Union

import pytz

from commons.type_converter import TypeConverter

type_converter = TypeConverter()

def replace_strip(data: list[str]) -> list[str]:
    """
    주어진 문자열 리스트에서 공백을 제거한 후, 빈 문자열을 제외한 새로운 리스트를 반환합니다.

    Args:
        data (list[str]): 공백을 제거하고자 하는 문자열 리스트.
    
    Returns:
        list[str]: 공백을 제거한 후 빈 문자열을 제외한 리스트.
    """
    return [x.strip() for x in data if x.strip()]

def format_date(year: str, month: str, day: str) -> str:
    """
    주어진 연도, 월, 일을 `yyyyMMdd` 형식의 문자열로 포맷합니다.

    Args:
        year (str): 연도.
        month (str): 월.
        day (str): 일.
    
    Returns:
        str: `yyyyMMdd` 형식의 문자열로 변환된 날짜.
    
    Raises:
        ValueError: 연도, 월, 일이 숫자 형식이 아니거나 0 이하일 경우.
    """
    if type_converter.validator((year, month, day), tuple[str, str, str]):
        return f"{year}{int(month):02d}{int(day):02d}"

def normalize_to_nfc(text: str) -> Union[str, None]:
    """
    주어진 텍스트를 NFC(Normalization Form C)로 정규화합니다.

    맥 환경에서 자주 발생하는 한글 자모 분리 문제를 해결하는 데 사용될 수 있습니다.
    예를 들어, "가나다"와 같이 표기된 파일명이 맥에서 "ㄱㅏㄴㅏㄷㅏ"와 같이 분리되어 보이는 문제를 해결할 수 있습니다.

    Args:
        text (str): 정규화할 텍스트.
    
    Returns:
        Union[str, None]: NFC로 정규화된 텍스트. 텍스트가 비어 있으면 `None`을 반환.
    """
    if not text: 
        return None

    return unicodedata.normalize('NFC', text)

def get_kst_yesterday_str() -> str:
    """
    서울 시간(KST)으로 어제 날짜를 `yyyyMMdd` 형식의 문자열로 반환합니다.

    Returns:
        str: 서울 시간(KST)으로 어제 날짜를 `yyyyMMdd` 형식으로 반환.
    """
    kst_now = datetime.now(pytz.timezone('Asia/Seoul'))
    kst_yesterday = kst_now - timedelta(days=1)
    return kst_yesterday.strftime("%Y%m%d")
