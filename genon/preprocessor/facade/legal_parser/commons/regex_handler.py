import re
from typing import Iterator, Pattern, Union

from pydantic import BaseModel, Field, PrivateAttr


class InvalidRegexError(ValueError):
    """정규표현식이 유효하지 않을 때 발생하는 사용자 정의 예외 클래스"""
    pass


class RegexProcessor(BaseModel):
    """
    법령/행정규칙 등에서 필요한 정규표현식 패턴을 관리하고,
    매칭, 치환 등 다양한 re 메서드를 이름 기반으로 사용할 수 있게 해주는 유틸리티 클래스입니다.
    """

    patterns: dict[str, str] = Field(default_factory=dict)
    _compiled_patterns: dict[str, Union[Pattern, None]] = PrivateAttr(default_factory=dict[str, None])

    class Config:
        arbitrary_types_allowed = True  # Pattern 타입 허용

    def model_post_init(self, context):
        """
        Pydantic 초기화 후 실행되는 후처리 메서드로, 기본 패턴을 등록합니다.
        """
        self.set_patterns()
        return super().model_post_init(context)

    def set_patterns(self):
        """
        주요 패턴들을 등록합니다.
        이 메서드는 RegexProcessor 초기화 시 자동으로 호출됩니다.
        """
        regex_patterns = {
            "CHAPTER_INFO": r"(제(\d+)장)\s*(.*?)(?:<|$)",
            "SECTION_INFO": r"(제(\d+)절)\s*(.*?)(?:<|$)",
            "IS_PREAMBLE": r"(제(\d+)장)\s*(.*?)(?:<|$)|(제(\d+)절)\s*(.*?)(?:<|$)",
            "AMEND_DATE": r"<개정 ([^<>]*)>",
            "BLANKET": r"\(([^)]+)\)",
            "BLANKET_DATE": r"\(.*?개정.*?\)",
            "DATE_KOR": r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일",
            "CHEVRON_DATE": r"<(\d{4})\.(\d{1,2})\.(\d{1,2})>",
            "DATE": r"(\d{4})[.\u00B7]\s*(\d{1,2})[.\u00B7]\s*(\d{1,2})",
            "ARTICLE_NUM": r"제(\d{1,4})조(?:의(\d+))?(?=\s|\(|$)",
            "APPENDIX_REF": r"별표\s*(\d+)(?:\-(\d+))?",
            "APPENDIX_FORM_REF": r"별지\s*제\s*(\d+)호\s*서식(?:\-(\d+))?",
            "ADDENDUM_NUM": r"<?제(\d+(?:-\d+)?)(?:호)?",
            "ADDENDUM_TITLE": r"^\s*부칙(?:\s*\(.*?\))?\s*<[^>]+>",
            "ADDENDUM_ARTICLE": r"^\s*(제\d+조(?:\s*\([^)]*\))?)",
        }
        for key, value in regex_patterns.items():
            self.add_pattern(name=key, pattern=value)

    def get_pattern(self, name: str) -> Union[str, None]:
        """등록된 패턴 문자열을 반환합니다."""
        return self.patterns[name] if self.__contains__(name) else None

    def get_compiled_pattern(self, name: str) -> Union[str, None]:
        """컴파일된 패턴 객체를 반환합니다."""
        return self._compiled_patterns[name] if name in self._compiled_patterns else None

    def add_pattern(self, name: str, pattern: str):
        """
        새로운 패턴을 추가하고 컴파일합니다.
        Raises:
            TypeError: pattern이 문자열이 아닌 경우
            InvalidRegexError: 유효하지 않은 정규식인 경우
        """
        if not isinstance(pattern, str):
            raise TypeError("pattern must be string type")
        self.patterns[name] = pattern
        _ = self.compile(name)

    def remove_pattern(self, name: str) -> bool:
        """
        패턴과 해당 컴파일 객체를 제거합니다.
        Returns:
            bool: 제거 성공 여부
        """
        if self.__contains__(name):
            del self.patterns[name]
            if name in self._compiled_patterns:
                del self._compiled_patterns[name]
            return True
        return False

    def __list_patterns__(self) -> list[str]:
        """등록된 패턴 이름 목록을 반환합니다."""
        return list(self.patterns.keys())

    def __contains__(self, name: str) -> bool:
        """패턴 이름이 등록되어 있는지 확인합니다."""
        return name in self.patterns

    def compile(self, name: str, flags: int = 0) -> Pattern:
        """
        지정한 패턴을 컴파일합니다.
        Returns:
            re.Pattern: 컴파일된 패턴 객체
        Raises:
            InvalidRegexError: 정규식 컴파일 실패 시
        """
        if name not in self._compiled_patterns and self.__contains__(name):
            try:
                _compiled_pattern = re.compile(self.patterns[name], flags)
            except re.error as e:
                del self.patterns[name]
                raise InvalidRegexError(f"유효하지 않은 정규표현식 패턴: {str(e)}")
            else:
                self._compiled_patterns[name] = _compiled_pattern
        return self._compiled_patterns[name]

    def _get_pattern_and_apply(
        self,
        name: str,
        *args,
        method: str,
        flags: int = 0
    ):
        """
        패턴 이름을 기반으로 특정 re 메서드를 실행합니다.
        Args:
            name: 등록된 패턴 이름
            *args: re 메서드 인자
            method: 사용할 re 메서드 이름
            flags: 선택적 re 플래그
        """
        if self.__contains__(name):
            if self._compiled_patterns[name] is None:
                _ = self.compile(name, flags)
            func = getattr(self._compiled_patterns[name], method)
            return func(*args)
        raise ValueError(f"Pattern '{name}' not found.")

    def match(self, name: str, text: str, flags: int = 0) -> Union[re.Match, None]:
        """정규표현식 match 실행"""
        return self._get_pattern_and_apply(name, text, method="match", flags=flags)

    def search(self, name: str, text: str, flags: int = 0) -> Union[re.Match]:
        """정규표현식 search 실행"""
        return self._get_pattern_and_apply(name, text, method="search", flags=flags)

    def findall(self, name: str, text: str, flags: int = 0) -> list:
        """정규표현식 findall 실행"""
        return self._get_pattern_and_apply(name, text, method="findall", flags=flags)

    def finditer(self, name: str, text: str, flags: int = 0) -> Iterator[re.Match]:
        """정규표현식 finditer 실행"""
        return self._get_pattern_and_apply(name, text, method="finditer", flags=flags)

    def sub(self, name: str, repl: str, text: str, flags: int = 0) -> str:
        """정규표현식 sub 실행"""
        return self._get_pattern_and_apply(name, repl, text, method="sub", flags=flags)

    def subn(self, name: str, repl: str, text: str, flags: int = 0) -> tuple[str, int]:
        """정규표현식 subn 실행"""
        return self._get_pattern_and_apply(name, repl, text, method="subn", flags=flags)

    def split(self, name: str, text: str, flags: int = 0) -> list[str]:
        """정규표현식 split 실행"""
        return self._get_pattern_and_apply(name, text, method="split", flags=flags)
