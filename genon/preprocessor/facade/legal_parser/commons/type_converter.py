from typing import Any, Type, TypeVar, Union, get_args, get_origin

from typeguard import TypeCheckError, check_type

T = TypeVar('T')

def type_key(tp):
    """
    주어진 타입 `tp`에 대해 원시 타입과 타입 파라미터를 튜플로 반환합니다.
    예) `list[str]` -> (`list[str]`,`List[str]`)
    """
    origin = get_origin(tp) or tp
    args = get_args(tp)
    return (origin, args)


class TypeConversionError(TypeError):
    """
    타입 변환 오류를 나타내는 커스텀 예외 클래스.
    """
    def __init__(self, src_type: type, dest_type: type):
        self.src_type = src_type
        self.dest_type = dest_type
        super().__init__(f"변환 실패: {src_type.__name__} → {getattr(dest_type, '__name__', str(dest_type))}")


class TypeConverter:
    """
    타입 변환을 처리하는 클래스입니다. 다양한 타입 변환 함수를 등록하여
    객체나 값을 원하는 타입으로 변환할 수 있습니다.
    """

    def __init__(self):
        """
        초기화 메서드. 여러 기본 변환 함수를 등록하고, 기본값들을 설정합니다.
        """
        self._registry = {}
        self.register(int, self._convert_to_int)
        self.register(str, self._convert_to_str)
        self.register(list[dict], self._convert_to_dict_list)
        self.register(list[str], self._convert_to_str_list)
        
        # 기본값 설정
        self._default_values = {
            type_key(int): 0,
            type_key(float): 0.0,
            type_key(str): '',
            type_key(list[dict]): [],
            type_key(list[str]): []
        }

    def register(self, tp, func):
        """
        특정 타입에 대한 변환 함수를 등록합니다.

        Args:
            tp (type): 변환하려는 타입.
            func (function): 해당 타입을 변환하는 함수.
        """
        self._registry[type_key(tp)] = func
        
    def converter(self, value: Any, target_type: Union[type, tuple[type, ...]], use_default: bool = False, use_strip: bool = False) -> Any:
        """
        주어진 값 `value`를 대상 타입 `target_type`으로 변환합니다.
        변환이 실패할 경우 기본값을 사용할지 여부를 선택할 수 있습니다.

        Args:
            value (Any): 변환할 값.
            target_type (Union[type, tuple[type, ...]]): 목표 타입 (단일 타입 또는 여러 타입).
            use_default (bool): 변환 실패 시 기본값을 사용할지 여부.
            use_strip (bool): 문자열 변환 시 공백을 제거할지 여부.
        
        Returns:
            Any: 변환된 값.

        Raises:
            TypeConversionError: 변환에 실패할 경우 발생.
        """
        if self.validator(value, target_type):
            return value
        
        for t in self._expand_types(target_type):
            try:
                converter_fn = self._registry.get(type_key(t))
                if not converter_fn:
                    continue
                # 해당 함수의 파라미터 이름 목록에서 "use_strip" 조회
                return converter_fn(value, use_strip) if "use_strip" in converter_fn.__code__.co_varnames else converter_fn(value)
            except Exception:
                if use_default:
                    return self.get_default_value(target_type)
                raise TypeConversionError(type(value), target_type)

        if use_default:
            return self.get_default_value(target_type)
        raise TypeConversionError(type(value), target_type)
    
    def get_default_value(self, target_type: Type[T]) -> T:
        """
        주어진 타입에 대해 기본값을 반환합니다.

        Args:
            target_type (Type[T]): 기본값을 찾을 대상 타입.

        Returns:
            T: 기본값
        """
        key = type_key(target_type)
        if key in self._default_values:
            return self._default_values[key]
        
        # 컬렉션 타입에 대한 처리
        origin, args = type_key(target_type)
        
        if origin is list:
            return []
        elif origin is dict:
            return {}
        elif origin is set:
            return set()
        elif origin is tuple:
            if args:
                return tuple(self.get_default_value(arg) for arg in args)
            return ()
        if origin is Union:
            for arg in args:
                if arg is not type(None):    # None 타입이 아닌 첫 번째 타입의 기본값 반환
                    return self.get_default_value(arg)
        return None
    
    def validator(self, value: Any, target_type: Union[type, tuple[type, ...]]) -> bool:
        """
        주어진 값이 대상 타입에 맞는지 검증합니다.

        Args:
            value (Any): 검증할 값.
            target_type (Union[type, tuple[type, ...]]): 검증할 대상 타입.
        
        Returns:
            bool: 타입이 맞으면 True, 아니면 False.
        """
        if not value:
            return False
        try:
            check_type(value, target_type)
            return True
        except TypeCheckError:
            return False

    def _expand_types(self, target_type: Union[type, tuple[type, ...]]) -> list[type]:
        """
        주어진 타입이 튜플일 경우, 그 안에 있는 타입들을 확장하여 반환합니다.

        Args:
            target_type (Union[type, tuple[type, ...]]): 대상 타입.
        
        Returns:
            list[type]: 확장된 타입 리스트.
        """
        if isinstance(target_type, tuple):
            return list(target_type)
        return [target_type]
    
    def _convert_to_int(self, value: Any) -> int:
        """
        문자열을 정수로 변환합니다.

        Args:
            value (Any): 변환할 값.
        
        Returns:
            int: 변환된 정수 값.

        Raises:
            TypeConversionError: 변환 실패 시 발생.
        """
        if self.validator(value, str) and value.isdigit():
            return int(value)
        raise TypeConversionError(type(value), int)

    def _convert_to_str(self, value: Any, use_strip: bool = False) -> str:
        """
        문자열 변환 함수. 리스트형 데이터를 문자열로 변환합니다.

        Args:
            value (Any): 변환할 값.
            use_strip (bool): 문자열의 양쪽 공백을 제거할지 여부.
        
        Returns:
            str: 변환된 문자열.

        Raises:
            TypeConversionError: 변환 실패 시 발생.
        """
        if self.validator(value, list[list[str]]):
            if use_strip:
                value = [line.strip() for sublist in value for line in sublist]
            else:
                value = [line for sublist in value for line in sublist]
            return ''.join(value)
        
        elif self.validator(value, list[str]):
            if use_strip:
                value = [line.strip() for line in value]
            return ''.join(value)
        raise TypeConversionError(type(value), str)

    def _convert_to_dict_list(self, value: Any) -> list[dict]:
        """
        리스트 형태로 변환된 dict들을 반환합니다.

        Args:
            value (Any): 변환할 값.
        
        Returns:
            list[dict]: 변환된 dict 리스트.

        Raises:
            TypeConversionError: 변환 실패 시 발생.
        """
        if self.validator(value, dict):
            return [value]
        if self.validator(value, list[list[dict]]):
            return [item for items in value for item in items]
        raise TypeConversionError(type(value), list[dict])

    def _convert_to_str_list(self, value: Any) -> list[str]:
        """
        문자열을 포함하는 리스트로 변환합니다.

        Args:
            value (Any): 변환할 값.
        
        Returns:
            list[str]: 변환된 문자열 리스트.

        Raises:
            TypeConversionError: 변환 실패 시 발생.
        """
        if self.validator(value, str):
            return [value]
        elif self.validator(value, list[list[str]]):
            return value[0]
        raise TypeConversionError(type(value), list[str])
