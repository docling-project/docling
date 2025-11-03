import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Optional

import pytz

from commons.constants import DIR_PATH_ERROR_LOG


class NewlineTracebackFormatter(logging.Formatter):
    """
    예외 정보가 줄바꿈을 포함해 보기 좋게 출력되도록 하는 커스텀 로그 포맷터 클래스입니다.
    """

    def formatException(self, exc_info):
        # 기본 예외 출력에 줄바꿈 추가
        result = super().formatException(exc_info)
        return result + "\n"
    
    def format(self, record):
        # 기본 포맷 호출 (예외 존재 여부만 확인)
        result = super().format(record)
        if record.exc_info:
            pass
        return result


class ErrorLogger:
    """
    오류 로그 전용 싱글턴 로거 클래스입니다.
    로그는 파일 및 콘솔에 동시에 기록되며, VDB/법령 관련 예외를 구분해서 출력합니다.
    """

    _instance: Optional["ErrorLogger"] = None

    def __init__(self):
        if ErrorLogger._instance is not None:
            raise RuntimeWarning("Use ErrorLogger.instance() instead of direct instantiation.")

        os.makedirs(DIR_PATH_ERROR_LOG, exist_ok=True)

        # 로그 파일명에 날짜 추가
        seoul_tz = pytz.timezone('Asia/Seoul')
        created_at = datetime.now(seoul_tz).strftime('%Y%m%d')
        log_file = f"{DIR_PATH_ERROR_LOG}/error_log_{created_at}.txt"

        # 로거 설정
        self.logger = logging.getLogger('error_logger')
        self.logger.setLevel(logging.ERROR)

        # 파일 핸들러 설정
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.ERROR)

        # 포맷터 적용
        formatter = NewlineTracebackFormatter('%(message)s')
        self.file_handler.setFormatter(formatter)

        # 콘솔 핸들러 설정
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.ERROR)
        self.console_handler.setFormatter(formatter)

        # 중복 핸들러 방지
        if not self.logger.handlers:
            self.logger.addHandler(self.file_handler)
            self.logger.addHandler(self.console_handler)

        # 상위 로거로 로그 전파 방지
        self.logger.propagate = False

    @classmethod
    def instance(cls) -> "ErrorLogger":
        """
        ErrorLogger 싱글턴 인스턴스를 반환합니다.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def vdb_error(self, text: str, e: Exception) -> None:
        """
        VDB 관련 에러 메시지를 출력합니다.

        Args:
            text (str): 에러 발생 설명
            e (Exception): 예외 객체
        """
        error_summary = self.get_error_summary(e)
        self.logger.error(f"Exception occurred : {text} {error_summary}", exc_info=True)

    def law_error(self, id: str, e: Exception) -> None:
        """
        법령 처리 중 발생한 예외를 로그로 남깁니다.

        Args:
            id (str): 법령 ID
            e (Exception): 예외 객체
        """
        error_summary = self.get_error_summary(e)
        self.logger.error(f"Exception occurred on {id}: {error_summary}", exc_info=True)

    def get_error_summary(self, e: Exception) -> str:
        """
        예외 정보를 요약 문자열로 반환합니다.

        Args:
            e (Exception): 예외 객체

        Returns:
            str: 예외 타입, 위치, 메시지를 포함한 요약
        """
        exc_type, _, exc_tb = sys.exc_info()
        tb = traceback.extract_tb(exc_tb)

        seoul_tz = pytz.timezone('Asia/Seoul')
        timestamp = datetime.now(seoul_tz).strftime('%Y-%m-%d %H:%M')

        if tb:
            last = tb[-1]
            filename = last.filename
            lineno = last.lineno
            funcname = last.name
            err_type = exc_type.__name__ if exc_type else type(e).__name__
            err_msg = str(e)
            return f"[{timestamp}] {err_type} at {filename}:{lineno} in {funcname}() - {err_msg}"
        else:
            return f"[{timestamp}] {type(e).__name__} - {str(e)}"


class MainLogger:
    """
    일반 정보 출력용 메인 로거 클래스입니다.
    싱글턴으로 사용되며, 콘솔에만 출력됩니다.
    """

    _instance: Optional["MainLogger"] = None

    def __init__(self, level=logging.INFO):
        if MainLogger._instance is not None:
            raise RuntimeWarning("Use MainLogger.instance() instead of direct instantiation.")

        self.logger = logging.getLogger('main_logger')
        self.logger.setLevel(level)

        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # 간단한 메시지 포맷
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(console_handler)

        self.logger.propagate = False

    @classmethod
    def instance(cls) -> "MainLogger":
        """
        MainLogger 싱글턴 인스턴스를 반환합니다.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def debug(self, message: str) -> None:
        """디버그 레벨 로그 출력"""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """정보 레벨 로그 출력"""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """경고 레벨 로그 출력"""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """에러 레벨 로그 출력"""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """치명적 에러 레벨 로그 출력"""
        self.logger.critical(message)
