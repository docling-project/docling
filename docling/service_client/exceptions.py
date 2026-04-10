"""Exceptions for the docling-serve client SDK."""

from __future__ import annotations

from dataclasses import dataclass


class DoclingServiceClientError(Exception):
    """Base error for all client SDK failures."""


@dataclass(slots=True)
class ServiceError(DoclingServiceClientError):
    """Raised for non-retryable HTTP service errors."""

    message: str
    status_code: int | None = None
    detail: str | None = None

    def __str__(self) -> str:
        if self.status_code is None:
            return self.message
        if self.detail:
            return f"{self.message} (status={self.status_code}, detail={self.detail})"
        return f"{self.message} (status={self.status_code})"


class ServiceUnavailableError(ServiceError):
    """Raised for unavailable service or exhausted HTTP 500 retries."""


class TaskTimeoutError(DoclingServiceClientError):
    """Raised when waiting for task completion exceeds timeout."""


class TaskNotFoundError(DoclingServiceClientError):
    """Raised when a task id is unknown to the service."""


class ResultNotReadyError(DoclingServiceClientError):
    """Raised when a result is requested before task reaches terminal state."""


class ResultExpiredError(DoclingServiceClientError):
    """Raised when a terminal task no longer has a stored result."""


class ConversionError(DoclingServiceClientError):
    """Raised when a single conversion completes with failure."""


@dataclass(slots=True)
class BatchConversionError(DoclingServiceClientError):
    """Raised when one or more sources fail in convert_all()."""

    message: str
    failures: list[Exception]

    def __str__(self) -> str:
        return f"{self.message} ({len(self.failures)} failure(s))"
