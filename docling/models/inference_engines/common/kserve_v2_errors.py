"""Typed errors for KServe v2 transport clients.

These exceptions preserve the underlying transport status (status code, details,
debug string) so callers can decide how to react without re-parsing
transport-specific exceptions. The exception *class* is the primary signal a
caller acts on:

- ``KserveV2RequestError``: request was invalid; fail fast, channel is healthy.
- ``KserveV2OverloadError``: server is overloaded / queue deadline exceeded;
  treat as backpressure, do not retry in the client.
- ``KserveV2TransportError``: transport/session failure the channel is expected
  to self-heal (gRPC reconnects subchannels automatically); safe to retry for
  stateless inference.
- ``KserveV2PersistentError``: persistent service/config problem (auth,
  model-not-found, backend errors); fail fast and alert.

``retryable`` is a convenience flag that mirrors the category; transport-class
retries for idempotent calls are otherwise handled declaratively by the gRPC
channel's retry service config, not by a Python retry loop.
"""

from __future__ import annotations


class KserveV2ClientError(RuntimeError):
    """Base error for KServe v2 transport clients.

    Preserves the underlying transport status so callers can classify failures
    without re-parsing transport-specific exceptions.
    """

    def __init__(
        self,
        message: str,
        *,
        model_name: str,
        operation: str,
        status_code: str | None = None,
        details: str | None = None,
        debug_error_string: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.model_name = model_name
        self.operation = operation
        # Symbolic status name (e.g. the gRPC code name "UNAVAILABLE" or an HTTP
        # status); kept transport-agnostic so this module needs no grpc import.
        self.status_code = status_code
        self.details = details
        self.debug_error_string = debug_error_string
        self.retryable = retryable


class KserveV2RequestError(KserveV2ClientError):
    """Request was rejected as invalid; the channel/session is healthy.

    Examples: malformed inference request, bad tensor shape/dtype. Fail fast; do
    not retry.
    """


class KserveV2OverloadError(KserveV2ClientError):
    """Server is overloaded or the request exceeded a queue deadline.

    Examples: Triton queue full (resource exhausted) or queue timeout. Treat as
    backpressure: do not retry in the client; signal upstream callers to slow
    down.
    """


class KserveV2TransportError(KserveV2ClientError):
    """Transport/session failure the channel is expected to self-heal.

    Examples: connection unavailable, HTTP/2 GOAWAY, connection reset. Safe to
    retry for stateless inference; gRPC reconnects subchannels automatically, so
    the channel is not recreated.
    """

    def __init__(self, message: str, **kwargs: object) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(message, **kwargs)  # type: ignore[arg-type]


class KserveV2PersistentError(KserveV2ClientError):
    """Persistent service/configuration problem requiring external change.

    Examples: authentication/authorization failure, model not found, repeated
    backend errors. Fail fast and alert; do not hot-loop retries.
    """
