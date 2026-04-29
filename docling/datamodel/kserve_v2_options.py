"""Common KServe v2 API configuration options mixin."""

import warnings
from typing import Any, Dict, Literal, Optional

from pydantic import AliasChoices, BaseModel, Field, model_validator


class KserveV2OptionsMixin(BaseModel):
    """Mixin providing common KServe v2 API configuration fields.

    This mixin can be used by any options class that needs to connect
    to a KServe v2-compatible inference server (Triton, KServe, etc.).
    It provides all the necessary configuration for both HTTP and gRPC
    transports, including authentication, TLS, and performance tuning.
    """

    url: str = Field(
        description=(
            "Endpoint URL for KServe v2 transport. "
            "For transport='http', use http(s)://host[:port] or plain host:port. "
            "For transport='grpc', use plain host:port."
        ),
    )

    model_name: Optional[str] = Field(
        default=None,
        description=(
            "Remote model name registered in the KServe v2 endpoint. "
            "If omitted, a repo_id-derived default is used."
        ),
    )

    model_version: Optional[str] = Field(
        default=None,
        description="Optional model version. If omitted, the server default is used.",
    )

    transport: Literal["grpc", "http"] = Field(
        default="grpc",
        description=(
            "Transport protocol for KServe v2 calls. "
            "Use 'grpc' or 'http' for KServe v2 inference."
        ),
    )

    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional HTTP headers for authentication/routing when transport='http'.",
    )

    grpc_metadata: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional gRPC metadata for authentication/routing when transport='grpc'. "
            "No HTTP headers are reused in gRPC mode."
        ),
    )

    grpc_use_tls: bool = Field(
        default=False,
        description=(
            "Whether to use TLS for the gRPC channel. "
            "When omitted, plain-text h2c is used."
        ),
    )

    grpc_max_message_bytes: int = Field(
        default=64 * 1024 * 1024,
        ge=1,
        description="Max send/receive gRPC message size in bytes.",
    )

    use_binary_data: bool = Field(
        default=True,
        validation_alias=AliasChoices("use_binary_data", "grpc_use_binary_data"),
        description=(
            "For gRPC this controls binary_data tensor handling; for HTTP this enables REST binary framing."
        ),
    )

    timeout: float = Field(
        default=60.0,
        description="Per-request timeout in seconds for both HTTP and gRPC calls.",
    )

    request_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional top-level KServe v2 infer request parameters.",
    )

    @model_validator(mode="before")
    @classmethod
    def _warn_deprecated_alias(cls, data):
        """Emit deprecation warning if old field name is used during initialization."""
        if isinstance(data, dict) and "grpc_use_binary_data" in data:
            warnings.warn(
                "grpc_use_binary_data is deprecated; use use_binary_data instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return data

    @property
    def grpc_use_binary_data(self) -> bool:
        warnings.warn(
            "grpc_use_binary_data is deprecated; use use_binary_data instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.use_binary_data


# Made with Bob
