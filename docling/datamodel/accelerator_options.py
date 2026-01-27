import logging
import os
import re
from enum import Enum
from typing import Annotated, Any, Union

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_log = logging.getLogger(__name__)


class AcceleratorDevice(str, Enum):
    """Devices to run model inference"""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    XPU = "xpu"


class AcceleratorOptions(BaseSettings):
    """Hardware acceleration configuration for model inference.

    Can be configured via environment variables with DOCLING_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="DOCLING_", env_nested_delimiter="_", populate_by_name=True
    )
    num_threads: Annotated[
        int,
        Field(
            description=(
                "Number of CPU threads to use for model inference. Higher values "
                "can improve throughput on multi-core systems but may increase "
                "memory usage. Can be set via DOCLING_NUM_THREADS or "
                "OMP_NUM_THREADS environment variables. Recommended: number of "
                "physical CPU cores."
            )
        ),
    ] = 4
    device: Annotated[
        Union[str, AcceleratorDevice],
        Field(
            description=(
                "Hardware device for model inference. Options: `auto` "
                "(automatic detection), `cpu` (CPU only), `cuda` (NVIDIA GPU), "
                "`cuda:N` (specific GPU), `mps` (Apple Silicon), `xpu` (Intel "
                "GPU). Auto mode selects the best available device. Can be set "
                "via DOCLING_DEVICE environment variable."
            )
        ),
    ] = "auto"
    cuda_use_flash_attention2: Annotated[
        bool,
        Field(
            description=(
                "Enable Flash Attention 2 optimization for CUDA devices. "
                "Provides significant speedup and memory reduction for "
                "transformer models on compatible NVIDIA GPUs (Ampere or newer). "
                "Requires flash-attn package installation. Can be set via "
                "DOCLING_CUDA_USE_FLASH_ATTENTION2 environment variable."
            )
        ),
    ] = False

    @field_validator("device")
    def validate_device(cls, value):
        # "auto", "cpu", "cuda", "mps", "xpu", or "cuda:N"
        if value in {d.value for d in AcceleratorDevice} or re.match(
            r"^cuda(:\d+)?$", value
        ):
            return value
        raise ValueError(
            "Invalid device option. Use `auto`, `cpu`, `mps`, `xpu`, `cuda`, "
            "or `cuda:N`."
        )

    @model_validator(mode="before")
    @classmethod
    def check_alternative_envvars(cls, data: Any) -> Any:
        r"""
        Set num_threads from the "alternative" envvar OMP_NUM_THREADS.
        The alternative envvar is used only if it is valid and the regular
        envvar is not set.

        Notice: The standard pydantic settings mechanism with parameter
        "aliases" does not provide the same functionality. In case the alias
        envvar is set and the user tries to override the parameter in settings
        initialization, Pydantic treats the parameter provided in __init__()
        as an extra input instead of simply overwriting the evvar value for
        that parameter.
        """
        if isinstance(data, dict):
            input_num_threads = data.get("num_threads")
            # Check if to set the num_threads from the alternative envvar
            if input_num_threads is None:
                docling_num_threads = os.getenv("DOCLING_NUM_THREADS")
                omp_num_threads = os.getenv("OMP_NUM_THREADS")
                if docling_num_threads is None and omp_num_threads is not None:
                    try:
                        data["num_threads"] = int(omp_num_threads)
                    except ValueError:
                        _log.error(
                            "Ignoring misformatted envvar OMP_NUM_THREADS '%s'",
                            omp_num_threads,
                        )
        return data
