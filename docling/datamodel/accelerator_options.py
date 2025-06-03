import logging
import os
import re
from enum import Enum
from typing import Any, Union

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_log = logging.getLogger(__name__)


class AcceleratorDevice(str, Enum):
    """Devices to run model inference"""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class AcceleratorOptions(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCLING_", env_nested_delimiter="_", populate_by_name=True
    )

    num_threads: int = 4
    device: Union[str, AcceleratorDevice] = "auto"
    cuda_use_flash_attention2: bool = False

    @field_validator("device")
    def validate_device(cls, value):
        # "auto", "cpu", "cuda", "mps", or "cuda:N"
        if value in {d.value for d in AcceleratorDevice} or re.match(
            r"^cuda(:\d+)?$", value
        ):
            return value
        raise ValueError(
            "Invalid device option. Use 'auto', 'cpu', 'mps', 'cuda', or 'cuda:N'."
        )

    @model_validator(mode="before")
    @classmethod
    def check_alternative_envvars(cls, data: Any) -> Any:
        r"""
        Set num_threads from the "alternative" envvar OMP_NUM_THREADS.
        The alternative envvar is used only if it is valid and the regular envvar is not set.

        Notice: The standard pydantic settings mechanism with parameter "aliases" does not provide
        the same functionality. In case the alias envvar is set and the user tries to override the
        parameter in settings initialization, Pydantic treats the parameter provided in __init__()
        as an extra input instead of simply overwriting the evvar value for that parameter.
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
