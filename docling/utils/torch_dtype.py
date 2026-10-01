"""Resolve a preferred torch dtype against what the target device supports."""

import logging
from typing import Optional

import torch

_log = logging.getLogger(__name__)


def cpu_has_native_bfloat16() -> bool:
    """Whether this CPU runs bfloat16 natively (AVX512-BF16 / AMX) via oneDNN.

    Without it, PyTorch emulates bfloat16 on CPU, which is much slower than float32.
    """
    return bool(
        torch.backends.mkldnn.is_available()
        and torch.ops.mkldnn._is_mkldnn_bf16_supported()
    )


def is_dtype_supported(dtype: str, device: str) -> bool:
    """Whether ``dtype`` runs natively and efficiently on ``device``.

    Only dtypes with known device limitations are rejected: bfloat16 on a CPU
    without native support (emulated, several times slower than float32) and
    float16 on CPU (poorly supported by many CPU kernels).
    """
    if not device.startswith("cpu"):
        return True
    if dtype == "bfloat16":
        return cpu_has_native_bfloat16()
    if dtype == "float16":
        return False
    return True


def resolve_torch_dtype(
    dtype: Optional[str], fallback: Optional[str], device: str
) -> Optional[str]:
    """Return ``dtype`` if the device supports it, otherwise ``fallback``.

    Without a ``fallback`` the configured dtype is returned unchanged, so callers
    that do not opt in keep their current behavior.
    """
    if dtype is None or fallback is None or dtype == fallback:
        return dtype
    if is_dtype_supported(dtype, device):
        return dtype
    _log.info(
        "torch dtype '%s' is not natively supported on device '%s'; using '%s'.",
        dtype,
        device,
        fallback,
    )
    return fallback
