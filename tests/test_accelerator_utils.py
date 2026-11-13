# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Unit tests for accelerator utility functions."""

import pytest

from docling.utils.accelerator_utils import resolve_torch_dtype


@pytest.mark.parametrize(
    ("preferred", "device", "cpu_capability", "cuda_capability", "expected"),
    [
        # CPUs without AVX-512 (e.g. AVX2 laptops, ARM) emulate bfloat16.
        ("bfloat16", "cpu", "AVX2", None, "float32"),
        ("bfloat16", "cpu", "DEFAULT", None, "float32"),
        # AVX-512 / AMX CPUs run bfloat16 natively.
        ("bfloat16", "cpu", "AVX512", None, "bfloat16"),
        # CUDA before Ampere (e.g. T4) has no native bfloat16.
        ("bfloat16", "cuda:0", None, (7, 5), "float32"),
        ("bfloat16", "cuda:0", None, (8, 0), "bfloat16"),
        # Devices other than CPU and CUDA keep the preferred dtype.
        ("bfloat16", "mps", None, None, "bfloat16"),
        ("bfloat16", "xpu", None, None, "bfloat16"),
        # Only bfloat16 is checked.
        ("float16", "cpu", "AVX2", None, "float16"),
        (None, "cpu", "AVX2", None, None),
    ],
)
def test_resolve_torch_dtype(
    monkeypatch, preferred, device, cpu_capability, cuda_capability, expected
):
    import torch

    monkeypatch.setattr(
        torch.backends.cpu, "get_cpu_capability", lambda: cpu_capability
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: cuda_capability
    )

    assert resolve_torch_dtype(preferred, device) == expected
