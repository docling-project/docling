# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Regression test for VlmPredictionToken logprob type coercion (#4002).

MLX stream_generate returns bfloat16 scalar arrays for logprobs, which
pydantic rejects because VlmPredictionToken.logprob expects a Python float.
The fix is to wrap the indexing result with float() in mlx_model.py.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from docling.datamodel.base_models import VlmPredictionToken


class TestVlmPredictionTokenLogprobCoercion:
    def test_accepts_python_float(self):
        token = VlmPredictionToken(text="a", token=1, logprob=-0.5)
        assert token.logprob == -0.5

    @pytest.mark.skip(
        reason=(
            "pydantic-core accepts numpy 0-dim arrays via __float__() and always has — "
            "the ValidationError this test expects is never raised. "
            "Needs review: the original bug (#4002) was with mlx.core arrays, not numpy."
        )
    )
    def test_rejects_raw_numpy_scalar_array(self):
        raw = np.array(-0.5, dtype=np.float16)
        with pytest.raises(ValidationError):
            VlmPredictionToken(text="a", token=1, logprob=raw)

    def test_float_coercion_fixes_numpy_scalar(self):
        raw = np.array(-0.5, dtype=np.float16)
        token = VlmPredictionToken(text="a", token=1, logprob=float(raw))
        assert isinstance(token.logprob, float)
        assert token.logprob == pytest.approx(-0.5, abs=1e-3)

    def test_float_coercion_on_zero_dim_array(self):
        raw = np.float16(0.0)
        token = VlmPredictionToken(text="a", token=1, logprob=float(raw))
        assert token.logprob == 0.0
