"""Shared protocol for KServe v2 transport clients."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Protocol

import numpy as np

from docling.models.inference_engines.common.kserve_v2_types import (
    KserveV2ModelMetadataResponse,
)


class KserveV2Client(Protocol):
    """Transport-agnostic KServe v2 client interface."""

    def get_model_metadata(self) -> KserveV2ModelMetadataResponse:
        """Fetch model metadata for tensor name/schema resolution."""

    def infer(
        self,
        *,
        inputs: Mapping[str, np.ndarray],
        output_names: list[str],
        request_parameters: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """Execute model inference and return outputs keyed by tensor name."""

    def close(self) -> None:
        """Release client resources."""
