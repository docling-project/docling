"""Shared KServe v2 protocol types and dtype mappings."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel

# KServe v2 protocol uses the same data type names as Triton Inference Server.
KSERVE_V2_NUMPY_DATATYPES: Dict[str, np.dtype[Any]] = {
    "BOOL": np.dtype(np.bool_),
    "UINT8": np.dtype(np.uint8),
    "UINT16": np.dtype(np.uint16),
    "UINT32": np.dtype(np.uint32),
    "UINT64": np.dtype(np.uint64),
    "INT8": np.dtype(np.int8),
    "INT16": np.dtype(np.int16),
    "INT32": np.dtype(np.int32),
    "INT64": np.dtype(np.int64),
    "FP16": np.dtype(np.float16),
    "FP32": np.dtype(np.float32),
    "FP64": np.dtype(np.float64),
}

NUMPY_KSERVE_V2_DATATYPES: Dict[np.dtype[Any], str] = {
    dtype: name for name, dtype in KSERVE_V2_NUMPY_DATATYPES.items()
}


class KserveV2ModelTensorSpec(BaseModel):
    """Tensor metadata entry returned by KServe v2 model metadata endpoint."""

    name: str
    datatype: str
    shape: List[int | str]


class KserveV2ModelMetadataResponse(BaseModel):
    """KServe v2 model metadata response payload."""

    name: str
    versions: Optional[List[str]] = None
    platform: Optional[str] = None
    inputs: List[KserveV2ModelTensorSpec]
    outputs: List[KserveV2ModelTensorSpec]
