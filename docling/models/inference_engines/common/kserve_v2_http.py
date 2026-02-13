"""Utilities for calling KServe v2 REST inference endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import requests
from pydantic import BaseModel

TRITON_NUMPY_DATATYPES: Dict[str, np.dtype[Any]] = {
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

NUMPY_TRITON_DATATYPES: Dict[np.dtype[Any], str] = {
    dtype: name for name, dtype in TRITON_NUMPY_DATATYPES.items()
}


def _encode_input_tensor(name: str, tensor: np.ndarray) -> Dict[str, Any]:
    triton_dtype = NUMPY_TRITON_DATATYPES.get(tensor.dtype)
    if triton_dtype is None:
        raise ValueError(f"Unsupported numpy dtype for Triton input: {tensor.dtype!s}")

    return {
        "name": name,
        "shape": list(tensor.shape),
        "datatype": triton_dtype,
        "data": tensor.reshape(-1).tolist(),
    }


class KserveV2OutputTensor(BaseModel):
    """Single output tensor in KServe v2 response payload."""

    name: str
    datatype: str
    shape: List[int]
    data: Optional[List[Any]] = None


class KserveV2InferResponse(BaseModel):
    """KServe v2 infer response payload."""

    outputs: List[KserveV2OutputTensor]


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


def _decode_output_tensor(raw_output: KserveV2OutputTensor) -> np.ndarray:
    np_dtype = TRITON_NUMPY_DATATYPES.get(raw_output.datatype)
    if np_dtype is None:
        raise RuntimeError(f"Unsupported Triton output datatype: {raw_output.datatype}")

    if raw_output.data is None:
        raise RuntimeError(
            "Triton binary output mode is not supported by this scaffold. "
            "Configure server/client for JSON outputs with inline data."
        )

    shape = tuple(int(dim) for dim in raw_output.shape)
    array = np.asarray(raw_output.data, dtype=np_dtype)
    return array.reshape(shape)


@dataclass(frozen=True)
class KserveV2HttpClient:
    """Minimal client for KServe v2 JSON infer requests."""

    base_url: str
    model_name: str
    model_version: Optional[str]
    timeout: float
    headers: Mapping[str, str]

    @property
    def infer_url(self) -> str:
        root = self.base_url.rstrip("/")
        if root.endswith("/v2"):
            root = root[: -len("/v2")]

        if "/v2/models/" in root:
            if root.endswith("/infer"):
                return root
            return f"{root}/infer"

        if self.model_version:
            return (
                f"{root}/v2/models/{self.model_name}/versions/"
                f"{self.model_version}/infer"
            )
        return f"{root}/v2/models/{self.model_name}/infer"

    @property
    def model_metadata_url(self) -> str:
        root = self.base_url.rstrip("/")
        if root.endswith("/v2"):
            root = root[: -len("/v2")]

        if "/v2/models/" in root:
            if root.endswith("/infer"):
                return root[: -len("/infer")]
            return root

        if self.model_version:
            return f"{root}/v2/models/{self.model_name}/versions/{self.model_version}"
        return f"{root}/v2/models/{self.model_name}"

    def get_model_metadata(self) -> KserveV2ModelMetadataResponse:
        response = requests.get(
            self.model_metadata_url,
            headers=dict(self.headers),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return KserveV2ModelMetadataResponse.model_validate(response.json())

    def infer(
        self,
        *,
        inputs: Mapping[str, np.ndarray],
        output_names: list[str],
        request_parameters: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        payload: Dict[str, Any] = {
            "inputs": [
                _encode_input_tensor(name=input_name, tensor=tensor)
                for input_name, tensor in inputs.items()
            ]
        }

        if output_names:
            payload["outputs"] = [{"name": output_name} for output_name in output_names]

        if request_parameters:
            payload["parameters"] = dict(request_parameters)

        response = requests.post(
            self.infer_url,
            json=payload,
            headers=dict(self.headers),
            timeout=self.timeout,
        )
        response.raise_for_status()

        body = KserveV2InferResponse.model_validate(response.json())

        decoded_outputs: Dict[str, np.ndarray] = {}
        for output in body.outputs:
            decoded_outputs[output.name] = _decode_output_tensor(output)

        return decoded_outputs
