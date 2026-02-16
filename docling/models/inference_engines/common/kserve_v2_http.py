"""Utilities for calling KServe v2 REST inference endpoints.

Note: This is a minimal synchronous implementation. The official KServe Python SDK
(https://github.com/kserve/kserve) provides an async InferenceRESTClient with similar
functionality, but is currently in alpha and requires async/await support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import requests
from pydantic import BaseModel

# KServe v2 protocol uses the same data type names as Triton Inference Server
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


def _encode_input_tensor(name: str, tensor: np.ndarray) -> Dict[str, Any]:
    kserve_dtype = NUMPY_KSERVE_V2_DATATYPES.get(tensor.dtype)
    if kserve_dtype is None:
        raise ValueError(
            f"Unsupported numpy dtype for KServe v2 input: {tensor.dtype!s}. "
            f"Supported types: {list(NUMPY_KSERVE_V2_DATATYPES.keys())}"
        )

    return {
        "name": name,
        "shape": list(tensor.shape),
        "datatype": kserve_dtype,
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
    np_dtype = KSERVE_V2_NUMPY_DATATYPES.get(raw_output.datatype)
    if np_dtype is None:
        raise RuntimeError(
            f"Unsupported KServe v2 output datatype: {raw_output.datatype}. "
            f"Supported types: {list(KSERVE_V2_NUMPY_DATATYPES.keys())}"
        )

    if raw_output.data is None:
        raise RuntimeError(
            "KServe v2 binary output mode is not supported. "
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

    def _build_model_url(self, *, include_infer_suffix: bool) -> str:
        """Build URL for model metadata or inference endpoint.

        Args:
            include_infer_suffix: If True, append '/infer' to the URL

        Returns:
            Fully constructed URL for the requested endpoint
        """
        root = self.base_url.rstrip("/")

        # Remove trailing /v2 if present
        if root.endswith("/v2"):
            root = root[: -len("/v2")]

        # If URL already contains the full model path, use it directly
        if "/v2/models/" in root:
            if root.endswith("/infer"):
                # Already points to infer endpoint
                return root if include_infer_suffix else root[: -len("/infer")]
            # Points to model path
            return f"{root}/infer" if include_infer_suffix else root

        # Build URL from scratch using model name and optional version
        if self.model_version:
            model_path = (
                f"{root}/v2/models/{self.model_name}/versions/{self.model_version}"
            )
        else:
            model_path = f"{root}/v2/models/{self.model_name}"

        return f"{model_path}/infer" if include_infer_suffix else model_path

    @property
    def infer_url(self) -> str:
        """Get the inference endpoint URL."""
        return self._build_model_url(include_infer_suffix=True)

    @property
    def model_metadata_url(self) -> str:
        """Get the model metadata endpoint URL."""
        return self._build_model_url(include_infer_suffix=False)

    def get_model_metadata(self) -> KserveV2ModelMetadataResponse:
        """Fetch model metadata from KServe v2 endpoint.

        Returns:
            Validated model metadata including inputs/outputs schema

        Raises:
            requests.exceptions.Timeout: If request exceeds timeout
            requests.exceptions.ConnectionError: If cannot connect to server
            requests.exceptions.HTTPError: If server returns error status
            RuntimeError: If response format is invalid
        """
        try:
            response = requests.get(
                self.model_metadata_url,
                headers=dict(self.headers),
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise requests.exceptions.Timeout(
                f"Timeout fetching model metadata from {self.model_metadata_url}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise requests.exceptions.ConnectionError(
                f"Failed to connect to KServe endpoint at {self.model_metadata_url}"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise requests.exceptions.HTTPError(
                f"HTTP error {response.status_code} from {self.model_metadata_url}: "
                f"{response.text}"
            ) from exc

        try:
            return KserveV2ModelMetadataResponse.model_validate(response.json())
        except Exception as exc:
            raise RuntimeError(
                f"Invalid metadata response from {self.model_metadata_url}: {exc}"
            ) from exc

    def infer(
        self,
        *,
        inputs: Mapping[str, np.ndarray],
        output_names: list[str],
        request_parameters: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """Execute inference request against KServe v2 endpoint.

        Args:
            inputs: Mapping of input tensor names to numpy arrays
            output_names: List of expected output tensor names
            request_parameters: Optional KServe v2 request-level parameters

        Returns:
            Mapping of output tensor names to numpy arrays

        Raises:
            requests.exceptions.Timeout: If request exceeds timeout
            requests.exceptions.ConnectionError: If cannot connect to server
            requests.exceptions.HTTPError: If server returns error status
            RuntimeError: If response format is invalid
        """
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

        try:
            response = requests.post(
                self.infer_url,
                json=payload,
                headers=dict(self.headers),
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise requests.exceptions.Timeout(
                f"Timeout during inference request to {self.infer_url}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise requests.exceptions.ConnectionError(
                f"Failed to connect to KServe inference endpoint at {self.infer_url}"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise requests.exceptions.HTTPError(
                f"HTTP error {response.status_code} from {self.infer_url}: "
                f"{response.text}"
            ) from exc

        try:
            body = KserveV2InferResponse.model_validate(response.json())
        except Exception as exc:
            raise RuntimeError(
                f"Invalid inference response from {self.infer_url}: {exc}"
            ) from exc

        decoded_outputs: Dict[str, np.ndarray] = {}
        for output in body.outputs:
            decoded_outputs[output.name] = _decode_output_tensor(output)

        return decoded_outputs
