"""Utilities for calling KServe v2 REST inference endpoints.

Note: This is a minimal synchronous implementation. The official KServe Python SDK
(https://github.com/kserve/kserve) provides an async InferenceRESTClient with similar
functionality, but is currently in alpha and requires async/await support.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import requests
from pydantic import BaseModel

from docling.models.inference_engines.common.kserve_v2_types import (
    KSERVE_V2_NUMPY_DATATYPES,
    NUMPY_KSERVE_V2_DATATYPES,
    KserveV2ModelMetadataResponse,
)

_log = logging.getLogger(__name__)


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

    def close(self) -> None:
        """No-op close for transport parity with gRPC client."""

    def _execute_http_request(
        self,
        url: str,
        method: str = "GET",
        **kwargs: Any,
    ) -> requests.Response:
        """Execute HTTP request with consistent error handling.

        Args:
            url: Target URL
            method: HTTP method (GET or POST)
            **kwargs: Additional arguments passed to requests

        Returns:
            HTTP response object

        Raises:
            requests.exceptions.Timeout: If request exceeds timeout
            requests.exceptions.ConnectionError: If cannot connect to server
            requests.exceptions.HTTPError: If server returns error status
        """
        try:
            if method == "GET":
                response = requests.get(
                    url, headers=dict(self.headers), timeout=self.timeout, **kwargs
                )
            else:  # POST
                response = requests.post(
                    url, headers=dict(self.headers), timeout=self.timeout, **kwargs
                )
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout as exc:
            raise requests.exceptions.Timeout(
                f"Timeout during {method} request to {url}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise requests.exceptions.ConnectionError(
                f"Failed to connect to {url}"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise requests.exceptions.HTTPError(
                f"HTTP error {response.status_code} from {url}: {response.text}"
            ) from exc

    @property
    def model_metadata_url(self) -> str:
        """Get the model metadata endpoint URL.

        Expects base_url to be exactly the base URL without any path components
        (e.g., 'http://localhost:8000', not 'http://localhost:8000/v2').

        Returns:
            Model metadata URL following KServe v2 protocol:
            - {base_url}/v2/models/{model_name}[/versions/{version}]
        """
        base = self.base_url.rstrip("/")
        if self.model_version:
            return f"{base}/v2/models/{self.model_name}/versions/{self.model_version}"
        return f"{base}/v2/models/{self.model_name}"

    @property
    def infer_url(self) -> str:
        """Get the inference endpoint URL.

        Returns:
            Inference URL following KServe v2 protocol:
            - {base_url}/v2/models/{model_name}[/versions/{version}]/infer
        """
        return f"{self.model_metadata_url}/infer"

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
        response = self._execute_http_request(self.model_metadata_url, method="GET")

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
        if _log.isEnabledFor(logging.DEBUG):
            _batch_size = next(iter(inputs.values())).shape[0] if inputs else 0
            _t_ser_start = time.time()
            _t_ser_mono = time.monotonic()
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

        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(
                "PIPELINE_PROFILING KServe infer serialization: batch_size=%d start=%.3f end=%.3f duration=%.3fs",
                _batch_size,
                _t_ser_start,
                time.time(),
                time.monotonic() - _t_ser_mono,
            )
            _t_http_start = time.time()
            _t_http_mono = time.monotonic()
        response = self._execute_http_request(
            self.infer_url, method="POST", json=payload
        )
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(
                "PIPELINE_PROFILING KServe infer http round-trip: batch_size=%d start=%.3f end=%.3f duration=%.3fs",
                _batch_size,
                _t_http_start,
                time.time(),
                time.monotonic() - _t_http_mono,
            )
            _t_deser_start = time.time()
            _t_deser_mono = time.monotonic()

        try:
            body = KserveV2InferResponse.model_validate(response.json())
        except Exception as exc:
            raise RuntimeError(
                f"Invalid inference response from {self.infer_url}: {exc}"
            ) from exc

        decoded_outputs: Dict[str, np.ndarray] = {}
        for output in body.outputs:
            decoded_outputs[output.name] = _decode_output_tensor(output)

        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(
                "PIPELINE_PROFILING KServe infer deserialization: batch_size=%d start=%.3f end=%.3f duration=%.3fs",
                _batch_size,
                _t_deser_start,
                time.time(),
                time.monotonic() - _t_deser_mono,
            )

        return decoded_outputs
