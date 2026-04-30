"""Utilities for calling KServe v2 REST inference endpoints.

Note: This is a minimal synchronous implementation. The official KServe Python SDK
(https://github.com/kserve/kserve) provides an async InferenceRESTClient with similar
functionality, but is currently in alpha and requires async/await support.
"""

from __future__ import annotations

import json
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
from docling.models.inference_engines.common.kserve_v2_utils import (
    decode_bytes_tensor,
    encode_bytes_tensor,
)

_log = logging.getLogger(__name__)
_INFERENCE_HEADER_CONTENT_LENGTH = "Inference-Header-Content-Length"


def _tensor_kserve_dtype(tensor: np.ndarray) -> str:
    kserve_dtype = NUMPY_KSERVE_V2_DATATYPES.get(tensor.dtype)
    if kserve_dtype is None:
        raise ValueError(
            f"Unsupported numpy dtype for KServe v2 input: {tensor.dtype!s}. "
            f"Supported types: {list(NUMPY_KSERVE_V2_DATATYPES.keys())}"
        )
    return kserve_dtype


def _encode_input_tensor(name: str, tensor: np.ndarray) -> Dict[str, Any]:
    kserve_dtype = _tensor_kserve_dtype(tensor)

    return {
        "name": name,
        "shape": list(tensor.shape),
        "datatype": kserve_dtype,
        "data": tensor.reshape(-1).tolist(),
    }


def _encode_binary_input_tensor(
    name: str, tensor: np.ndarray
) -> tuple[Dict[str, Any], bytes]:
    kserve_dtype = _tensor_kserve_dtype(tensor)
    if kserve_dtype == "BYTES":
        raw_payload = encode_bytes_tensor(tensor)
    else:
        raw_payload = np.ascontiguousarray(tensor).tobytes()

    return (
        {
            "name": name,
            "shape": list(tensor.shape),
            "datatype": kserve_dtype,
            "parameters": {"binary_data_size": len(raw_payload)},
        },
        raw_payload,
    )


class KserveV2OutputTensor(BaseModel):
    """Single output tensor in KServe v2 response payload."""

    name: str
    datatype: str
    shape: List[int]
    data: Optional[List[Any]] = None
    parameters: Optional[Dict[str, Any]] = None


class KserveV2InferResponse(BaseModel):
    """KServe v2 infer response payload."""

    outputs: List[KserveV2OutputTensor]


def _decode_output_tensor(raw_output: KserveV2OutputTensor) -> np.ndarray:
    shape = tuple(int(dim) for dim in raw_output.shape)
    np_dtype = KSERVE_V2_NUMPY_DATATYPES.get(raw_output.datatype)
    if np_dtype is None:
        raise RuntimeError(
            f"Unsupported KServe v2 output datatype: {raw_output.datatype}. "
            f"Supported types: {list(KSERVE_V2_NUMPY_DATATYPES.keys())}"
        )

    if raw_output.data is not None:
        array = np.asarray(raw_output.data, dtype=np_dtype)
        return array.reshape(shape)

    raise RuntimeError(
        f"KServe v2 output tensor {raw_output.name} did not include inline data."
    )


def _decode_binary_output_tensor(
    raw_output: KserveV2OutputTensor, raw_payload: bytes
) -> np.ndarray:
    np_dtype = KSERVE_V2_NUMPY_DATATYPES.get(raw_output.datatype)
    if np_dtype is None:
        raise RuntimeError(
            f"Unsupported KServe v2 output datatype: {raw_output.datatype}. "
            f"Supported types: {list(KSERVE_V2_NUMPY_DATATYPES.keys())}"
        )

    shape = tuple(int(dim) for dim in raw_output.shape)
    if raw_output.datatype == "BYTES":
        return decode_bytes_tensor(raw_payload, shape)

    return np.frombuffer(raw_payload, dtype=np_dtype).reshape(shape)


def _parse_binary_data_size(parameters: Mapping[str, Any] | None) -> int | None:
    if not parameters or "binary_data_size" not in parameters:
        return None

    size = parameters["binary_data_size"]
    try:
        parsed_size = int(size)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid binary_data_size value: {size!r}") from exc
    if parsed_size < 0:
        raise RuntimeError(f"Invalid binary_data_size value: {parsed_size}")
    return parsed_size


def _build_binary_request(
    *,
    inputs: Mapping[str, np.ndarray],
    output_names: list[str],
    request_parameters: Optional[Mapping[str, Any]],
) -> tuple[Dict[str, str], bytes]:
    raw_inputs: list[bytes] = []
    payload: Dict[str, Any] = {"inputs": []}
    for input_name, tensor in inputs.items():
        encoded_tensor, raw_payload = _encode_binary_input_tensor(
            name=input_name, tensor=np.asarray(tensor)
        )
        payload["inputs"].append(encoded_tensor)
        raw_inputs.append(raw_payload)

    if output_names:
        payload["outputs"] = [
            {"name": output_name, "parameters": {"binary_data": True}}
            for output_name in output_names
        ]

    if request_parameters:
        payload["parameters"] = dict(request_parameters)

    header_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request_body = header_bytes + b"".join(raw_inputs)
    request_headers = {
        "Content-Type": "application/octet-stream",
        _INFERENCE_HEADER_CONTENT_LENGTH: str(len(header_bytes)),
    }
    return request_headers, request_body


def _decode_binary_response(response: requests.Response) -> KserveV2InferResponse:
    header_len_text = response.headers.get(_INFERENCE_HEADER_CONTENT_LENGTH)
    if header_len_text is None:
        try:
            return KserveV2InferResponse.model_validate(response.json())
        except Exception as exc:
            raise RuntimeError(
                f"Binary KServe response from {response.url} did not include "
                f"{_INFERENCE_HEADER_CONTENT_LENGTH} and was not valid JSON: {exc}"
            ) from exc

    try:
        header_len = int(header_len_text)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid {_INFERENCE_HEADER_CONTENT_LENGTH} value: {header_len_text!r}"
        ) from exc

    if header_len < 0 or header_len > len(response.content):
        raise RuntimeError(
            f"Invalid {_INFERENCE_HEADER_CONTENT_LENGTH} value: {header_len}"
        )

    try:
        header_json = json.loads(response.content[:header_len].decode("utf-8"))
        return KserveV2InferResponse.model_validate(header_json)
    except Exception as exc:
        raise RuntimeError(
            f"Invalid binary inference response header from {response.url}: {exc}"
        ) from exc


@dataclass(frozen=True)
class KserveV2HttpClient:
    """Minimal client for KServe v2 REST infer requests."""

    base_url: str
    model_name: str
    model_version: Optional[str]
    timeout: float
    headers: Mapping[str, str]
    use_binary_data: bool = True

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
        request_headers = dict(self.headers)
        extra_headers = kwargs.pop("headers", None)
        if extra_headers is not None:
            request_headers.update(dict(extra_headers))
        try:
            if method == "GET":
                response = requests.get(
                    url, headers=request_headers, timeout=self.timeout, **kwargs
                )
            else:  # POST
                response = requests.post(
                    url, headers=request_headers, timeout=self.timeout, **kwargs
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
            response = exc.response
            status_code = response.status_code if response is not None else "unknown"
            response_text = response.text if response is not None else ""
            raise requests.exceptions.HTTPError(
                f"HTTP error {status_code} from {url}: {response_text}"
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
        _batch_size = next(iter(inputs.values())).shape[0] if inputs else 0
        _t_ser_start = _t_ser_mono = _t_http_start = _t_http_mono = 0.0
        _t_deser_start = _t_deser_mono = 0.0

        if _log.isEnabledFor(logging.DEBUG):
            _t_ser_start = time.time()
            _t_ser_mono = time.monotonic()
        request_kwargs: Dict[str, Any]
        if self.use_binary_data:
            binary_headers, request_body = _build_binary_request(
                inputs=inputs,
                output_names=output_names,
                request_parameters=request_parameters,
            )
            request_kwargs = {
                "data": request_body,
                "headers": {**dict(self.headers), **binary_headers},
            }
        else:
            payload: Dict[str, Any] = {
                "inputs": [
                    _encode_input_tensor(name=input_name, tensor=tensor)
                    for input_name, tensor in inputs.items()
                ]
            }

            if output_names:
                payload["outputs"] = [
                    {"name": output_name} for output_name in output_names
                ]

            if request_parameters:
                payload["parameters"] = dict(request_parameters)

            request_kwargs = {"json": payload}

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
            self.infer_url, method="POST", **request_kwargs
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
            body = (
                _decode_binary_response(response)
                if self.use_binary_data
                else KserveV2InferResponse.model_validate(response.json())
            )
        except Exception as exc:
            raise RuntimeError(
                f"Invalid inference response from {self.infer_url}: {exc}"
            ) from exc

        decoded_outputs: Dict[str, np.ndarray] = {}
        header_len_text = response.headers.get(_INFERENCE_HEADER_CONTENT_LENGTH)
        raw_body = b""
        if self.use_binary_data and header_len_text is not None:
            raw_body = response.content[int(header_len_text) :]
        raw_offset = 0
        for output in body.outputs:
            binary_data_size = _parse_binary_data_size(output.parameters)
            if binary_data_size is None:
                decoded_outputs[output.name] = _decode_output_tensor(output)
                continue

            raw_end = raw_offset + binary_data_size
            if raw_end > len(raw_body):
                raise RuntimeError(
                    "KServe v2 HTTP response did not include enough binary output data "
                    f"for tensor {output.name}: expected {binary_data_size} bytes at "
                    f"offset {raw_offset}, got {len(raw_body) - raw_offset}"
                )
            decoded_outputs[output.name] = _decode_binary_output_tensor(
                output, raw_body[raw_offset:raw_end]
            )
            raw_offset = raw_end

        if raw_offset != len(raw_body):
            raise RuntimeError(
                "KServe v2 HTTP response included trailing binary output data that was "
                f"not consumed: {len(raw_body) - raw_offset} bytes"
            )

        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(
                "PIPELINE_PROFILING KServe infer deserialization: batch_size=%d start=%.3f end=%.3f duration=%.3fs",
                _batch_size,
                _t_deser_start,
                time.time(),
                time.monotonic() - _t_deser_mono,
            )

        return decoded_outputs
