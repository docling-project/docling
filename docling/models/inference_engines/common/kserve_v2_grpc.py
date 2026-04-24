"""Utilities for calling KServe v2 gRPC inference endpoints via tritonclient."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

try:
    import grpc  # type: ignore[import-untyped]
    from tritonclient.grpc import (  # type: ignore[import-untyped, import-not-found]
        service_pb2,
        service_pb2_grpc,
    )
except ImportError:
    grpc = None  # type: ignore[assignment]
    service_pb2 = None  # type: ignore[assignment]
    service_pb2_grpc = None  # type: ignore[assignment]

from docling.models.inference_engines.common.kserve_v2_types import (
    KSERVE_V2_NUMPY_DATATYPES,
    NUMPY_KSERVE_V2_DATATYPES,
    KserveV2ModelMetadataResponse,
    KserveV2ModelTensorSpec,
)
from docling.models.inference_engines.common.kserve_v2_utils import (
    decode_bytes_tensor,
    encode_bytes_element,
    encode_bytes_tensor,
)

_log = logging.getLogger(__name__)


def _resolve_grpc_endpoint(
    *,
    base_url: str,
) -> str:
    if "://" in base_url:
        raise ValueError(f"KServe gRPC URL must be plain host:port. Got: {base_url}")

    host, separator, port_text = base_url.rpartition(":")
    if separator == "" or host == "" or port_text == "":
        raise ValueError(
            "Invalid KServe URL for gRPC transport. "
            f"Expected plain host:port, got: {base_url}"
        )
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(
            "Invalid KServe URL for gRPC transport. "
            f"Expected plain host:port with numeric port, got: {base_url}"
        ) from exc
    if port < 1 or port > 65535:
        raise ValueError(
            "Invalid KServe URL for gRPC transport. "
            f"Port must be in range 1..65535, got: {port}"
        )
    return f"{host}:{port}"


def _to_grpc_metadata(metadata: Mapping[str, str]) -> Sequence[Tuple[str, str]]:
    return [(key, value) for key, value in metadata.items()]


def _set_request_parameter(
    parameter_map: Any,
    key: str,
    value: Any,
) -> None:
    parameter = parameter_map[key]
    if isinstance(value, bool | np.bool_):
        parameter.bool_param = bool(value)
        return
    if isinstance(value, int | np.integer):
        int_value = int(value)
        if int_value < 0:
            parameter.int64_param = int_value
            return
        if int_value <= (2**63 - 1):
            parameter.int64_param = int_value
            return
        if int_value <= (2**64 - 1):
            parameter.uint64_param = int_value
            return
        raise ValueError(
            "Unsupported KServe request parameter integer range for gRPC: "
            f"key={key}, value={int_value}"
        )
    if isinstance(value, float | np.floating):
        parameter.double_param = float(value)
        return
    if isinstance(value, str):
        parameter.string_param = value
        return
    raise ValueError(
        "Unsupported KServe request parameter type for gRPC: "
        f"key={key}, type={type(value)}. Supported: bool, int, float, str."
    )


def _encode_contents(tensor: np.ndarray, contents: Any) -> None:
    """Populate an InferTensorContents message from a numpy array (non-binary path)."""
    flat = tensor.flatten()
    if tensor.dtype == np.float32:
        contents.fp32_contents.extend(flat.tolist())
    elif tensor.dtype == np.float64:
        contents.fp64_contents.extend(flat.tolist())
    elif tensor.dtype in (np.int8, np.int16, np.int32):
        contents.int_contents.extend(flat.astype(np.int32).tolist())
    elif tensor.dtype == np.int64:
        contents.int64_contents.extend(flat.tolist())
    elif tensor.dtype in (np.uint8, np.uint16, np.uint32):
        contents.uint_contents.extend(flat.astype(np.uint32).tolist())
    elif tensor.dtype == np.uint64:
        contents.uint64_contents.extend(flat.tolist())
    elif tensor.dtype == np.bool_:
        contents.bool_contents.extend(flat.tolist())
    elif tensor.dtype == object:
        contents.bytes_contents.extend(encode_bytes_element(value) for value in flat)
    else:
        raise ValueError(
            f"Unsupported numpy dtype for gRPC inline (non-binary) encoding: {tensor.dtype!s}. "
            "Supported non-binary dtypes: bool, uint8/uint16/uint32/uint64, "
            "int8/int16/int32/int64, float32/float64, BYTES."
        )


def _decode_contents(
    contents: Any, np_dtype: np.dtype[Any], shape: tuple[int, ...]
) -> np.ndarray:
    """Decode an InferTensorContents message to a numpy array (non-binary path)."""
    canonical_dtype = np.dtype(np_dtype)

    if canonical_dtype == np.dtype(np.float32):
        data = list(contents.fp32_contents)
    elif canonical_dtype == np.dtype(np.float64):
        data = list(contents.fp64_contents)
    elif canonical_dtype in (
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.int32),
    ):
        data = list(contents.int_contents)
    elif canonical_dtype == np.dtype(np.int64):
        data = list(contents.int64_contents)
    elif canonical_dtype in (
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.uint32),
    ):
        data = list(contents.uint_contents)
    elif canonical_dtype == np.dtype(np.uint64):
        data = list(contents.uint64_contents)
    elif canonical_dtype == np.dtype(np.bool_):
        data = list(contents.bool_contents)
    elif canonical_dtype == np.dtype(object):
        data = list(contents.bytes_contents)
    else:
        raise RuntimeError(
            f"Unsupported numpy dtype for gRPC inline (non-binary) decoding: {canonical_dtype!s}. "
            "Supported non-binary dtypes: bool, uint8/uint16/uint32/uint64, "
            "int8/int16/int32/int64, float32/float64, BYTES."
        )
    return np.asarray(data, dtype=canonical_dtype).reshape(shape)


@dataclass
class KserveV2GrpcClient:
    """Minimal client for KServe v2 gRPC infer requests."""

    base_url: str
    model_name: str
    model_version: str | None
    timeout: float
    metadata: Mapping[str, str]
    use_tls: bool
    max_message_bytes: int
    use_binary_data: bool = True

    def __post_init__(self) -> None:
        if grpc is None or service_pb2 is None or service_pb2_grpc is None:
            raise ImportError(
                "gRPC transport requires the 'remote-serving' extras. "
                "Install with: pip install 'docling[remote-serving]'"
            )

        endpoint = _resolve_grpc_endpoint(
            base_url=self.base_url,
        )
        channel_options = [
            ("grpc.max_send_message_length", self.max_message_bytes),
            ("grpc.max_receive_message_length", self.max_message_bytes),
        ]
        if self.use_tls:
            credentials = grpc.ssl_channel_credentials()
            self._channel = grpc.secure_channel(
                endpoint,
                credentials,
                options=channel_options,
            )
        else:
            self._channel = grpc.insecure_channel(
                endpoint,
                options=channel_options,
            )
        self._stub = service_pb2_grpc.GRPCInferenceServiceStub(self._channel)
        self._grpc_metadata = _to_grpc_metadata(self.metadata)

    def __del__(self) -> None:
        try:
            self._channel.close()
        except Exception:
            pass

    def close(self) -> None:
        self._channel.close()

    def get_model_metadata(self) -> KserveV2ModelMetadataResponse:
        request = service_pb2.ModelMetadataRequest(name=self.model_name)
        if self.model_version:
            request.version = self.model_version

        try:
            response = self._stub.ModelMetadata(
                request,
                timeout=self.timeout,
                metadata=self._grpc_metadata,
            )
        except grpc.RpcError as exc:
            raise RuntimeError(
                f"gRPC metadata call failed for model {self.model_name}: {exc}"
            ) from exc

        inputs = [
            KserveV2ModelTensorSpec(
                name=input_tensor.name,
                datatype=input_tensor.datatype,
                shape=[int(dim) for dim in input_tensor.shape],
            )
            for input_tensor in response.inputs
        ]
        outputs = [
            KserveV2ModelTensorSpec(
                name=output_tensor.name,
                datatype=output_tensor.datatype,
                shape=[int(dim) for dim in output_tensor.shape],
            )
            for output_tensor in response.outputs
        ]

        return KserveV2ModelMetadataResponse(
            name=response.name,
            versions=list(response.versions) if response.versions else None,
            platform=response.platform or None,
            inputs=inputs,
            outputs=outputs,
        )

    def infer(
        self,
        *,
        inputs: Mapping[str, np.ndarray],
        output_names: list[str],
        request_parameters: Mapping[str, Any] | None = None,
    ) -> Dict[str, np.ndarray]:
        _batch_size = next(iter(inputs.values())).shape[0] if inputs else 0

        if _log.isEnabledFor(logging.DEBUG):
            _t_ser_start = time.time()
            _t_ser_mono = time.monotonic()

        request = service_pb2.ModelInferRequest(model_name=self.model_name)
        if self.model_version:
            request.model_version = self.model_version

        if request_parameters:
            for key, value in request_parameters.items():
                _set_request_parameter(request.parameters, key=key, value=value)

        for input_name, tensor in inputs.items():
            np_tensor = np.asarray(tensor)
            kserve_dtype = NUMPY_KSERVE_V2_DATATYPES.get(np_tensor.dtype)
            if kserve_dtype is None:
                raise ValueError(
                    f"Unsupported numpy dtype for KServe v2 gRPC input: {np_tensor.dtype!s}. "
                    f"Supported types: {list(NUMPY_KSERVE_V2_DATATYPES.keys())}"
                )

            input_tensor = request.inputs.add()
            input_tensor.name = input_name
            input_tensor.datatype = kserve_dtype
            input_tensor.shape.extend(int(dim) for dim in np_tensor.shape)

            if self.use_binary_data:
                input_tensor.parameters["binary_data"].bool_param = True
                if kserve_dtype == "BYTES":  # Bytes encoding
                    request.raw_input_contents.append(encode_bytes_tensor(np_tensor))
                else:
                    contiguous = np.ascontiguousarray(np_tensor)
                    request.raw_input_contents.append(contiguous.tobytes())
            else:
                _encode_contents(np_tensor, input_tensor.contents)

        for output_name in output_names:
            output_tensor = request.outputs.add()
            output_tensor.name = output_name
            if self.use_binary_data:
                output_tensor.parameters["binary_data"].bool_param = True

        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(
                "PIPELINE_PROFILING KServe gRPC infer serialization: batch_size=%d start=%.3f end=%.3f duration=%.3fs",
                _batch_size,
                _t_ser_start,
                time.time(),
                time.monotonic() - _t_ser_mono,
            )
            _t_grpc_start = time.time()
            _t_grpc_mono = time.monotonic()

        try:
            response = self._stub.ModelInfer(
                request,
                timeout=self.timeout,
                metadata=self._grpc_metadata,
            )
        except grpc.RpcError as exc:
            raise RuntimeError(
                f"gRPC infer call failed for model {self.model_name}: {exc}"
            ) from exc

        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(
                "PIPELINE_PROFILING KServe gRPC infer round-trip: batch_size=%d start=%.3f end=%.3f duration=%.3fs",
                _batch_size,
                _t_grpc_start,
                time.time(),
                time.monotonic() - _t_grpc_mono,
            )
            _t_deser_start = time.time()
            _t_deser_mono = time.monotonic()

        decoded_outputs: Dict[str, np.ndarray] = {}

        if self.use_binary_data:
            if len(response.raw_output_contents) != len(response.outputs):
                raise RuntimeError(
                    "KServe v2 gRPC response did not include binary output payloads for all tensors. "
                    "Set use_binary_data=False or ensure server supports binary_data outputs."
                )
            for output_tensor, raw_output in zip(
                response.outputs, response.raw_output_contents
            ):
                np_dtype = KSERVE_V2_NUMPY_DATATYPES.get(output_tensor.datatype)
                if np_dtype is None:
                    raise RuntimeError(
                        f"Unsupported KServe v2 gRPC output datatype: {output_tensor.datatype}. "
                        f"Supported types: {list(KSERVE_V2_NUMPY_DATATYPES.keys())}"
                    )
                shape = tuple(int(dim) for dim in output_tensor.shape)

                # Bytes decoding
                # Special handling for BYTES datatype (variable-length strings)
                if output_tensor.datatype == "BYTES":
                    decoded_outputs[output_tensor.name] = decode_bytes_tensor(
                        raw_output, shape
                    )
                else:
                    array = np.frombuffer(raw_output, dtype=np_dtype)
                    decoded_outputs[output_tensor.name] = array.reshape(shape)
        else:
            for output_tensor in response.outputs:
                np_dtype = KSERVE_V2_NUMPY_DATATYPES.get(output_tensor.datatype)
                if np_dtype is None:
                    raise RuntimeError(
                        f"Unsupported KServe v2 gRPC output datatype: {output_tensor.datatype}. "
                        f"Supported types: {list(KSERVE_V2_NUMPY_DATATYPES.keys())}"
                    )
                shape = tuple(int(dim) for dim in output_tensor.shape)
                decoded_outputs[output_tensor.name] = _decode_contents(
                    output_tensor.contents, np_dtype, shape
                )

        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(
                "PIPELINE_PROFILING KServe gRPC infer deserialization: batch_size=%d start=%.3f end=%.3f duration=%.3fs",
                _batch_size,
                _t_deser_start,
                time.time(),
                time.monotonic() - _t_deser_mono,
            )

        return decoded_outputs
