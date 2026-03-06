"""Tests for KServe v2 gRPC client behavior."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("tritonclient.grpc")

from tritonclient.grpc import service_pb2

from docling.models.inference_engines.common import kserve_v2_grpc as grpc_module


class _DummyChannel:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _RecordingStub:
    def __init__(self, response: service_pb2.ModelInferResponse) -> None:
        self.response = response
        self.last_request: service_pb2.ModelInferRequest | None = None

    def ModelMetadata(
        self, request: Any, timeout: float, metadata: Any
    ) -> service_pb2.ModelMetadataResponse:
        metadata_response = service_pb2.ModelMetadataResponse()
        metadata_response.name = request.name
        return metadata_response

    def ModelInfer(
        self, request: Any, timeout: float, metadata: Any
    ) -> service_pb2.ModelInferResponse:
        self.last_request = request
        return self.response


def _install_stub(
    monkeypatch: pytest.MonkeyPatch, response: service_pb2.ModelInferResponse
) -> tuple[_DummyChannel, _RecordingStub]:
    dummy_channel = _DummyChannel()
    stub = _RecordingStub(response=response)

    monkeypatch.setattr(
        grpc_module.grpc,
        "insecure_channel",
        lambda endpoint, options: dummy_channel,
    )
    monkeypatch.setattr(
        grpc_module.service_pb2_grpc,
        "GRPCInferenceServiceStub",
        lambda channel: stub,
    )
    return dummy_channel, stub


def test_resolve_grpc_endpoint_requires_explicit_port() -> None:
    with pytest.raises(ValueError, match="must include an explicit port"):
        grpc_module._resolve_grpc_endpoint(base_url="dns://localhost")


@pytest.mark.parametrize(
    ("raw_url", "expected"),
    [
        ("dns://localhost:9000", "dns:///localhost:9000"),
        ("static://localhost:9000", "static:///localhost:9000"),
        ("localhost:9000", "dns:///localhost:9000"),
    ],
)
def test_resolve_grpc_endpoint_accepts_supported_formats(
    raw_url: str, expected: str
) -> None:
    assert grpc_module._resolve_grpc_endpoint(base_url=raw_url) == expected


def test_infer_binary_mode_encodes_binary_parameters_and_decodes_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = service_pb2.ModelInferResponse()
    output = response.outputs.add()
    output.name = "scores"
    output.datatype = "FP32"
    output.shape.extend([1, 2])
    response.raw_output_contents.append(
        np.asarray([0.1, 0.9], dtype=np.float32).tobytes()
    )

    dummy_channel, stub = _install_stub(monkeypatch=monkeypatch, response=response)

    client = grpc_module.KserveV2GrpcClient(
        base_url="dns://localhost:9000",
        model_name="m",
        model_version=None,
        timeout=10.0,
        metadata={},
        use_tls=False,
        max_message_bytes=4 * 1024 * 1024,
        use_binary_data=True,
    )

    outputs = client.infer(
        inputs={"pixel_values": np.asarray([[1.0, 2.0]], dtype=np.float32)},
        output_names=["scores"],
    )

    assert np.allclose(outputs["scores"], np.asarray([[0.1, 0.9]], dtype=np.float32))
    assert stub.last_request is not None
    assert len(stub.last_request.raw_input_contents) == 1
    assert stub.last_request.inputs[0].parameters["binary_data"].bool_param
    assert stub.last_request.outputs[0].parameters["binary_data"].bool_param

    client.close()
    assert dummy_channel.closed


def test_infer_non_binary_mode_uses_contents_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = service_pb2.ModelInferResponse()
    output = response.outputs.add()
    output.name = "scores"
    output.datatype = "FP32"
    output.shape.extend([1, 2])
    output.contents.fp32_contents.extend([0.25, 0.75])

    _, stub = _install_stub(monkeypatch=monkeypatch, response=response)

    client = grpc_module.KserveV2GrpcClient(
        base_url="dns://localhost:9000",
        model_name="m",
        model_version=None,
        timeout=10.0,
        metadata={},
        use_tls=False,
        max_message_bytes=4 * 1024 * 1024,
        use_binary_data=False,
    )

    outputs = client.infer(
        inputs={"pixel_values": np.asarray([[3.0, 4.0]], dtype=np.float32)},
        output_names=["scores"],
    )

    assert np.allclose(outputs["scores"], np.asarray([[0.25, 0.75]], dtype=np.float32))
    assert stub.last_request is not None
    assert len(stub.last_request.raw_input_contents) == 0
    assert len(stub.last_request.inputs[0].contents.fp32_contents) == 2
    assert "binary_data" not in stub.last_request.outputs[0].parameters
