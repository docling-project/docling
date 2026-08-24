"""The KServe v2 REST transport, driven against a real inference server.

``KserveV2HttpClient`` is the shared foundation under every KServe consumer --
object detection, image classification and OCR -- and it uses ``requests``,
so an httpx-only mock cannot see its traffic at all.

The protocol has two wire formats: plain JSON, and a binary extension where
raw tensor bytes follow a JSON header whose length is carried in the
``Inference-Header-Content-Length`` header. The binary form is the default,
and both directions of it are exercised here.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import requests

from docling.models.inference_engines.common.kserve_v2_http import KserveV2HttpClient
from docling.models.inference_engines.common.kserve_v2_types import (
    KserveV2ModelTensorSpec,
)
from tests.fakes.http_service import FakeService
from tests.fakes.kserve_v2 import FakeKserveV2

INFERENCE_HEADER = "inference-header-content-length"


@pytest.fixture
def kserve() -> Iterator[FakeKserveV2]:
    service = FakeService()
    fake = FakeKserveV2()
    service.include(fake.router)
    service.start()
    fake.service = service
    try:
        yield fake
    finally:
        service.stop()


def _client(kserve: FakeKserveV2, **overrides) -> KserveV2HttpClient:
    settings = {
        "base_url": kserve.service.base_url,
        "model_name": "test-model",
        "model_version": None,
        "timeout": 10.0,
        "headers": {},
        "use_binary_data": True,
    }
    settings.update(overrides)
    return KserveV2HttpClient(**settings)


def _infer_once(client: KserveV2HttpClient, **kwargs) -> dict[str, np.ndarray]:
    return client.infer(
        inputs={"input": np.zeros((1, 3, 8, 8), dtype=np.float32)},
        output_names=["output"],
        **kwargs,
    )


# -- URL construction ----------------------------------------------------


def test_unversioned_urls_omit_the_version_segment(kserve):
    client = _client(kserve)

    client.get_model_metadata()
    _infer_once(client)

    paths = {r.path for r in kserve.service.requests}
    assert paths == {"/v2/models/test-model", "/v2/models/test-model/infer"}


def test_a_model_version_is_included_in_both_urls(kserve):
    client = _client(kserve, model_version="1")

    client.get_model_metadata()
    _infer_once(client)

    paths = {r.path for r in kserve.service.requests}
    assert paths == {
        "/v2/models/test-model/versions/1",
        "/v2/models/test-model/versions/1/infer",
    }


def test_a_base_url_with_a_trailing_slash_does_not_double_up(kserve):
    client = _client(kserve, base_url=f"{kserve.service.base_url}/")

    client.get_model_metadata()

    assert kserve.service.requests[-1].path == "/v2/models/test-model"


# -- metadata ------------------------------------------------------------


def test_model_metadata_is_returned_as_the_typed_model(kserve):
    kserve.platform = "onnxruntime_onnx"
    kserve.inputs = [
        KserveV2ModelTensorSpec(
            name="images", datatype="FP32", shape=[-1, 3, 640, 640]
        ),
        KserveV2ModelTensorSpec(name="sizes", datatype="INT64", shape=[-1, 2]),
    ]

    metadata = _client(kserve).get_model_metadata()

    assert metadata.name == "test-model"
    assert metadata.platform == "onnxruntime_onnx"
    assert [spec.name for spec in metadata.inputs] == ["images", "sizes"]
    assert metadata.inputs[0].shape == [-1, 3, 640, 640]


# -- the two wire formats ------------------------------------------------


@pytest.mark.parametrize(
    ("use_binary_data", "binary_response"),
    [
        pytest.param(False, False, id="json-request-json-response"),
        pytest.param(True, False, id="binary-request-json-response"),
        pytest.param(True, True, id="binary-request-binary-response"),
    ],
)
def test_tensors_round_trip_through_every_wire_format(
    kserve, use_binary_data, binary_response
):
    kserve.binary_response = binary_response
    kserve.infer_handler = lambda payload: {
        "output": np.array([[1.5, 2.5, 3.5]], dtype=np.float32)
    }

    outputs = _infer_once(_client(kserve, use_binary_data=use_binary_data))

    assert outputs["output"].tolist() == [[1.5, 2.5, 3.5]]
    assert outputs["output"].dtype == np.float32


def test_a_binary_request_carries_the_header_length_and_raw_tensor_bytes(kserve):
    _infer_once(_client(kserve, use_binary_data=True))

    request = kserve.service.requests_for("POST", r".*/infer")[-1]
    header_len = int(request.headers[INFERENCE_HEADER])
    # The JSON header is a prefix; the tensor bytes follow it.
    assert 0 < header_len < len(request.body)
    assert b'"binary_data_size"' in request.body[:header_len]


def test_a_json_request_sends_the_tensor_inline(kserve):
    _infer_once(_client(kserve, use_binary_data=False))

    request = kserve.service.requests_for("POST", r".*/infer")[-1]
    assert INFERENCE_HEADER not in request.headers
    tensor = request.json()["inputs"][0]
    assert tensor["name"] == "input"
    assert tensor["shape"] == [1, 3, 8, 8]
    assert tensor["datatype"] == "FP32"
    assert len(tensor["data"]) == 1 * 3 * 8 * 8


@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.float64, np.int32, np.int64, np.uint8, np.bool_],
)
def test_each_supported_dtype_survives_the_round_trip(kserve, dtype):
    expected = np.ones((2, 2), dtype=dtype)
    kserve.binary_response = True
    kserve.infer_handler = lambda payload: {"output": expected}

    outputs = _infer_once(_client(kserve))

    assert outputs["output"].dtype == expected.dtype
    assert outputs["output"].tolist() == expected.tolist()


# -- request shaping -----------------------------------------------------


def test_requested_output_names_are_sent(kserve):
    kserve.infer_handler = lambda payload: {
        name["name"]: np.zeros((1, 1), dtype=np.float32) for name in payload["outputs"]
    }

    outputs = _client(kserve, use_binary_data=False).infer(
        inputs={"input": np.zeros((1, 3, 8, 8), dtype=np.float32)},
        output_names=["labels", "boxes", "scores"],
    )

    assert set(outputs) == {"labels", "boxes", "scores"}


def test_request_parameters_are_forwarded(kserve):
    _infer_once(
        _client(kserve, use_binary_data=False),
        request_parameters={"sequence_id": 7},
    )

    assert kserve.service.requests[-1].json()["parameters"] == {"sequence_id": 7}


def test_custom_headers_are_sent_on_both_calls(kserve):
    client = _client(kserve, headers={"Authorization": "Bearer sk-not-real"})

    client.get_model_metadata()
    _infer_once(client)

    for request in kserve.service.requests:
        assert request.headers["authorization"] == "Bearer sk-not-real"


# -- failure modes -------------------------------------------------------


@pytest.mark.parametrize("status", [400, 404, 500, 503])
def test_error_statuses_are_raised_as_http_errors(kserve, status):
    kserve.fail_status = status

    with pytest.raises(requests.exceptions.HTTPError, match=str(status)):
        _infer_once(_client(kserve))


def test_a_malformed_response_body_is_reported_as_such(kserve):
    from tests.fakes.http_service import Response

    kserve.service.add_route(
        "POST",
        r".*/infer",
        lambda request, match: Response(body="this is not the protocol"),
    )

    with pytest.raises(RuntimeError, match="Invalid inference response"):
        _infer_once(_client(kserve))


def test_an_unknown_output_datatype_is_rejected(kserve):
    from tests.fakes.http_service import Response

    kserve.service.add_route(
        "POST",
        r".*/infer",
        lambda request, match: Response(
            body={
                "outputs": [
                    {
                        "name": "output",
                        "datatype": "COMPLEX128",
                        "shape": [1],
                        "data": [0],
                    }
                ]
            }
        ),
    )

    with pytest.raises(RuntimeError, match="Unsupported KServe v2 output datatype"):
        _infer_once(_client(kserve, use_binary_data=False))


def test_a_truncated_binary_payload_is_detected(kserve):
    """A tensor claiming more bytes than were sent must not be read past."""
    from tests.fakes.http_service import Response

    header = (
        b'{"outputs":[{"name":"output","datatype":"FP32","shape":[1,4],'
        b'"parameters":{"binary_data_size":16}}]}'
    )
    kserve.service.add_route(
        "POST",
        r".*/infer",
        lambda request, match: Response(
            body=header + b"\x00\x00\x00\x00",  # only 4 of the 16 bytes
            headers={"Inference-Header-Content-Length": str(len(header))},
        ),
    )

    with pytest.raises(RuntimeError, match="did not include enough binary output data"):
        _infer_once(_client(kserve, use_binary_data=True))


def test_a_timeout_propagates_to_the_caller(kserve):
    from tests.fakes.http_service import Response

    def slow(request, match):
        import time

        time.sleep(0.6)
        return Response(body={"outputs": []})

    kserve.service.add_route("POST", r".*/infer", slow)

    with pytest.raises(requests.exceptions.Timeout):
        _infer_once(_client(kserve, timeout=0.1))


def test_an_unreachable_server_raises_a_connection_error(kserve):
    base_url = kserve.service.base_url
    kserve.service.stop()

    with pytest.raises(requests.exceptions.ConnectionError):
        _client(kserve, base_url=base_url).get_model_metadata()


def test_close_is_a_no_op_for_transport_parity(kserve):
    """The HTTP client keeps no connection to release, unlike the gRPC one."""
    client = _client(kserve)
    client.close()

    # Still usable afterwards: nothing was actually torn down.
    assert client.get_model_metadata().name == "test-model"
