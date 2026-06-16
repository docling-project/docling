import numpy as np
import requests

from docling.models.inference_engines.common.kserve_v2_http import (
    _INFERENCE_HEADER_CONTENT_LENGTH,
    KserveV2HttpClient,
)
from docling.models.inference_engines.common.kserve_v2_utils import (
    decode_bytes_tensor,
    encode_bytes_tensor,
)


def test_bytes_tensor_binary_encoding_round_trip() -> None:
    texts = [
        "ch",
        "ch_doc",
        "en",
        "arabic",
        "chinese_cht",
        "cyrillic",
        "devanagari",
        "japan",
        "korean",
        "ka",
        "latin",
        "ta",
        "te",
        "eslav",
        "th",
        "el",
    ]

    for text in texts:
        tensor = np.array([[text]], dtype=object)

        # Encode the text
        encoded = encode_bytes_tensor(tensor)

        expected_text = text.encode("utf-8")
        assert encoded[:4] == len(expected_text).to_bytes(4, byteorder="little")
        assert encoded[4:] == expected_text

        decoded = decode_bytes_tensor(encoded, tensor.shape)

        assert np.array_equal(decoded, np.array([[expected_text]], dtype=object))


def test_http_binary_request_serialization() -> None:
    captured: dict[str, object] = {}
    client = KserveV2HttpClient(
        base_url="",  # The URL is not used.
        model_name="rapidocr",
        model_version="1",
        timeout=1.0,
        headers={"Authorization": "Bearer token"},
        use_binary_data=True,
    )

    response = requests.Response()
    response.status_code = 200
    response._content = b'{"outputs":[]}'
    response.url = client.infer_url
    response.headers["Content-Type"] = "application/json"

    def fake_request(
        url: str, method: str = "GET", **kwargs: object
    ) -> requests.Response:
        captured["url"] = url
        captured["method"] = method
        captured["kwargs"] = kwargs
        return response

    object.__setattr__(client, "_execute_http_request", fake_request)

    client.infer(
        inputs={
            "lang_type": np.array([["en"]], dtype=object),
            "image": np.array([[[[1, 2, 3]]]], dtype=np.uint8),
        },
        output_names=["txts"],
    )

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    headers = kwargs["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer token"
    assert headers["Content-Type"] == "application/octet-stream"

    body = kwargs["data"]
    assert isinstance(body, bytes)
    header_len = int(headers[_INFERENCE_HEADER_CONTENT_LENGTH])
    request_header = body[:header_len].decode("utf-8")
    request_payload = body[header_len:]

    assert '"binary_data_size":6' in request_header
    assert '"binary_data_size":3' in request_header
    assert '"name":"txts","parameters":{"binary_data":true}' in request_header
    assert request_payload.endswith(b"\x01\x02\x03")


def test_http_binary_response_decoding() -> None:
    client = KserveV2HttpClient(
        base_url="",  # The URL is not used.
        model_name="rapidocr",
        model_version="1",
        timeout=1.0,
        headers={},
        use_binary_data=True,
    )

    txts = encode_bytes_tensor(np.array([[b"hello"]], dtype=object))
    scores = np.array([[0.5]], dtype=np.float32).tobytes()
    response_header = (
        b'{"outputs":['
        b'{"name":"txts","datatype":"BYTES","shape":[1,1],"parameters":{"binary_data_size":9}},'
        b'{"name":"scores","datatype":"FP32","shape":[1,1],"parameters":{"binary_data_size":4}}'
        b"]}"
    )

    response = requests.Response()
    response.status_code = 200
    response.url = client.infer_url
    response.headers[_INFERENCE_HEADER_CONTENT_LENGTH] = str(len(response_header))
    response._content = response_header + txts + scores

    def fake_request(
        url: str, method: str = "GET", **kwargs: object
    ) -> requests.Response:
        return response

    object.__setattr__(client, "_execute_http_request", fake_request)

    outputs = client.infer(
        inputs={"image": np.array([[[[1, 2, 3]]]], dtype=np.uint8)},
        output_names=["txts", "scores"],
    )

    assert outputs["txts"].shape == (1, 1)
    assert outputs["txts"][0, 0] == b"hello"
    assert outputs["scores"].shape == (1, 1)
    assert outputs["scores"][0, 0] == np.float32(0.5)
