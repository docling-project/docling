import numpy as np

from docling.models.inference_engines.common.kserve_v2_grpc import (
    _decode_bytes_tensor,
    _encode_bytes_tensor,
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
        encoded = _encode_bytes_tensor(tensor)

        expected_text = text.encode("utf-8")
        assert encoded[:4] == len(expected_text).to_bytes(4, byteorder="little")
        assert encoded[4:] == expected_text

        decoded = _decode_bytes_tensor(encoded, tensor.shape)

        assert np.array_equal(decoded, np.array([[expected_text]], dtype=object))
