"""Unit tests for UpstageOcrModel — covers response parsing, coordinate scaling,
TOPLEFT origin preservation, ocr_rect offset, confidence filtering, and api_key
fallback behavior."""

import os
from unittest.mock import MagicMock, patch

import pytest
from docling_core.types.doc import BoundingBox

from docling.datamodel.pipeline_options import UpstageOcrOptions
from docling.models.upstage_ocr_model import UpstageOcrModel


SAMPLE_RESPONSE = {
    "apiVersion": "1.1",
    "confidence": 0.99,
    "metadata": {"pages": [{"height": 256, "page": 1, "width": 786}]},
    "mimeType": "multipart/form-data",
    "modelVersion": "ocr-250904",
    "numBilledPages": 1,
    "pages": [
        {
            "confidence": 0.99,
            "height": 256,
            "id": 0,
            "text": "Print the words \nhello, world",
            "width": 786,
            "words": [
                {
                    "boundingBox": {
                        "vertices": [
                            {"x": 65, "y": 52},
                            {"x": 221, "y": 55},
                            {"x": 221, "y": 104},
                            {"x": 64, "y": 101},
                        ]
                    },
                    "confidence": 0.99,
                    "id": 0,
                    "text": "Print",
                },
                {
                    "boundingBox": {
                        "vertices": [
                            {"x": 243, "y": 49},
                            {"x": 341, "y": 52},
                            {"x": 340, "y": 105},
                            {"x": 241, "y": 102},
                        ]
                    },
                    "confidence": 0.99,
                    "id": 1,
                    "text": "the",
                },
                {
                    "boundingBox": {
                        "vertices": [
                            {"x": 368, "y": 52},
                            {"x": 553, "y": 51},
                            {"x": 553, "y": 105},
                            {"x": 368, "y": 105},
                        ]
                    },
                    "confidence": 0.98,
                    "id": 2,
                    "text": "words",
                },
                {
                    "boundingBox": {
                        "vertices": [
                            {"x": 214, "y": 131},
                            {"x": 470, "y": 149},
                            {"x": 467, "y": 206},
                            {"x": 210, "y": 188},
                        ]
                    },
                    "confidence": 0.99,
                    "id": 3,
                    "text": "hello,",
                },
                {
                    "boundingBox": {
                        "vertices": [
                            {"x": 527, "y": 145},
                            {"x": 748, "y": 143},
                            {"x": 749, "y": 192},
                            {"x": 527, "y": 194},
                        ]
                    },
                    "confidence": 0.98,
                    "id": 4,
                    "text": "world",
                },
            ],
        }
    ],
}


@pytest.fixture
def upstage_model(monkeypatch):
    monkeypatch.setenv("UPSTAGE_API_KEY", "test-key")
    options = UpstageOcrOptions()
    with patch(
        "docling.models.base_ocr_model.BaseOcrModel.__init__", return_value=None
    ):
        model = UpstageOcrModel.__new__(UpstageOcrModel)
        model.enabled = True
        model.options = options
        model.scale = 1
        model.api_key = "test-key"
        model.api_endpoint = options.api_endpoint
        model.model = options.model
        model.timeout = options.timeout
    return model


def _ocr_rect(l=0.0, t=0.0, r=786.0, b=256.0):
    return BoundingBox(l=l, t=t, r=r, b=b)


def test_extract_cells_full_response(upstage_model):
    cells = upstage_model._extract_cells(SAMPLE_RESPONSE, _ocr_rect(), (786, 256))
    assert [c.text for c in cells] == ["Print", "the", "words", "hello,", "world"]
    assert all(c.from_ocr for c in cells)
    assert all(0.0 < c.confidence <= 1.0 for c in cells)


def test_extract_cells_axis_aligned_bbox(upstage_model):
    cells = upstage_model._extract_cells(SAMPLE_RESPONSE, _ocr_rect(), (786, 256))
    # First word "Print" — vertices x in [64, 221], y in [52, 104]
    rect = cells[0].rect
    xs = [rect.r_x0, rect.r_x1, rect.r_x2, rect.r_x3]
    ys = [rect.r_y0, rect.r_y1, rect.r_y2, rect.r_y3]
    assert min(xs) == pytest.approx(64.0)
    assert max(xs) == pytest.approx(221.0)
    assert min(ys) == pytest.approx(52.0)
    assert max(ys) == pytest.approx(104.0)


def test_topleft_origin_y_increases_downward(upstage_model):
    """Upstage 의 vertex y 가 위→아래 증가 (TOPLEFT) 라는 가정이 유지되는지."""
    cells = upstage_model._extract_cells(SAMPLE_RESPONSE, _ocr_rect(), (786, 256))
    by_text = {c.text: c for c in cells}
    # "Print" 는 첫 줄 (y≈52), "hello," 는 둘째 줄 (y≈131) — 둘째 줄의 y 가 더 커야 한다.
    print_y_min = min(
        by_text["Print"].rect.r_y0, by_text["Print"].rect.r_y1,
        by_text["Print"].rect.r_y2, by_text["Print"].rect.r_y3,
    )
    hello_y_min = min(
        by_text["hello,"].rect.r_y0, by_text["hello,"].rect.r_y1,
        by_text["hello,"].rect.r_y2, by_text["hello,"].rect.r_y3,
    )
    assert hello_y_min > print_y_min


def test_scale_defensive_when_response_is_half_size(upstage_model):
    """응답이 절반 차원 (Upstage 가 내부 리사이즈 했다고 가정) 이면 좌표는 2배 스케일."""
    resp = {
        "pages": [
            {
                "width": 393,
                "height": 128,
                "words": [
                    {
                        "boundingBox": {
                            "vertices": [
                                {"x": 32, "y": 26},
                                {"x": 110, "y": 26},
                                {"x": 110, "y": 52},
                                {"x": 32, "y": 52},
                            ]
                        },
                        "confidence": 0.99,
                        "text": "Print",
                    }
                ],
            }
        ]
    }
    cells = upstage_model._extract_cells(resp, _ocr_rect(), (786, 256))
    assert len(cells) == 1
    rect = cells[0].rect
    xs = [rect.r_x0, rect.r_x1, rect.r_x2, rect.r_x3]
    ys = [rect.r_y0, rect.r_y1, rect.r_y2, rect.r_y3]
    # 32 * (786/393) = 64, 110 * 2 = 220 ; 26 * (256/128) = 52, 52 * 2 = 104
    assert min(xs) == pytest.approx(64.0)
    assert max(xs) == pytest.approx(220.0)
    assert min(ys) == pytest.approx(52.0)
    assert max(ys) == pytest.approx(104.0)


def test_ocr_rect_offset_applied(upstage_model):
    """부분 crop 의 경우 vertex 좌표에 ocr_rect 의 left/top 오프셋이 더해져야 한다."""
    cells = upstage_model._extract_cells(
        SAMPLE_RESPONSE, _ocr_rect(l=100.0, t=200.0, r=886.0, b=456.0), (786, 256)
    )
    rect = cells[0].rect  # "Print"
    xs = [rect.r_x0, rect.r_x1, rect.r_x2, rect.r_x3]
    ys = [rect.r_y0, rect.r_y1, rect.r_y2, rect.r_y3]
    assert min(xs) == pytest.approx(64.0 + 100.0)
    assert max(xs) == pytest.approx(221.0 + 100.0)
    assert min(ys) == pytest.approx(52.0 + 200.0)
    assert max(ys) == pytest.approx(104.0 + 200.0)


def test_text_score_filters_low_confidence(upstage_model):
    upstage_model.options = UpstageOcrOptions(text_score=0.985)
    cells = upstage_model._extract_cells(SAMPLE_RESPONSE, _ocr_rect(), (786, 256))
    # confidence < 0.985 인 단어 ("words"=0.98, "world"=0.98) 는 필터링되어야 한다.
    texts = {c.text for c in cells}
    assert "words" not in texts
    assert "world" not in texts
    assert {"Print", "the", "hello,"} <= texts


def test_get_options_type_returns_upstage_options():
    assert UpstageOcrModel.get_options_type() is UpstageOcrOptions


def test_init_raises_when_no_api_key(monkeypatch):
    monkeypatch.delenv("UPSTAGE_API_KEY", raising=False)
    options = UpstageOcrOptions(api_key="")
    accel = MagicMock()
    with pytest.raises(ValueError, match="api_key"):
        UpstageOcrModel(
            enabled=True,
            artifacts_path=None,
            options=options,
            accelerator_options=accel,
        )


def test_init_uses_env_api_key_when_options_empty(monkeypatch):
    monkeypatch.setenv("UPSTAGE_API_KEY", "env-secret")
    options = UpstageOcrOptions(api_key="")
    accel = MagicMock()
    accel.num_threads = 4
    accel.device = "cpu"
    model = UpstageOcrModel(
        enabled=True,
        artifacts_path=None,
        options=options,
        accelerator_options=accel,
    )
    assert model.api_key == "env-secret"


def test_init_options_api_key_takes_precedence_over_env(monkeypatch):
    monkeypatch.setenv("UPSTAGE_API_KEY", "env-secret")
    options = UpstageOcrOptions(api_key="options-secret")
    accel = MagicMock()
    accel.num_threads = 4
    accel.device = "cpu"
    model = UpstageOcrModel(
        enabled=True,
        artifacts_path=None,
        options=options,
        accelerator_options=accel,
    )
    assert model.api_key == "options-secret"


def test_call_upstage_uses_bearer_auth(upstage_model):
    fake_response = MagicMock()
    fake_response.ok = True
    fake_response.status_code = 200
    fake_response.json.return_value = SAMPLE_RESPONSE

    captured = {}

    def fake_post(url, headers=None, files=None, data=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["files"] = files
        captured["data"] = data
        captured["timeout"] = timeout
        return fake_response

    from PIL import Image
    img = Image.new("RGB", (786, 256), "white")

    with patch("docling.models.upstage_ocr_model.requests.post", side_effect=fake_post):
        result = upstage_model._call_upstage(img)

    assert result == SAMPLE_RESPONSE
    assert captured["url"] == upstage_model.api_endpoint
    assert captured["headers"]["Authorization"] == f"Bearer {upstage_model.api_key}"
    assert "document" in captured["files"]
    assert captured["data"]["model"] == upstage_model.model
    assert captured["timeout"] == upstage_model.timeout


def test_call_upstage_raises_on_http_error(upstage_model):
    fake_response = MagicMock()
    fake_response.ok = False
    fake_response.status_code = 502
    fake_response.text = "Bad Gateway"

    from PIL import Image
    img = Image.new("RGB", (10, 10), "white")

    with patch("docling.models.upstage_ocr_model.requests.post", return_value=fake_response):
        with pytest.raises(RuntimeError, match="Upstage OCR HTTP 502"):
            upstage_model._call_upstage(img)
