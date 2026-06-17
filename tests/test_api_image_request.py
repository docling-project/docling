"""Tests for api_image_request module."""

import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from requests.adapters import HTTPAdapter

from docling.datamodel.base_models import (
    ApiImageRequestResult,
    ApiImageStreamingRequestResult,
    VlmStopReason,
)
from docling.models.utils.generation_utils import GenerationStopper
from docling.utils.api_image_request import (
    _make_retry_session,
    api_image_request,
    api_image_request_streaming,
)

pytestmark = pytest.mark.cross_platform


class TestApiImageRequest:
    """Test cases for api_image_request function."""

    @pytest.fixture
    def sample_image(self):
        """Create a simple test image."""
        return Image.new("RGB", (100, 100), color="red")

    @pytest.fixture
    def mock_response_factory(self):
        """Factory to create mock API responses."""

        def _create_mock_response(
            content="Test response",
            finish_reason="stop",
            total_tokens=100,
            status_ok=True,
            message=None,
        ):
            mock_resp = MagicMock()
            mock_resp.ok = status_ok
            mock_resp.text = json.dumps(
                {
                    "id": "test-id",
                    "created": 1234567890,
                    "choices": [
                        {
                            "index": 0,
                            "message": message
                            or {"role": "assistant", "content": content},
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 50,
                        "total_tokens": total_tokens,
                    },
                }
            )
            return mock_resp

        return _create_mock_response

    @patch("docling.utils.api_image_request._make_retry_session")
    def test_content_filter_finish_reason(
        self, mock_session_factory, sample_image, mock_response_factory
    ):
        """Test that content_filter finish reason returns CONTENT_FILTERED."""
        mock_session_factory.return_value.__enter__.return_value.post.return_value = (
            mock_response_factory(
                content="Filtered content", finish_reason="content_filter"
            )
        )

        result_text, _tokens, stop_reason = api_image_request(
            image=sample_image,
            prompt="Test prompt",
            url="http://test.api/v1/chat/completions",
        )

        assert result_text == "Filtered content"
        assert stop_reason == VlmStopReason.CONTENT_FILTERED

    @patch("docling.utils.api_image_request._make_retry_session")
    def test_length_finish_reason(
        self, mock_session_factory, sample_image, mock_response_factory
    ):
        """Test that length finish reason returns LENGTH."""
        mock_session_factory.return_value.__enter__.return_value.post.return_value = (
            mock_response_factory(content="Truncated content", finish_reason="length")
        )

        result_text, _tokens, stop_reason = api_image_request(
            image=sample_image,
            prompt="Test prompt",
            url="http://test.api/v1/chat/completions",
        )

        assert result_text == "Truncated content"
        assert stop_reason == VlmStopReason.LENGTH

    @patch("docling.utils.api_image_request._make_retry_session")
    def test_stop_finish_reason(
        self, mock_session_factory, sample_image, mock_response_factory
    ):
        """Test that stop finish reason returns END_OF_SEQUENCE."""
        mock_session_factory.return_value.__enter__.return_value.post.return_value = (
            mock_response_factory(content="Normal completion", finish_reason="stop")
        )

        result_text, _tokens, stop_reason = api_image_request(
            image=sample_image,
            prompt="Test prompt",
            url="http://test.api/v1/chat/completions",
        )

        assert result_text == "Normal completion"
        assert stop_reason == VlmStopReason.END_OF_SEQUENCE

    @patch("docling.utils.api_image_request._make_retry_session")
    def test_tool_calls_response(
        self, mock_session_factory, sample_image, mock_response_factory
    ):
        """Test that tool calling responses are converted into generated text."""
        mock_session_factory.return_value.__enter__.return_value.post.return_value = (
            mock_response_factory(
                message={
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "markdown_no_bbox",
                                "arguments": json.dumps(
                                    [
                                        {"text": "Extracted text"},
                                        {"text": "Second block"},
                                    ]
                                ),
                            }
                        }
                    ],
                }
            )
        )

        result_text, tokens, stop_reason = api_image_request(
            image=sample_image,
            prompt="Test prompt",
            url="http://test.api/v1/chat/completions",
        )

        assert result_text == "Extracted text\nSecond block"
        assert tokens == 100
        assert stop_reason == VlmStopReason.END_OF_SEQUENCE

    @patch("docling.utils.api_image_request._make_retry_session")
    def test_exposes_full_usage_payload_while_preserving_tuple_unpacking(
        self, mock_session_factory, sample_image, mock_response_factory
    ):
        """Test that callers can access raw usage without breaking tuple unpacking."""
        mock_session_factory.return_value.__enter__.return_value.post.return_value = (
            mock_response_factory(total_tokens=123)
        )

        response = api_image_request(
            image=sample_image,
            prompt="Test prompt",
            url="http://test.api/v1/chat/completions",
        )
        result_text, tokens, stop_reason = response

        assert result_text == "Test response"
        assert response[0] == "Test response"
        assert len(response) == 3
        assert tokens == 123
        assert stop_reason == VlmStopReason.END_OF_SEQUENCE
        assert response.usage == {
            "prompt_tokens": 50,
            "completion_tokens": 50,
            "total_tokens": 123,
        }

    @patch("docling.utils.api_image_request._make_retry_session")
    def test_exposes_custom_usage_payload_key(self, mock_session_factory, sample_image):
        """Test that non-OpenAI usage payload keys can be preserved."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.text = json.dumps(
            {
                "id": "test-id",
                "created": 1234567890,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Test response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
                "providerUsage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "cache_read_tokens": 5,
                },
            }
        )
        mock_session_factory.return_value.__enter__.return_value.post.return_value = (
            mock_resp
        )

        response = api_image_request(
            image=sample_image,
            prompt="Test prompt",
            url="http://test.api/v1/chat/completions",
            usage_response_key="providerUsage",
        )

        assert response.usage == {
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_read_tokens": 5,
        }
        assert response.num_tokens == 30

    @patch("docling.utils.api_image_request._make_retry_session")
    def test_token_extract_key_alias_selects_usage_payload(
        self, mock_session_factory, sample_image
    ):
        """Test the plugin-style token_extract_key alias for usage extraction."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.text = json.dumps(
            {
                "id": "test-id",
                "created": 1234567890,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Test response"},
                        "finish_reason": "stop",
                    }
                ],
                "providerUsage": {"input_tokens": 1, "output_tokens": 2},
            }
        )
        mock_session_factory.return_value.__enter__.return_value.post.return_value = (
            mock_resp
        )

        response = api_image_request(
            image=sample_image,
            prompt="Test prompt",
            url="http://test.api/v1/chat/completions",
            token_extract_key="providerUsage",
        )

        assert response.usage == {"input_tokens": 1, "output_tokens": 2}

    def test_request_result_preserves_tuple_compatibility(self):
        """Test that result objects compare like legacy tuples when needed."""
        response = ApiImageRequestResult(
            text="text",
            num_tokens=5,
            stop_reason=VlmStopReason.END_OF_SEQUENCE,
            usage={"total_tokens": 5},
        )

        assert tuple(response) == ("text", 5, VlmStopReason.END_OF_SEQUENCE)
        assert response[1] == 5
        assert len(response) == 3
        assert response == ("text", 5, VlmStopReason.END_OF_SEQUENCE)
        assert response == ApiImageRequestResult(
            "text", 5, VlmStopReason.END_OF_SEQUENCE, {"total_tokens": 5}
        )
        assert response != "text"

    def test_streaming_result_preserves_tuple_compatibility(self):
        """Test that streaming result objects keep the legacy two-value API."""
        response = ApiImageStreamingRequestResult(
            text="text",
            num_tokens=5,
            usage={"total_tokens": 5},
        )

        assert tuple(response) == ("text", 5)
        assert response[1] == 5
        assert len(response) == 2
        assert response == ("text", 5)
        assert response == ApiImageStreamingRequestResult(
            "text", 5, {"total_tokens": 5}
        )
        assert response != "text"

    @patch("docling.utils.api_image_request._make_retry_session")
    def test_nested_usage_key_supplies_token_count_when_openai_usage_is_absent(
        self, mock_session_factory, sample_image
    ):
        """Test dotted usage paths for providers that do not return OpenAI usage."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.text = json.dumps(
            {
                "id": "test-id",
                "created": 1234567890,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Test response"},
                        "finish_reason": "stop",
                    }
                ],
                "meta": {"usage": {"total_tokens": 44, "cache_read_tokens": 5}},
            }
        )
        mock_session_factory.return_value.__enter__.return_value.post.return_value = (
            mock_resp
        )

        response = api_image_request(
            image=sample_image,
            prompt="Test prompt",
            url="http://test.api/v1/chat/completions",
            usage_response_key="meta.usage",
        )

        assert response.num_tokens == 44
        assert response.usage == {"total_tokens": 44, "cache_read_tokens": 5}

    @patch("docling.utils.api_image_request._make_retry_session")
    def test_invalid_json_response_logs_preview_and_returns_unspecified(
        self, mock_session_factory, sample_image, caplog
    ):
        """Test that malformed provider responses include useful diagnostics."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.text = "not-json" * 100
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/plain"}
        mock_session_factory.return_value.__enter__.return_value.post.return_value = (
            mock_resp
        )

        result_text, tokens, stop_reason = api_image_request(
            image=sample_image,
            prompt="Test prompt",
            url="http://test.api/v1/chat/completions",
        )

        assert result_text == ""
        assert tokens == 0
        assert stop_reason == VlmStopReason.UNSPECIFIED
        assert "API response body was not JSON" in caplog.text
        assert "not-json" in caplog.text

    @patch("docling.utils.api_image_request._make_retry_session")
    def test_empty_api_response_logs_status_and_returns_unspecified(
        self, mock_session_factory, sample_image, caplog
    ):
        """Test that empty provider responses include useful diagnostics."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.text = ""
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/plain"}
        mock_session_factory.return_value.__enter__.return_value.post.return_value = (
            mock_resp
        )

        result_text, tokens, stop_reason = api_image_request(
            image=sample_image,
            prompt="Test prompt",
            url="http://test.api/v1/chat/completions",
        )

        assert result_text == ""
        assert tokens == 0
        assert stop_reason == VlmStopReason.UNSPECIFIED
        assert "API response body was empty" in caplog.text
        assert "status=200" in caplog.text

    def test_retry_session_retries_transient_api_errors(self):
        """Test that remote API calls retry common transient failures."""
        with _make_retry_session() as session:
            adapter = session.get_adapter("https://")

            assert isinstance(adapter, HTTPAdapter)

            retry_config = adapter.max_retries
            assert retry_config.total == 5
            assert retry_config.connect == 5
            assert retry_config.read == 0
            assert retry_config.status == 5
            assert retry_config.backoff_factor == 0.1
            assert set(retry_config.status_forcelist) == {429, 500, 502, 503, 504}
            assert retry_config.allowed_methods == {"POST"}
            assert retry_config.respect_retry_after_header is True

    @patch("docling.utils.api_image_request._make_retry_session")
    def test_streaming_response_preserves_usage_payload(
        self, mock_session_factory, sample_image
    ):
        """Test usage extraction from OpenAI-compatible SSE chunks."""

        class _StreamingResponse:
            ok = True
            text = ""

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self):
                return None

            def iter_lines(self, decode_unicode=True):
                yield ""
                yield "event: keepalive"
                yield "data: not-json"
                yield 'data: {"choices": [{"delta": {"content": "hel"}}]}'
                yield (
                    'data: {"choices": [{"delta": {"content": "lo"}}], '
                    '"usage": {"total_tokens": 8}}'
                )
                yield "data: [DONE]"

        mock_session_factory.return_value.__enter__.return_value.post.return_value = (
            _StreamingResponse()
        )

        response = api_image_request_streaming(
            image=sample_image,
            prompt="Test prompt",
            url="http://test.api/v1/chat/completions",
        )

        assert response.text == "hello"
        assert response.num_tokens == 8
        assert response.usage == {"total_tokens": 8}

    @patch("docling.utils.api_image_request._make_retry_session")
    def test_streaming_response_returns_usage_when_stopper_triggers(
        self, mock_session_factory, sample_image
    ):
        """Test early streaming abort still returns usage seen so far."""

        class _StopOnDone(GenerationStopper):
            def lookback_tokens(self):
                return 4

            def should_stop(self, s: str) -> bool:
                return "done" in s

        class _StreamingResponse:
            ok = True
            text = ""

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self):
                return None

            def iter_lines(self, decode_unicode=True):
                yield (
                    'data: {"choices": [{"delta": {"content": "done"}}], '
                    '"meta": {"usage": {"total_tokens": 3}}}'
                )
                yield 'data: {"choices": [{"delta": {"content": " ignored"}}]}'

        mock_session_factory.return_value.__enter__.return_value.post.return_value = (
            _StreamingResponse()
        )

        response = api_image_request_streaming(
            image=sample_image,
            prompt="Test prompt",
            url="http://test.api/v1/chat/completions",
            generation_stoppers=[_StopOnDone()],
            usage_response_key="meta.usage",
        )

        assert response.text == "done"
        assert response.num_tokens == 3
        assert response.usage == {"total_tokens": 3}
