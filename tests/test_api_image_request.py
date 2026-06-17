"""Tests for api_image_request module."""

import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from requests.adapters import HTTPAdapter

from docling.datamodel.base_models import VlmStopReason
from docling.utils.api_image_request import _make_retry_session, api_image_request

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
