"""Tests for api_image_request module."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from docling.datamodel.base_models import VlmStopReason
from docling.utils.api_image_request import api_image_request


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
        ):
            mock_resp = MagicMock()
            mock_resp.ok = status_ok
            mock_resp.text = f"""{{
                "id": "test-id",
                "created": 1234567890,
                "choices": [{{
                    "index": 0,
                    "message": {{"role": "assistant", "content": "{content}"}},
                    "finish_reason": "{finish_reason}"
                }}],
                "usage": {{"prompt_tokens": 50, "completion_tokens": 50, "total_tokens": {total_tokens}}}
            }}"""
            return mock_resp

        return _create_mock_response

    @patch("docling.utils.api_image_request.requests.post")
    def test_content_filter_finish_reason(
        self, mock_post, sample_image, mock_response_factory
    ):
        """Test that content_filter finish reason returns CONTENT_FILTERED."""
        mock_post.return_value = mock_response_factory(
            content="Filtered content", finish_reason="content_filter"
        )

        result_text, _tokens, stop_reason = api_image_request(
            image=sample_image,
            prompt="Test prompt",
            url="http://test.api/v1/chat/completions",
        )

        assert result_text == "Filtered content"
        assert stop_reason == VlmStopReason.CONTENT_FILTERED

    @patch("docling.utils.api_image_request.requests.post")
    def test_length_finish_reason(self, mock_post, sample_image, mock_response_factory):
        """Test that length finish reason returns LENGTH."""
        mock_post.return_value = mock_response_factory(
            content="Truncated content", finish_reason="length"
        )

        result_text, _tokens, stop_reason = api_image_request(
            image=sample_image,
            prompt="Test prompt",
            url="http://test.api/v1/chat/completions",
        )

        assert result_text == "Truncated content"
        assert stop_reason == VlmStopReason.LENGTH

    @patch("docling.utils.api_image_request.requests.post")
    def test_stop_finish_reason(self, mock_post, sample_image, mock_response_factory):
        """Test that stop finish reason returns END_OF_SEQUENCE."""
        mock_post.return_value = mock_response_factory(
            content="Normal completion", finish_reason="stop"
        )

        result_text, _tokens, stop_reason = api_image_request(
            image=sample_image,
            prompt="Test prompt",
            url="http://test.api/v1/chat/completions",
        )

        assert result_text == "Normal completion"
        assert stop_reason == VlmStopReason.END_OF_SEQUENCE
