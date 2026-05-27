import json
from unittest.mock import patch

import pytest

from docling.prompts.prompt_manager import LLMApiError, PromptManager


@pytest.mark.unit
def test_prompt_precheck_blocks_before_api_call():
    pm = PromptManager(
        custom_api_configs={
            "toc_extraction": {
                "provider": "custom",
                "api_base_url": "http://dummy.local/v1/chat/completions",
                "model": "dummy-model",
                "precheck_enabled": True,
                "max_context_tokens": 128000,
                "completion_reserved_tokens": 12000,
            }
        }
    )

    with patch.object(
        PromptManager,
        "_rough_token_count",
        return_value=169000,
    ), patch("docling.prompts.prompt_manager.requests.post") as post_mock:
        with pytest.raises(LLMApiError) as exc_info:
            pm.call_ai_model(
                category="toc_extraction",
                prompt_type="korean_document",
                raise_on_error=True,
                raw_text="테스트 본문",
            )

    post_mock.assert_not_called()
    payload = json.loads(exc_info.value.raw_error_message)
    assert payload["object"] == "error"
    assert payload["type"] == "BadRequestError"
    assert payload["param"] == "prompt"
    assert payload["code"] == 400
    assert "프롬프트 입력 토큰 (169000) 초과" in payload["message"]
    assert "(128000 - reserved 12000)." in payload["message"]

