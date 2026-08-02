import json
from unittest.mock import Mock

import httpx
from pydantic import BaseModel

from src.llm.models import ChatOpenAICompatible
from src.utils import llm as llm_utils


class AnalysisResult(BaseModel):
    signal: str


def _json_capable_model_info():
    model_info = Mock()
    model_info.has_json_mode.return_value = True
    return model_info


def test_call_llm_uses_json_schema_structured_output(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:1234/v1")
    expected = AnalysisResult(signal="bullish")
    structured_llm = Mock()
    structured_llm.invoke.return_value = expected
    base_llm = Mock()
    base_llm.with_structured_output.return_value = structured_llm

    monkeypatch.setattr(llm_utils, "get_model_info", Mock(return_value=_json_capable_model_info()))
    monkeypatch.setattr(llm_utils, "get_model", Mock(return_value=base_llm))

    result = llm_utils.call_llm("Analyze AAPL", AnalysisResult, max_retries=1)

    base_llm.with_structured_output.assert_called_once_with(
        AnalysisResult,
        method="json_schema",
    )
    structured_llm.invoke.assert_called_once_with("Analyze AAPL")
    assert result == expected


def test_call_llm_preserves_json_mode_for_hosted_providers(monkeypatch):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    expected = AnalysisResult(signal="neutral")
    structured_llm = Mock()
    structured_llm.invoke.return_value = expected
    base_llm = Mock()
    base_llm.with_structured_output.return_value = structured_llm

    monkeypatch.setattr(llm_utils, "get_model_info", Mock(return_value=_json_capable_model_info()))
    monkeypatch.setattr(llm_utils, "get_model", Mock(return_value=base_llm))

    result = llm_utils.call_llm("Analyze AAPL", AnalysisResult, max_retries=1)

    base_llm.with_structured_output.assert_called_once_with(
        AnalysisResult,
        method="json_mode",
    )
    assert result == expected


def test_json_schema_method_emits_lm_studio_compatible_response_format(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:1234/v1")
    request_body = {}

    def handle_request(request):
        request_body.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen3.5-122b-a10b/ud",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "parsed": None,
                            "refusal": None,
                            "reasoning_content": '{"signal":"bullish"}',
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    local_llm = ChatOpenAICompatible(
        model="qwen3.5-122b-a10b/ud",
        api_key="lm-studio",
        base_url="http://lm-studio.test/v1",
        http_client=http_client,
    )
    monkeypatch.setattr(llm_utils, "get_model_info", Mock(return_value=_json_capable_model_info()))
    monkeypatch.setattr(llm_utils, "get_model", Mock(return_value=local_llm))

    result = llm_utils.call_llm("Analyze AAPL", AnalysisResult, max_retries=1)

    assert request_body["response_format"]["type"] == "json_schema"
    assert request_body["response_format"]["json_schema"]["schema"]["properties"] == {"signal": {"title": "Signal", "type": "string"}}
    assert result == AnalysisResult(signal="bullish")
