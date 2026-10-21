"""MiniMax selections use the Dot's existing gateway, limits and metering contract."""

from __future__ import annotations

import json

import httpx
import pytest
from fakes.dot_config import runtime_config_body
from openai import AsyncOpenAI

from nanobot.dots.projection import project
from nanobot.dots.protocol import parse_runtime_config
from nanobot.dots.provider import OpenRouterProviders
from nanobot.providers.base import ModelLimits


@pytest.mark.parametrize("model", [
    "minimax/minimax-m3.1-flash:free",
    "minimax/minimax-m3",
    "minimax/minimax-m2.7",
    "custom/model",
    "z-ai/glm-5.3-flash",
])
@pytest.mark.parametrize("base_url", [None, "https://gateway.example.test/api/v1"])
@pytest.mark.parametrize("cost", [0, 0.025, None])
async def test_selected_model_reaches_the_sdk_http_boundary_without_changing_history(
    model: str, base_url: str | None, cost: float | None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Read this picture."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]},
        {"role": "assistant", "content": None, "reasoning_content": "Read the file first.",
         "tool_calls": [{"id": "call_1", "type": "function", "function": {
             "name": "read_file", "arguments": json.dumps({"path": "note.txt"}),
         }}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "The note."},
    ]
    tools = [{"type": "function", "function": {
        "name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    }}]

    def serve(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            assert request.url.path == f"/api/v1/models/{model}/endpoints"
            # Deliberately different from direct API metadata: the gateway owns these limits.
            return httpx.Response(200, json={"data": {"endpoints": [{
                "tag": "test-route", "context_length": 123456,
                "max_completion_tokens": 4567, "status": 0,
            }]}})
        assert request.url.path == "/api/v1/chat/completions"
        chunk = {
            "id": "test-completion", "object": "chat.completion.chunk", "created": 0, "model": model,
            "choices": [{"index": 0, "delta": {"content": "Done."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 2, "total_tokens": 32,
                      **({"cost": cost} if cost is not None else {})},
        }
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

    # Keep the real SDK, replacing only the HTTP transport it constructs.
    def client_with_transport(**kwargs):
        kwargs["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(serve))
        return AsyncOpenAI(**kwargs)

    monkeypatch.setattr("nanobot.providers.openai_compat_provider.AsyncOpenAI", client_with_transport)
    config = parse_runtime_config(runtime_config_body(model={"provider": "openrouter", "id": model}))
    settings = project(config, workspace="/workspace", openrouter_base_url=base_url)
    provider = OpenRouterProviders().current(settings, "test-key")
    try:
        limits = await provider.model_limits(settings.model_id)
        assert limits == ModelLimits(context_tokens=123456, answer_tokens=4567)
        response = await provider.chat_stream(messages, tools=tools, max_tokens=limits.answer_tokens)
        assert response.content == "Done."
        assert response.cost_usd == cost
        assert response.finish_reason == "stop"
        assert len(requests) == 2
        sent = json.loads(requests[-1].content)
        assert sent["model"] == model
        assert sent["messages"] == messages
        assert sent["tools"] == tools
        assert sent["max_tokens"] == 4567
        assert sent["provider"] == {"only": ["test-route"]}
        assert sent["stream"] is True
        assert sent["stream_options"] == {"include_usage": True}
        assert sent["tool_choice"] == "auto"
        assert not {"reasoning", "reasoning_effort", "thinking", "reasoning_split", "service_tier"} & sent.keys()
        assert requests[-1].headers["authorization"] == "Bearer test-key"
        assert ("HTTP-Referer" in requests[-1].headers) is (base_url is None)
        assert requests[-1].url.host == ("openrouter.ai" if base_url is None else "gateway.example.test")
    finally:
        if provider._client is not None:
            await provider._client.close()
