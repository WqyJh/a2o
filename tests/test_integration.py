"""Integration tests: proxy server with mock upstream."""

import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from a2o.config import Config
from a2o.server import create_app, AnthropicMessageHandler


# ---------------------------------------------------------------------------
# Mock upstream responses
# ---------------------------------------------------------------------------

MOCK_OPENAI_RESPONSE = {
    "id": "chatcmpl-mock123",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "42 is the answer.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    },
}

MOCK_OPENAI_TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-tool456",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Tokyo"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 10},
}


# ---------------------------------------------------------------------------
# Mock upstream server fixture
# ---------------------------------------------------------------------------


def _create_mock_upstream() -> FastAPI:
    """Create a mock OpenAI-compatible upstream FastAPI app."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body = await request.json()

        # Check if this is a tool call scenario
        messages = body.get("messages", [])
        last_msg = messages[-1] if messages else {}
        is_tool_result = last_msg.get("role") == "tool" or (
            isinstance(last_msg.get("content"), list)
            and any(
                b.get("type") == "tool_result"
                for b in last_msg["content"]
                if isinstance(b, dict)
            )
        )

        if is_tool_result or any(m.get("role") == "tool" for m in messages):
            return JSONResponse(
                {
                    "id": "chatcmpl-after-tool",
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "The weather in Tokyo is sunny.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 10},
                }
            )

        # Check for tools in request
        tools = body.get("tools")
        if tools:
            return JSONResponse(MOCK_OPENAI_TOOL_CALL_RESPONSE)

        # Default: return text response, but preserve the model name
        resp = json.loads(json.dumps(MOCK_OPENAI_RESPONSE))
        resp["model"] = body.get("model", "gpt-4o")
        return JSONResponse(resp)

    return app


def _create_stream_upstream() -> FastAPI:
    """Create a mock upstream that returns SSE streaming chunks."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def stream_chat(request: Request) -> Response:
        chunks = [
            'data: {"id":"s-123","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
            'data: {"id":"s-123","object":"chat.completion.chunk","model":"","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
            'data: {"id":"s-123","object":"chat.completion.chunk","model":"","choices":[{"index":0,"delta":{"content":" world!"},"finish_reason":null}]}\n\n',
            'data: {"id":"s-123","object":"chat.completion.chunk","model":"","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3}}\n\n',
            "data: [DONE]\n\n",
        ]
        body = "".join(chunks)
        return Response(content=body, media_type="text/event-stream")

    return app


# ---------------------------------------------------------------------------
# Helper to create proxy client that routes upstream to mock ASGI app
# ---------------------------------------------------------------------------


def _make_proxy_client(
    upstream_app: FastAPI, config_overrides: dict | None = None,
) -> httpx.AsyncClient:
    """Create a test client for the proxy, wiring upstream to a mock ASGI app.

    The proxy's httpx client is replaced with one that routes to the mock
    upstream via ASGITransport, so no real network is involved.
    """
    config = Config(
        openai_base_url="http://mock-upstream/v1/chat/completions",
        default_model="test-model",
        **(config_overrides or {}),
    )
    proxy_app = create_app(config)

    # Replace the handler's HTTP client with one pointing at the mock upstream
    handler: AnthropicMessageHandler = proxy_app.state.handler
    handler._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=upstream_app),
        base_url="http://mock-upstream",
    )

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_app),
        base_url="http://test-proxy",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProxyNonStreaming:
    @pytest.fixture
    def proxy_app(self):
        return _make_proxy_client(_create_mock_upstream())

    async def test_simple_text_request(self, proxy_app):
        async with proxy_app as client:
            resp = await client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "What is 6*7?"}],
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["type"] == "message"
            assert body["role"] == "assistant"
            text_blocks = [c for c in body["content"] if c["type"] == "text"]
            assert len(text_blocks) == 1
            assert "42" in text_blocks[0]["text"]

    async def test_with_system_prompt(self, proxy_app):
        async with proxy_app as client:
            resp = await client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 100,
                    "system": "You are a helpful math tutor.",
                    "messages": [{"role": "user", "content": "1+1?"}],
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["type"] == "message"

    async def test_model_override(self, proxy_app):
        """Proxy should use the configured model name."""
        async with proxy_app as client:
            resp = await client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["model"] == "test-model"

    async def test_invalid_request(self, proxy_app):
        async with proxy_app as client:
            # Missing messages
            resp = await client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 100,
                },
            )
            assert resp.status_code == 400

    async def test_missing_model(self, proxy_app):
        async with proxy_app as client:
            resp = await client.post(
                "/v1/messages",
                json={
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            assert resp.status_code == 400


class TestProxyToolCall:
    @pytest.fixture
    def proxy_app(self):
        return _make_proxy_client(_create_mock_upstream())

    async def test_tool_call_returns_tool_use(self, proxy_app):
        """When upstream returns tool_calls, proxy should return tool_use blocks."""
        async with proxy_app as client:
            resp = await client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 100,
                    "tools": [{"name": "get_weather", "description": "Get weather"}],
                    "messages": [{"role": "user", "content": "Weather in Tokyo?"}],
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            tool_blocks = [c for c in body["content"] if c["type"] == "tool_use"]
            assert len(tool_blocks) == 1
            assert tool_blocks[0]["name"] == "get_weather"
            assert tool_blocks[0]["input"]["city"] == "Tokyo"
            assert body["stop_reason"] == "tool_use"

    async def test_tool_result_round_trip(self, proxy_app):
        """After tool_result, proxy should forward and return text."""
        async with proxy_app as client:
            resp = await client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 100,
                    "messages": [
                        {"role": "user", "content": "Weather?"},
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Let me check."},
                                {
                                    "type": "tool_use",
                                    "id": "call_abc123",
                                    "name": "get_weather",
                                    "input": {"city": "Tokyo"},
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "call_abc123",
                                    "content": "Sunny, 25C",
                                }
                            ],
                        },
                    ],
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            text_blocks = [c for c in body["content"] if c["type"] == "text"]
            assert len(text_blocks) == 1
            assert "sunny" in text_blocks[0]["text"].lower()


class TestProxyStreaming:
    @pytest.fixture
    def proxy_with_stream(self):
        return _make_proxy_client(_create_stream_upstream())

    async def test_streaming_returns_sse(self, proxy_with_stream):
        async with proxy_with_stream as client:
            resp = await client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert resp.status_code == 200
            text = resp.text
            assert "message_start" in text
            assert "text_delta" in text
            assert "message_stop" in text
            assert "Hello" in text
            assert "world!" in text
