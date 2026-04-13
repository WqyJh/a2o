"""Unit tests for Anthropic → OpenAI request conversion."""

import json

from a2o.converters.parser import parse_anthropic_request
from a2o.converters.request import convert_anthropic_to_openai

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _parse_and_convert(data: dict) -> dict:
    req = parse_anthropic_request(data)
    return convert_anthropic_to_openai(req)


# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------


class TestBasicRequest:
    def test_simple_text_message(self):
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        body = _parse_and_convert(data)
        assert body["model"] == "claude-sonnet-4-20250514"
        assert body["max_tokens"] == 1024
        assert body["messages"] == [{"role": "user", "content": "Hello"}]

    def test_system_string(self):
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 500,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        body = _parse_and_convert(data)
        assert body["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert body["messages"][1] == {"role": "user", "content": "Hi"}

    def test_system_array(self):
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 500,
            "system": [
                {"type": "text", "text": "Part one."},
                {"type": "text", "text": "Part two."},
            ],
            "messages": [{"role": "user", "content": "test"}],
        }
        body = _parse_and_convert(data)
        assert body["messages"][0]["content"] == "Part one.\n\nPart two."

    def test_temperature_top_p(self):
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "messages": [{"role": "user", "content": "test"}],
        }
        body = _parse_and_convert(data)
        assert body["temperature"] == 0.7
        assert body["top_p"] == 0.9


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class TestTools:
    def test_single_tool(self):
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                    },
                }
            ],
            "messages": [{"role": "user", "content": "Weather in Tokyo?"}],
        }
        body = _parse_and_convert(data)
        assert "tools" in body
        assert len(body["tools"]) == 1
        tool = body["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert "parameters" in tool["function"]

    def test_tool_choice_auto(self):
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "tool_choice": {"type": "auto"},
            "messages": [{"role": "user", "content": "hi"}],
        }
        body = _parse_and_convert(data)
        assert body["tool_choice"] == "auto"

    def test_tool_choice_required(self):
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "tool_choice": {"type": "any"},
            "messages": [{"role": "user", "content": "hi"}],
        }
        body = _parse_and_convert(data)
        assert body["tool_choice"] == "required"

    def test_tool_choice_none(self):
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "tool_choice": {"type": "none"},
            "messages": [{"role": "user", "content": "hi"}],
        }
        body = _parse_and_convert(data)
        assert body["tool_choice"] == "none"

    def test_tool_choice_specific_tool(self):
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "tool_choice": {"type": "tool", "name": "get_weather"},
            "messages": [{"role": "user", "content": "hi"}],
        }
        body = _parse_and_convert(data)
        assert body["tool_choice"]["type"] == "function"
        assert body["tool_choice"]["function"]["name"] == "get_weather"

    def test_disable_parallel_tool_use(self):
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            "messages": [{"role": "user", "content": "hi"}],
        }
        body = _parse_and_convert(data)
        assert body["parallel_tool_calls"] is False


# ---------------------------------------------------------------------------
# Assistant messages with tool_use and thinking
# ---------------------------------------------------------------------------


class TestAssistantMessages:
    def test_text_only(self):
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
        }
        body = _parse_and_convert(data)
        assert body["messages"][-1]["content"] == "Hello!"

    def test_tool_use_block(self):
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Get weather"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_01A09q90qw90lq917835lq9",
                            "name": "get_weather",
                            "input": {"city": "Tokyo"},
                        }
                    ],
                },
            ],
        }
        body = _parse_and_convert(data)
        assistant_msg = body["messages"][-1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg.get("content") is None
        assert len(assistant_msg["tool_calls"]) == 1
        tc = assistant_msg["tool_calls"][0]
        assert tc["id"] == "toolu_01A09q90qw90lq917835lq9"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Tokyo"}

    def test_thinking_block(self):
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "I need to think about this...",
                            "signature": "abc123",
                        },
                        {"type": "text", "text": "Here's my answer."},
                    ],
                },
            ],
        }
        body = _parse_and_convert(data)
        assistant_msg = body["messages"][-1]
        assert assistant_msg["reasoning_content"] == "I need to think about this..."
        assert assistant_msg["content"] == "Here's my answer."

    def test_text_and_tool_use(self):
        data = {
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
                            "id": "toolu_123",
                            "name": "get_weather",
                            "input": {"city": "NYC"},
                        },
                    ],
                },
            ],
        }
        body = _parse_and_convert(data)
        assistant_msg = body["messages"][-1]
        assert assistant_msg["content"] == "Let me check."
        assert len(assistant_msg["tool_calls"]) == 1


# ---------------------------------------------------------------------------
# User messages with tool_result
# ---------------------------------------------------------------------------


class TestToolResults:
    def test_tool_result_string(self):
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_123",
                            "content": "Sunny, 25C",
                        }
                    ],
                },
            ],
        }
        body = _parse_and_convert(data)
        # Should produce a tool message
        tool_msg = body["messages"][0]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "toolu_123"
        assert tool_msg["content"] == "Sunny, 25C"

    def test_tool_result_with_text(self):
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Also consider the forecast.",
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_456",
                            "content": "Cloudy",
                        },
                    ],
                },
            ],
        }
        body = _parse_and_convert(data)
        # Should have tool message + user message
        tool_msgs = [m for m in body["messages"] if m["role"] == "tool"]
        user_msgs = [m for m in body["messages"] if m["role"] == "user"]
        assert len(tool_msgs) == 1
        assert len(user_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "toolu_456"
        assert user_msgs[0]["content"] == "Also consider the forecast."

    def test_tool_result_array_content(self):
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_789",
                            "content": [
                                {"type": "text", "text": "Result part 1"},
                                {"type": "text", "text": "Result part 2"},
                            ],
                        }
                    ],
                },
            ],
        }
        body = _parse_and_convert(data)
        tool_msg = body["messages"][0]
        assert tool_msg["role"] == "tool"
        assert "Result part 1" in tool_msg["content"]
        assert "Result part 2" in tool_msg["content"]


# ---------------------------------------------------------------------------
# Stop sequences
# ---------------------------------------------------------------------------


class TestStopSequences:
    def test_single(self):
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "stop_sequences": ["END"],
            "messages": [{"role": "user", "content": "hi"}],
        }
        body = _parse_and_convert(data)
        assert body["stop"] == "END"

    def test_multiple(self):
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "stop_sequences": ["STOP", "HALT"],
            "messages": [{"role": "user", "content": "hi"}],
        }
        body = _parse_and_convert(data)
        assert body["stop"] == ["STOP", "HALT"]


# ---------------------------------------------------------------------------
# Thinking / Extended Thinking
# ---------------------------------------------------------------------------


class TestThinking:
    def test_thinking_enabled(self):
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "thinking": {"type": "enabled"},
            "messages": [{"role": "user", "content": "Solve 2+2"}],
        }
        body = _parse_and_convert(data)
        assert body.get("thinking") == {"type": "enabled"}
