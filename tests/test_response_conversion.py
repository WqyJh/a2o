"""Unit tests for OpenAI → Anthropic response conversion."""

from a2o.converters.response import convert_openai_to_anthropic


class TestBasicResponse:
    def test_text_only(self):
        openai = {
            "id": "chatcmpl-abc123",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello there!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = convert_openai_to_anthropic(openai)
        assert result["id"] == "chatcmpl-abc123"
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["stop_reason"] == "end_turn"
        assert len(result["content"]) == 1
        assert result["content"][0] == {"type": "text", "text": "Hello there!"}
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_tool_calls(self):
        openai = {
            "id": "chatcmpl-def456",
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
        result = convert_openai_to_anthropic(openai)
        assert result["stop_reason"] == "tool_use"
        text_blocks = [c for c in result["content"] if c["type"] == "text"]
        tool_blocks = [c for c in result["content"] if c["type"] == "tool_use"]
        assert len(text_blocks) == 0
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["id"] == "call_abc123"
        assert tool_blocks[0]["name"] == "get_weather"
        assert tool_blocks[0]["input"] == {"city": "Tokyo"}

    def test_reasoning_content(self):
        openai = {
            "id": "chatcmpl-ghi789",
            "model": "deepseek-reasoner",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 42.",
                        "reasoning_content": "Let me think step by step...",
                        "_thinking_signature": "sig_abc",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 100},
        }
        result = convert_openai_to_anthropic(openai)
        thinking_blocks = [c for c in result["content"] if c["type"] == "thinking"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "Let me think step by step..."
        assert thinking_blocks[0]["signature"] == "sig_abc"

    def test_length_finish_reason(self):
        openai = {
            "id": "chatcmpl-len",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "truncated"},
                    "finish_reason": "length",
                }
            ],
        }
        result = convert_openai_to_anthropic(openai)
        assert result["stop_reason"] == "max_tokens"

    def test_empty_choices(self):
        openai = {"id": "chatcmpl-empty", "model": "gpt-4o", "choices": []}
        result = convert_openai_to_anthropic(openai)
        assert result["content"] == []

    def test_cached_tokens(self):
        openai = {
            "id": "chatcmpl-cached",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "prompt_tokens_details": {"cached_tokens": 50},
            },
        }
        result = convert_openai_to_anthropic(openai)
        assert result["usage"]["cache_read_input_tokens"] == 50

    def test_model_override(self):
        openai = {
            "id": "chatcmpl-id",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                }
            ],
        }
        result = convert_openai_to_anthropic(openai, model="0312_pro")
        assert result["model"] == "0312_pro"

    def test_reasoning_content_without_text(self):
        """When content is None/empty but reasoning_content has text,
        treat reasoning_content as the response text."""
        openai = {
            "id": "chatcmpl-reasoning-only",
            "model": "0312_pro",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "Hello! I am doing well.",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        result = convert_openai_to_anthropic(openai)
        # Should be text, not thinking
        text_blocks = [c for c in result["content"] if c["type"] == "text"]
        thinking_blocks = [c for c in result["content"] if c["type"] == "thinking"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Hello! I am doing well."
        assert len(thinking_blocks) == 0
