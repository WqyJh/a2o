"""Unit tests for OpenAI → Anthropic streaming conversion."""

import json

from a2o.converters.streaming import (
    EVENT_CONTENT_BLOCK_DELTA,
    EVENT_CONTENT_BLOCK_START,
    EVENT_CONTENT_BLOCK_STOP,
    EVENT_MESSAGE_DELTA,
    EVENT_MESSAGE_START,
    EVENT_MESSAGE_STOP,
    convert_openai_stream_to_anthropic_sse,
)


def _make_chunk(
    content: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str | None = None,
    reasoning_content: str | None = None,
    model: str = "test-model",
    usage: dict | None = None,
    first: bool = False,
) -> dict:
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content

    choice: dict = {
        "index": 0,
        "delta": delta,
    }
    if finish_reason:
        choice["finish_reason"] = finish_reason

    chunk: dict = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": model if first else "",
        "choices": [choice],
    }
    if usage:
        chunk["usage"] = usage
    return chunk


def _parse_sse_strings(sse_strings: list[str]) -> list[dict]:
    """Parse SSE event strings (each is a multi-line event: X\ndata: Y\n\n) into event dicts."""
    events = []
    for sse_str in sse_strings:
        # Each element is like "event: X\ndata: Y\n\n"
        lines = sse_str.strip().split("\n")
        event_type = None
        data_str = None
        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data_str = line[6:]
        if event_type and data_str:
            try:
                obj = json.loads(data_str)
                obj["_event_type"] = event_type
                events.append(obj)
            except json.JSONDecodeError:
                pass
    return events


class TestSimpleStream:
    def test_text_streaming(self):
        chunks = [
            _make_chunk(first=True),
            _make_chunk(content="Hello"),
            _make_chunk(content=" there!"),
            _make_chunk(
                finish_reason="stop",
                usage={
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                },
            ),
        ]
        sse = convert_openai_stream_to_anthropic_sse(chunks, "test-model")
        events = _parse_sse_strings(sse)

        event_types = [e["_event_type"] for e in events]
        assert EVENT_MESSAGE_START in event_types
        assert EVENT_CONTENT_BLOCK_START in event_types
        assert EVENT_CONTENT_BLOCK_DELTA in event_types
        assert EVENT_CONTENT_BLOCK_STOP in event_types
        assert EVENT_MESSAGE_DELTA in event_types
        assert EVENT_MESSAGE_STOP in event_types

        # Collect text deltas
        text_deltas = [e for e in events if e.get("delta", {}).get("type") == "text_delta"]
        combined = "".join(e["delta"]["text"] for e in text_deltas)
        assert combined == "Hello there!"

    def test_tool_call_streaming(self):
        chunks = [
            _make_chunk(first=True),
            _make_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_123",
                        "function": {"name": "get_weather", "arguments": ""},
                    }
                ]
            ),
            _make_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "function": {"arguments": '{"city":'},
                    }
                ]
            ),
            _make_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "function": {"arguments": ' "Tokyo"}'},
                    }
                ]
            ),
            _make_chunk(finish_reason="tool_calls"),
        ]
        sse = convert_openai_stream_to_anthropic_sse(chunks, "test-model")
        events = _parse_sse_strings(sse)

        # Verify tool_use start block
        tool_starts = [
            e
            for e in events
            if e["_event_type"] == EVENT_CONTENT_BLOCK_START
            and e.get("content_block", {}).get("type") == "tool_use"
        ]
        assert len(tool_starts) == 1
        assert tool_starts[0]["content_block"]["id"] == "call_123"
        assert tool_starts[0]["content_block"]["name"] == "get_weather"

        # Verify input_json_delta
        json_deltas = [
            e
            for e in events
            if e["_event_type"] == EVENT_CONTENT_BLOCK_DELTA
            and e.get("delta", {}).get("type") == "input_json_delta"
        ]
        assert len(json_deltas) >= 1

        # The final JSON should be complete
        combined_json = "".join(d["delta"]["partial_json"] for d in json_deltas)
        parsed = json.loads(combined_json)
        assert parsed == {"city": "Tokyo"}

        # Verify message_delta has tool_use stop reason
        msg_deltas = [e for e in events if e["_event_type"] == EVENT_MESSAGE_DELTA]
        assert len(msg_deltas) == 1
        assert msg_deltas[0]["delta"]["stop_reason"] == "tool_use"

    def test_thinking_streaming(self):
        chunks = [
            _make_chunk(first=True),
            _make_chunk(reasoning_content="Let me think..."),
            _make_chunk(reasoning_content=" More thoughts."),
            _make_chunk(content="The answer is 4."),
            _make_chunk(finish_reason="stop"),
        ]
        sse = convert_openai_stream_to_anthropic_sse(chunks, "test")
        events = _parse_sse_strings(sse)

        # Should have thinking content block
        thinking_starts = [
            e
            for e in events
            if e["_event_type"] == EVENT_CONTENT_BLOCK_START
            and e.get("content_block", {}).get("type") == "thinking"
        ]
        assert len(thinking_starts) == 1

        # Should have thinking deltas
        thinking_deltas = [
            e
            for e in events
            if e["_event_type"] == EVENT_CONTENT_BLOCK_DELTA
            and e.get("delta", {}).get("type") == "thinking_delta"
        ]
        assert len(thinking_deltas) == 2

        # Should have text after thinking (different block)
        text_starts = [
            e
            for e in events
            if e["_event_type"] == EVENT_CONTENT_BLOCK_START
            and e.get("content_block", {}).get("type") == "text"
        ]
        assert len(text_starts) == 1

    def test_usage_included(self):
        chunks = [
            _make_chunk(first=True),
            _make_chunk(content="Hi"),
            _make_chunk(
                finish_reason="stop",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 3},
                },
            ),
        ]
        sse = convert_openai_stream_to_anthropic_sse(chunks)
        events = _parse_sse_strings(sse)

        # message_delta should include usage
        msg_deltas = [e for e in events if e["_event_type"] == EVENT_MESSAGE_DELTA]
        assert len(msg_deltas) == 1
        usage = msg_deltas[0].get("usage", {})
        assert usage.get("input_tokens") == 10
        assert usage.get("output_tokens") == 5
        assert usage.get("cache_read_input_tokens") == 3


class TestToolCallWithArgsInSameChunk:
    """Some APIs send tool_call id/name/arguments in the same initial chunk."""

    def test_complete_tool_call(self):
        chunks = [
            _make_chunk(first=True),
            # Complete tool call with arguments in one chunk
            _make_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_xyz",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"/test.py"}',
                        },
                    }
                ]
            ),
            _make_chunk(finish_reason="tool_calls"),
        ]
        sse = convert_openai_stream_to_anthropic_sse(chunks)
        events = _parse_sse_strings(sse)

        json_deltas = [
            e
            for e in events
            if e["_event_type"] == EVENT_CONTENT_BLOCK_DELTA
            and e.get("delta", {}).get("type") == "input_json_delta"
        ]
        assert len(json_deltas) == 1
        assert json.loads(json_deltas[0]["delta"]["partial_json"]) == {"path": "/test.py"}


class TestMultipleToolCalls:
    def test_parallel_tool_call_streaming(self):
        chunks = [
            _make_chunk(first=True),
            # First tool call
            _make_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_a",
                        "function": {"name": "tool_a", "arguments": ""},
                    }
                ]
            ),
            # Second tool call
            _make_chunk(
                tool_calls=[
                    {
                        "index": 1,
                        "id": "call_b",
                        "function": {"name": "tool_b", "arguments": ""},
                    }
                ]
            ),
            # Args for first tool
            _make_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "function": {"arguments": '{"x":1}'},
                    }
                ]
            ),
            # Args for second tool
            _make_chunk(
                tool_calls=[
                    {
                        "index": 1,
                        "function": {"arguments": '{"y":2}'},
                    }
                ]
            ),
            _make_chunk(finish_reason="tool_calls"),
        ]
        sse = convert_openai_stream_to_anthropic_sse(chunks)
        events = _parse_sse_strings(sse)

        tool_starts = [
            e
            for e in events
            if e["_event_type"] == EVENT_CONTENT_BLOCK_START
            and e.get("content_block", {}).get("type") == "tool_use"
        ]
        assert len(tool_starts) == 2
        assert tool_starts[0]["content_block"]["id"] == "call_a"
        assert tool_starts[1]["content_block"]["id"] == "call_b"
