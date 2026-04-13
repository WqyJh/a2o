"""Convert OpenAI streaming chunks to Anthropic SSE events."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

# Event type constants
EVENT_MESSAGE_START = "message_start"
EVENT_CONTENT_BLOCK_START = "content_block_start"
EVENT_CONTENT_BLOCK_DELTA = "content_block_delta"
EVENT_CONTENT_BLOCK_STOP = "content_block_stop"
EVENT_MESSAGE_DELTA = "message_delta"
EVENT_MESSAGE_STOP = "message_stop"

# Content types
CONTENT_TEXT = "text"
CONTENT_THINKING = "thinking"
CONTENT_TOOL_USE = "tool_use"

# Delta types
DELTA_TEXT = "text_delta"
DELTA_THINKING = "thinking_delta"
DELTA_INPUT_JSON = "input_json_delta"

# Stop reasons
STOP_END_TURN = "end_turn"
STOP_MAX_TOKENS = "max_tokens"
STOP_TOOL_USE = "tool_use"

_FINISH_REASON_MAP = {
    "stop": STOP_END_TURN,
    "length": STOP_MAX_TOKENS,
    "tool_calls": STOP_TOOL_USE,
    "function_call": STOP_TOOL_USE,
}


def _map_finish(reason: str | None) -> str:
    if reason is None:
        return STOP_END_TURN
    return _FINISH_REASON_MAP.get(reason, reason)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"))


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


class _StreamState:
    """Tracks state for converting an OpenAI streaming response to Anthropic SSE."""

    def __init__(self, stream_id: str, model: str) -> None:
        self.stream_id = stream_id
        self.model = model
        self.next_block_index = 0
        self.active_content_block_index: int = -1
        self.active_content_block_type: str | None = None
        self.message_started = False
        self.final_stop_reason = STOP_END_TURN
        self.latest_usage: dict[str, int] | None = None
        # Track tool call states: index -> {id, name, args_buffer, ...}
        self._tool_calls: dict[int, dict[str, Any]] = {}
        # Track whether text and thinking blocks were emitted
        # Used to decide if thinking→text demotion is needed
        self._had_text = False
        self._had_thinking = False
        self._thinking_buffer: list[str] = []

    def _ensure_active_block(self, events: list[str], block_type: str) -> int:
        if block_type != self.active_content_block_type:
            self._close_active_block(events)
            idx = self.next_block_index
            self.next_block_index += 1
            self.active_content_block_index = idx
            self.active_content_block_type = block_type

            if block_type == CONTENT_TEXT:
                data = _json_dumps(
                    {
                        "type": EVENT_CONTENT_BLOCK_START,
                        "index": idx,
                        "content_block": {"type": CONTENT_TEXT, "text": ""},
                    }
                )
            elif block_type == CONTENT_THINKING:
                data = _json_dumps(
                    {
                        "type": EVENT_CONTENT_BLOCK_START,
                        "index": idx,
                        "content_block": {"type": CONTENT_THINKING, "thinking": ""},
                    }
                )
            else:
                return self.active_content_block_index
            events.append(_sse(EVENT_CONTENT_BLOCK_START, data))
        return self.active_content_block_index

    def _close_active_block(self, events: list[str]) -> None:
        if self.active_content_block_index >= 0:
            data = _json_dumps(
                {
                    "type": EVENT_CONTENT_BLOCK_STOP,
                    "index": self.active_content_block_index,
                }
            )
            events.append(_sse(EVENT_CONTENT_BLOCK_STOP, data))
            self.active_content_block_index = -1
            self.active_content_block_type = None

    def _process_tool_call(self, events: list[str], tc: dict[str, Any]) -> None:
        tc_index = tc.get("index", 0)

        tool_state = self._tool_calls.setdefault(
            tc_index,
            {
                "id": None,
                "name": None,
                "args_buffer": "",
                "claude_index": None,
                "started": False,
                "json_sent": False,
                "stopped": False,
            },
        )

        # Update id and name
        tc_id = tc.get("id")
        if tc_id:
            tool_state["id"] = tc_id
        func = tc.get("function", {})
        tc_name = func.get("name")
        if tc_name:
            tool_state["name"] = tc_name
        # Buffer arguments
        args = func.get("arguments")
        if args and not tool_state["json_sent"]:
            tool_state["args_buffer"] += args

        # Start content block when we have id + name
        if tool_state["id"] and tool_state["name"] and not tool_state["started"]:
            self._close_active_block(events)
            tool_state["claude_index"] = self.next_block_index
            self.next_block_index += 1
            tool_state["started"] = True

            start_data = _json_dumps(
                {
                    "type": EVENT_CONTENT_BLOCK_START,
                    "index": tool_state["claude_index"],
                    "content_block": {
                        "type": CONTENT_TOOL_USE,
                        "id": tool_state["id"],
                        "name": tool_state["name"],
                        "input": {},
                    },
                }
            )
            events.append(_sse(EVENT_CONTENT_BLOCK_START, start_data))

        # Try to emit accumulated JSON
        if tool_state["started"] and tool_state["args_buffer"] and not tool_state["json_sent"]:
            self._maybe_emit_tool_args(events, tool_state)

    def _maybe_emit_tool_args(self, events: list[str], tool_state: dict[str, Any]) -> None:
        buf = tool_state["args_buffer"]
        if not buf:
            return
        try:
            # Try to parse as JSON - if valid, emit it
            json.loads(buf)
            delta_data = _json_dumps(
                {
                    "type": EVENT_CONTENT_BLOCK_DELTA,
                    "index": tool_state["claude_index"],
                    "delta": {"type": DELTA_INPUT_JSON, "partial_json": buf},
                }
            )
            events.append(_sse(EVENT_CONTENT_BLOCK_DELTA, delta_data))
            tool_state["json_sent"] = True
            if not tool_state["stopped"]:
                stop_data = _json_dumps(
                    {
                        "type": EVENT_CONTENT_BLOCK_STOP,
                        "index": tool_state["claude_index"],
                    }
                )
                events.append(_sse(EVENT_CONTENT_BLOCK_STOP, stop_data))
                tool_state["stopped"] = True
        except json.JSONDecodeError:
            pass  # Wait for more data

    def _flush_final_tool_calls(self, events: list[str]) -> None:
        # Emit remaining tool call data (even if not valid JSON - keep expanding)
        for _idx in sorted(self._tool_calls.keys()):
            tool_state = self._tool_calls[_idx]
            if not tool_state["started"] or tool_state["stopped"]:
                continue
            buf = tool_state["args_buffer"]
            if buf and not tool_state["json_sent"]:
                # If we have buffered args but couldn't parse as complete JSON,
                # emit them as-is (partial JSON)
                delta_data = _json_dumps(
                    {
                        "type": EVENT_CONTENT_BLOCK_DELTA,
                        "index": tool_state["claude_index"],
                        "delta": {"type": DELTA_INPUT_JSON, "partial_json": buf},
                    }
                )
                events.append(_sse(EVENT_CONTENT_BLOCK_DELTA, delta_data))
                tool_state["json_sent"] = True
            if not tool_state["stopped"] and tool_state["claude_index"] is not None:
                stop_data = _json_dumps(
                    {
                        "type": EVENT_CONTENT_BLOCK_STOP,
                        "index": tool_state["claude_index"],
                    }
                )
                events.append(_sse(EVENT_CONTENT_BLOCK_STOP, stop_data))
                tool_state["stopped"] = True

    def process_chunk(self, chunk: dict[str, Any]) -> list[str]:
        """Process a single OpenAI streaming chunk. Returns list of SSE strings."""
        events: list[str] = []

        try:
            # Emit message_start on first chunk
            if not self.message_started:
                model = chunk.get("model", self.model) or self.model
                self.model = model
                msg_start = _json_dumps(
                    {
                        "type": EVENT_MESSAGE_START,
                        "message": {
                            "id": self.stream_id,
                            "type": "message",
                            "role": "assistant",
                            "model": self.model,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    }
                )
                events.append(_sse(EVENT_MESSAGE_START, msg_start))
                self.message_started = True

            # Update usage from chunk if available
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                self.latest_usage = {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                }
                pd = usage.get("prompt_tokens_details")
                if isinstance(pd, dict) and pd.get("cached_tokens") is not None:
                    self.latest_usage["cache_read_input_tokens"] = pd["cached_tokens"]

            # Process choices
            for choice in chunk.get("choices") or []:
                self._process_choice(events, choice)

        except Exception as exc:
            import logging

            logging.error("Error processing chunk: %s", exc, exc_info=True)

        return events

    def _process_choice(self, events: list[str], choice: dict[str, Any]) -> None:
        delta = choice.get("delta", {})
        if delta is None:
            delta = {}

        # Text content
        if delta.get("content"):
            self._had_text = True
            idx = self._ensure_active_block(events, CONTENT_TEXT)
            data = _json_dumps(
                {
                    "type": EVENT_CONTENT_BLOCK_DELTA,
                    "index": idx,
                    "delta": {"type": DELTA_TEXT, "text": delta["content"]},
                }
            )
            events.append(_sse(EVENT_CONTENT_BLOCK_DELTA, data))

        # Thinking / reasoning content
        additional = delta.get("_additionalProperties") or {}
        reasoning = additional.get("reasoning_content") or delta.get("reasoning_content")
        if reasoning:
            self._had_thinking = True
            self._thinking_buffer.append(reasoning)
            idx = self._ensure_active_block(events, CONTENT_THINKING)
            data = _json_dumps(
                {
                    "type": EVENT_CONTENT_BLOCK_DELTA,
                    "index": idx,
                    "delta": {"type": DELTA_THINKING, "thinking": reasoning},
                }
            )
            events.append(_sse(EVENT_CONTENT_BLOCK_DELTA, data))

        # Tool calls
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                self._process_tool_call(events, tc)

        # Finish reason
        finish = choice.get("finish_reason")
        if finish:
            self.final_stop_reason = _map_finish(finish)

    def finalize(self) -> list[str]:
        """Emit final events (stop blocks, message_delta, message_stop)."""
        events: list[str] = []

        self._close_active_block(events)

        # Stop remaining tool call blocks
        self._flush_final_tool_calls(events)

        # message_delta
        usage = self.latest_usage or {"input_tokens": 0, "output_tokens": 0}
        msg_delta = _json_dumps(
            {
                "type": EVENT_MESSAGE_DELTA,
                "delta": {
                    "stop_reason": self.final_stop_reason,
                    "stop_sequence": None,
                },
                "usage": usage,
            }
        )
        events.append(_sse(EVENT_MESSAGE_DELTA, msg_delta))

        # message_stop
        events.append(_sse(EVENT_MESSAGE_STOP, _json_dumps({"type": EVENT_MESSAGE_STOP})))

        return events


def convert_openai_stream_to_anthropic_sse(
    chunks: list[dict[str, Any]], model: str = ""
) -> list[str]:
    """Convert a list of OpenAI streaming chunk dicts to Anthropic SSE events."""
    stream_id = f"msg_{uuid.uuid4().hex[:24]}"
    state = _StreamState(stream_id, model)
    all_events: list[str] = []

    for chunk in chunks:
        all_events.extend(state.process_chunk(chunk))

    all_events.extend(state.finalize())
    return all_events


async def async_convert_openai_stream_to_anthropic_sse(
    chunk_iter: AsyncIterator[dict[str, Any]], model: str = ""
) -> AsyncIterator[str]:
    """Async generator version for streaming proxy use."""
    stream_id = f"msg_{uuid.uuid4().hex[:24]}"
    state = _StreamState(stream_id, model)

    async for chunk in chunk_iter:
        for event in state.process_chunk(chunk):
            yield event

    for event in state.finalize():
        yield event
