"""Convert OpenAI chat completion response to Anthropic message format."""

from __future__ import annotations

import json
import uuid
from typing import Any

from a2o.models import AnthropicUsage


_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
}


def _map_finish_reason(reason: str | None) -> str:
    if reason is None:
        return "end_turn"
    return _FINISH_REASON_MAP.get(reason, reason)


def _parse_tool_arguments(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, str):
        try:
            return json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return {}
    if isinstance(args, dict):
        return args
    return {}


def _build_usage(usage: dict[str, Any] | None) -> AnthropicUsage | None:
    if not usage:
        return None
    result = AnthropicUsage(
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
    )
    # Handle cached tokens
    prompt_details = usage.get("prompt_tokens_details")
    if prompt_details and isinstance(prompt_details, dict):
        cached = prompt_details.get("cached_tokens")
        if cached is not None:
            result.cache_read_input_tokens = cached
    return result


def convert_openai_to_anthropic(
    openai_resp: dict[str, Any], model: str | None = None
) -> dict[str, Any]:
    """Convert a non-streaming OpenAI chat completion response to Anthropic format.

    Returns a dict suitable for JSON serialization of the Anthropic response.
    """
    choices = openai_resp.get("choices") or []
    msg_id = openai_resp.get("id") or f"msg_{uuid.uuid4().hex[:24]}"

    content: list[dict[str, Any]] = []
    finish_reason: str | None = None
    used_model = model or openai_resp.get("model", "")

    if choices:
        first = choices[0]
        message = first.get("message", {})
        finish_reason = first.get("finish_reason")

        # Text content
        text = message.get("content")
        if text:
            content.append({"type": "text", "text": text})

        # Reasoning content (thinking)
        additional = message.get("_additionalProperties") or {}
        # Some backends put reasoning_content at the top level of the message dict
        reasoning = additional.get("reasoning_content") or message.get(
            "reasoning_content"
        )
        signature = (
            additional.get("thinking_signature")
            or message.get("_thinking_signature")
            or ""
        )
        if reasoning:
            # If content is empty and reasoning_content has text, it's the model's
            # actual answer being placed in the thinking field (common with extended
            # thinking models). Put it as text content in that case.
            if not text and reasoning.strip():
                content.append({"type": "text", "text": reasoning})
            else:
                content.append(
                    {
                        "type": "thinking",
                        "thinking": reasoning,
                        "signature": signature,
                    }
                )
        # Redacted thinking support
        redacted = additional.get("redacted_thinking") or message.get(
            "redacted_thinking"
        )
        if redacted:
            content.append(
                redacted
                if isinstance(redacted, dict)
                else {
                    "type": "redacted_thinking",
                    "data": redacted,
                }
            )

        # Tool calls
        tool_calls = message.get("tool_calls") or []
        for tc in tool_calls:
            func = tc.get("function", {})
            content.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": func.get("name", ""),
                    "input": _parse_tool_arguments(func.get("arguments")),
                }
            )

    result: dict[str, Any] = {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": used_model,
        "stop_reason": _map_finish_reason(finish_reason),
        "stop_sequence": None,
        "content": content,
        "usage": _serialize_usage(_build_usage(openai_resp.get("usage"))),
    }
    return result


def _serialize_usage(usage: AnthropicUsage | None) -> dict[str, Any]:
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0}
    d: dict[str, Any] = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }
    if usage.cache_read_input_tokens is not None:
        d["cache_read_input_tokens"] = usage.cache_read_input_tokens
    return d
