"""Parse raw JSON dicts into Anthropic model dataclasses."""

from __future__ import annotations

from typing import Any

from a2o.models import (
    AnthropicMessage,
    AnthropicMessageRequest,
    AnthropicThinking,
    AnthropicTool,
    AnthropicToolChoice,
    ContentBlock,
    SystemContentBlock,
)


class ParseError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def _validate(cond: bool, msg: str) -> None:
    if not cond:
        raise ParseError(msg)


def _parse_thinking(data: dict[str, Any] | None) -> AnthropicThinking | None:
    if not data:
        return None
    t = data.get("type", "disabled")
    return AnthropicThinking(type=t)


def _parse_tool_choice(data: dict[str, Any] | None) -> AnthropicToolChoice | None:
    if not data:
        return None
    return AnthropicToolChoice(
        type=data.get("type"),
        name=data.get("name"),
        disable_parallel_tool_use=data.get("disable_parallel_tool_use"),
    )


def _parse_tools(data: list[dict[str, Any]] | None) -> list[AnthropicTool] | None:
    if not data:
        return None
    tools: list[AnthropicTool] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        tools.append(
            AnthropicTool(
                name=item.get("name", ""),
                description=item.get("description"),
                input_schema=item.get("input_schema"),
            )
        )
    return tools if tools else None


def _parse_content_block(raw: dict[str, Any]) -> ContentBlock:
    block_type = raw.get("type", "")
    cb = ContentBlock(type=block_type)
    if block_type == "text":
        cb.text = raw.get("text")
        cb.cache_control = raw.get("cache_control")
    elif block_type == "tool_use":
        cb.id = raw.get("id")
        cb.name = raw.get("name")
        cb.input = raw.get("input")
    elif block_type == "tool_result":
        cb.tool_use_id = raw.get("tool_use_id")
        raw_content = raw.get("content")
        if isinstance(raw_content, str):
            cb.content = raw_content
        elif isinstance(raw_content, list):
            cb.content = [_parse_content_block(c) for c in raw_content if isinstance(c, dict)]
        elif raw_content is None:
            cb.content = None
        else:
            cb.content = str(raw_content)
        cb.is_error = raw.get("is_error", False)
    elif block_type == "thinking":
        cb.thinking = raw.get("thinking")
        cb.signature = raw.get("signature")
    elif block_type == "redacted_thinking":
        cb.signature = raw.get("data") or raw.get("signature")
    elif block_type == "image":
        cb.source = raw.get("source")
    else:
        # Preserve unknown blocks
        for k, v in raw.items():
            if k != "type":
                setattr(cb, k, v)
    return cb


def _parse_content(raw: str | list[dict[str, Any]]) -> str | list[ContentBlock]:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return [_parse_content_block(b) for b in raw if isinstance(b, dict)]
    return str(raw)


def _parse_system(
    raw: str | list[dict[str, Any]] | None,
) -> str | list[SystemContentBlock] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        blocks: list[SystemContentBlock] = []
        for item in raw:
            if isinstance(item, dict):
                blocks.append(
                    SystemContentBlock(
                        type=item.get("type"),
                        text=item.get("text"),
                    )
                )
            elif isinstance(item, str):
                blocks.append(SystemContentBlock(type="text", text=item))
        return blocks if blocks else None
    return str(raw)


def _parse_messages(
    data: list[dict[str, Any]],
) -> list[AnthropicMessage]:
    messages: list[AnthropicMessage] = []
    for i, raw in enumerate(data):
        _validate(isinstance(raw, dict), f"messages[{i}] must be a JSON object")
        role = raw.get("role")
        _validate(bool(role) and isinstance(role, str), f"messages[{i}].role is required")
        _validate(
            role in ("user", "assistant"),
            f"messages[{i}].role must be 'user' or 'assistant', got '{role}'",
        )
        content = raw.get("content")
        assert content is not None  # validated above
        messages.append(
            AnthropicMessage(
                role=str(role),
                content=_parse_content(content),
            )
        )
    return messages


def parse_anthropic_request(data: dict[str, Any]) -> AnthropicMessageRequest:
    """Parse a raw JSON dict into an AnthropicMessageRequest."""
    _validate(isinstance(data, dict), "Request body must be a JSON object")

    model = data.get("model")
    _validate(bool(model) and isinstance(model, str), "'model' is required")
    assert isinstance(model, str) and model  # guaranteed by validation
    model = model.strip()
    _validate(len(model) > 0, "'model' must not be empty")

    max_tokens = data.get("max_tokens")

    messages_raw = data.get("messages")
    _validate(
        messages_raw is not None and isinstance(messages_raw, list) and len(messages_raw) > 0,
        "'messages' must contain at least one entry",
    )
    assert isinstance(messages_raw, list)  # guaranteed by validation
    messages = _parse_messages(messages_raw)

    system = _parse_system(data.get("system"))

    return AnthropicMessageRequest(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        system=system,
        stop_sequences=data.get("stop_sequences"),
        stream=bool(data.get("stream")),
        temperature=data.get("temperature"),
        top_p=data.get("top_p"),
        top_k=data.get("top_k"),
        metadata=data.get("metadata"),
        tools=_parse_tools(data.get("tools")),
        tool_choice=_parse_tool_choice(data.get("tool_choice")),
        thinking=_parse_thinking(data.get("thinking")),
    )
