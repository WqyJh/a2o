"""Convert Anthropic message format to OpenAI chat completion format."""

from __future__ import annotations

from typing import Any

from a2o.models import (
    AnthropicMessage,
    AnthropicMessageRequest,
    SystemContentBlock,
)


def _build_system(req: AnthropicMessageRequest) -> str | None:
    """Extract system prompt from Anthropic request."""
    if req.system is None:
        return None
    if isinstance(req.system, str):
        return req.system
    parts: list[str] = []
    for block in req.system:
        if block.text:
            parts.append(block.text)
    return "\n\n".join(parts) if parts else None


def _tools_to_openai(tools) -> list[dict[str, Any]] | None:
    """Convert Anthropic tools to OpenAI format."""
    if not tools:
        return None
    result: list[dict[str, Any]] = []
    for tool in tools:
        if not tool or not tool.name:
            continue
        f: dict[str, Any] = {"name": tool.name}
        if tool.description:
            f["description"] = tool.description
        if tool.input_schema:
            # Anthropic uses "input_schema", OpenAI uses "parameters"
            params = dict(tool.input_schema)
            # Ensure "type": "object" is present as OpenAI requires it
            if "type" not in params:
                params["type"] = "object"
            f["parameters"] = params
        result.append({"type": "function", "function": f})
    return result if result else None


def _tool_choice_to_openai(tc) -> dict[str, Any] | str | None:
    """Convert Anthropic tool_choice to OpenAI format."""
    if not tc or not tc.type:
        return None
    if tc.type == "auto":
        return "auto"
    if tc.type == "any":
        return "required"
    if tc.type == "none":
        return "none"
    if tc.type == "tool" and tc.name:
        return {"type": "function", "function": {"name": tc.name}}
    return "auto"


def _flatten_content(content: str | list) -> list[dict[str, Any]]:
    """Convert Anthropic content blocks to OpenAI content parts list.
    Text parts are joined. Others are converted individually."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    parts: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for block in content:
        if block.type == "text":
            if block.text:
                text_parts.append(block.text)
        elif block.type == "image" and block.source:
            url = ""
            if block.source.get("type") == "url":
                url = block.source.get("url", "")
            elif block.source.get("type") == "base64":
                media = block.source.get("media_type", "image/png")
                data = block.source.get("data", "")
                url = f"data:{media};base64,{data}"
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})

    joined_text = "\n".join(text_parts)
    if joined_text:
        parts.insert(0, {"type": "text", "text": joined_text})
    return parts


def _process_assistant_message(msg: AnthropicMessage) -> dict[str, Any]:
    """Build an OpenAI assistant message from an Anthropic assistant message."""
    oai_msg: dict[str, Any] = {"role": "assistant"}
    content = msg.content
    if isinstance(content, str):
        oai_msg["content"] = content
        return oai_msg

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in content:
        if block.type == "text" and block.text:
            text_parts.append(block.text)
        elif block.type == "thinking":
            if block.thinking:
                thinking_parts.append(block.thinking)
            # Preserve signature for cross-turn continuity
            if block.signature:
                oai_msg["_thinking_signature"] = block.signature
        elif block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id or "",
                    "type": "function",
                    "function": {
                        "name": block.name or "",
                        "arguments": _serialize_input(block.input),
                    },
                }
            )

    if text_parts:
        oai_msg["content"] = "\n".join(text_parts)
    else:
        oai_msg["content"] = None

    if thinking_parts:
        oai_msg["reasoning_content"] = "\n".join(thinking_parts)

    if tool_calls:
        oai_msg["tool_calls"] = tool_calls

    return oai_msg


def _serialize_input(inp: Any) -> str:
    """Serialize tool_use input to JSON string."""
    import json as _json

    if inp is None:
        return "{}"
    if isinstance(inp, str):
        return inp
    return _json.dumps(inp, ensure_ascii=True)


def _process_user_message(msg: AnthropicMessage) -> list[dict[str, Any]]:
    """Build OpenAI user messages from an Anthropic user message.
    Handles tool_result blocks by emitting separate tool messages,
    then user message with remaining content."""
    content = msg.content
    if isinstance(content, str):
        return [{"role": "user", "content": content}]

    messages: list[dict[str, Any]] = []
    text_parts: list[str] = []
    image_parts: list[dict[str, Any]] = []

    for block in content:
        if block.type == "tool_result":
            # Emit a tool message for the tool result
            tool_content = _flatten_tool_result_content(block.content)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": block.tool_use_id or "",
                    "content": tool_content or "",
                }
            )
            # If there are image parts in tool_result, emit as user message too
            if isinstance(block.content, list):
                for sub in block.content:
                    if sub.type == "image" and sub.source:
                        url = ""
                        if sub.source.get("type") == "url":
                            url = sub.source.get("url", "")
                        elif sub.source.get("type") == "base64":
                            media = sub.source.get("media_type", "image/png")
                            data = sub.source.get("data", "")
                            url = f"data:{media};base64,{data}"
                        if url:
                            image_parts.append(
                                {"type": "image_url", "image_url": {"url": url}}
                            )
        elif block.type == "text" and block.text:
            text_parts.append(block.text)
        elif block.type == "image" and block.source:
            url = ""
            if block.source.get("type") == "url":
                url = block.source.get("url", "")
            elif block.source.get("type") == "base64":
                media = block.source.get("media_type", "image/png")
                data = block.source.get("data", "")
                url = f"data:{media};base64,{data}"
            if url:
                image_parts.append({"type": "image_url", "image_url": {"url": url}})

    # Emit user message with remaining text + images
    user_content_parts: list[dict[str, Any]] = []
    if text_parts:
        user_content_parts.append({"type": "text", "text": "\n".join(text_parts)})
    user_content_parts.extend(image_parts)

    if user_content_parts:
        if len(user_content_parts) == 1 and user_content_parts[0]["type"] == "text":
            # Simple text message
            messages.append({"role": "user", "content": user_content_parts[0]["text"]})
        else:
            messages.append({"role": "user", "content": user_content_parts})
    elif not messages:
        messages.append({"role": "user", "content": ""})

    return messages


def _flatten_tool_result_content(content: str | list | None) -> str:
    """Flatten tool result content to a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if block.type == "text" and block.text:
                parts.append(block.text)
        return "\n".join(parts)
    return str(content)


def convert_anthropic_to_openai(req: AnthropicMessageRequest) -> dict[str, Any]:
    """Convert an Anthropic messages request body to an OpenAI chat completions body."""
    body: dict[str, Any] = {"model": req.model}

    if req.max_tokens is not None:
        body["max_tokens"] = req.max_tokens

    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p

    # Thinking
    if req.thinking and req.thinking.is_enabled():
        body["thinking"] = {"type": "enabled"}

    # System
    system_text = _build_system(req)
    if system_text:
        body["messages"] = body.get("messages", [])
        body["messages"].insert(0, {"role": "system", "content": system_text})

    # Messages
    openai_messages: list[dict[str, Any]] = []
    for msg in req.messages:
        if not msg or not msg.role:
            continue
        if msg.role == "assistant":
            openai_messages.append(_process_assistant_message(msg))
        elif msg.role == "user":
            openai_messages.extend(_process_user_message(msg))

    if "messages" in body:
        body["messages"].extend(openai_messages)
    else:
        body["messages"] = openai_messages

    # Tools
    tools = _tools_to_openai(req.tools)
    if tools:
        body["tools"] = tools

    # Tool choice
    tc = _tool_choice_to_openai(req.tool_choice)
    if tc is not None:
        body["tool_choice"] = tc
        if req.tool_choice and req.tool_choice.disable_parallel_tool_use:
            body["parallel_tool_calls"] = False

    # Stop
    if req.stop_sequences:
        if len(req.stop_sequences) == 1:
            body["stop"] = req.stop_sequences[0]
        else:
            body["stop"] = req.stop_sequences

    # Stream
    if req.stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}

    return body
