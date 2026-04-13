"""OpenAI / Anthropic data models for request/response conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Anthropic request types
# ---------------------------------------------------------------------------


@dataclass
class AnthropicThinking:
    type: str = "disabled"

    def is_enabled(self) -> bool:
        return self.type == "enabled"


@dataclass
class AnthropicToolChoice:
    type: str | None = None
    name: str | None = None
    disable_parallel_tool_use: bool | None = False


@dataclass
class AnthropicTool:
    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None


@dataclass
class ContentBlock:
    """Polymorphic content block for Anthropic messages."""

    type: str  # text | tool_use | tool_result | thinking | image
    # text
    text: str | None = None
    # tool_use
    id: str | None = None
    name: str | None = None
    input: Any = None
    # tool_result
    tool_use_id: str | None = None
    content: str | list[ContentBlock] | None = None
    is_error: bool = False
    # thinking
    thinking: str | None = None
    signature: str | None = None
    # image
    source: dict[str, Any] | None = None
    # cache_control
    cache_control: dict[str, str] | None = None


@dataclass
class AnthropicMessage:
    role: str  # user | assistant
    content: str | list[ContentBlock]


@dataclass
class SystemContentBlock:
    type: str | None = None
    text: str | None = None


@dataclass
class AnthropicMessageRequest:
    model: str
    max_tokens: int | None = None
    messages: list[AnthropicMessage] = field(default_factory=list)
    system: str | list[SystemContentBlock] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    metadata: dict[str, str] | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: AnthropicToolChoice | None = None
    thinking: AnthropicThinking | None = None


# ---------------------------------------------------------------------------
# Anthropic response types
# ---------------------------------------------------------------------------


@dataclass
class AnthropicUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int | None = None


@dataclass
class AnthropicMessageResponse:
    id: str
    type: str = "message"
    role: str = "assistant"
    model: str = ""
    stop_reason: str | None = None
    stop_sequence: str | None = None
    content: list[dict[str, Any]] = field(default_factory=list)
    usage: AnthropicUsage | None = None
    # Optional error fields
    error: dict[str, Any] | None = None
