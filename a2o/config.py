"""Configuration for the Anthropic proxy."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    openai_base_url: str = ""
    default_model: str = ""
    host: str = "127.0.0.1"
    port: int = 3578
    request_timeout: int = 300
    stream_timeout: int = 300
    workers: int = 1
    max_connections: int = 1000
    max_connections_per_host: int = 500
