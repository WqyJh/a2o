"""Configuration for the Anthropic proxy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    openai_base_url: str = ""
    default_model: str = ""
    host: str = "127.0.0.1"
    port: int = 3578
    request_timeout: int | None = None
    stream_timeout: int | None = None
    workers: int = 1
    max_connections: int = 1000
    max_connections_per_host: int = 500
    log_file: str = ""
    debug: int = 0
