"""HTTP server that exposes the Anthropic Messages API and proxies to OpenAI."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from a2o.config import Config
from a2o.converters.parser import ParseError, parse_anthropic_request
from a2o.converters.request import convert_anthropic_to_openai
from a2o.converters.response import convert_openai_to_anthropic
from a2o.converters.streaming import (
    async_convert_openai_stream_to_anthropic_sse,
)

logger = logging.getLogger("a2o")


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _anthropic_error(status: int, error_type: str, message: str) -> dict:
    return {
        "type": "error",
        "error": {"type": error_type, "message": message},
    }


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class AnthropicMessageHandler:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._client: httpx.AsyncClient | None = None

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=self.config.max_connections,
                    max_keepalive_connections=self.config.max_connections_per_host,
                ),
                timeout=httpx.Timeout(self.config.request_timeout),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def handle_messages(self, request: Request) -> JSONResponse | StreamingResponse:
        """Handle POST /v1/messages (Anthropic API format)."""
        try:
            raw_body = await request.json()
        except Exception:
            return JSONResponse(
                _anthropic_error(400, "invalid_request_error", "Invalid JSON body"),
                status_code=400,
            )

        # Parse Anthropic request
        try:
            anthropic_req = parse_anthropic_request(raw_body)
        except ParseError as e:
            return JSONResponse(
                _anthropic_error(400, "invalid_request_error", e.message),
                status_code=400,
            )

        # Apply model override if configured
        model = anthropic_req.model
        if self.config.default_model:
            model = self.config.default_model

        # Force stream from raw body if needed
        streaming = raw_body.get("stream", False)
        if not streaming and anthropic_req.stream:
            streaming = True

        # Convert to OpenAI format
        openai_body = convert_anthropic_to_openai(anthropic_req)
        openai_body["model"] = model

        # Build upstream URL
        upstream_url = self.config.openai_base_url
        if not upstream_url:
            return JSONResponse(
                _anthropic_error(500, "api_error", "No upstream URL configured"),
                status_code=500,
            )

        # Forward incoming headers that might be useful
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        # Pass through Authorization header
        auth = request.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth

        if streaming:
            return await self._handle_stream(upstream_url, openai_body, headers, model)
        else:
            return await self._handle_nonstream(upstream_url, openai_body, headers, model)

    async def _handle_nonstream(
        self,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
        model: str,
    ) -> JSONResponse:
        client = await self.get_client()
        try:
            resp = await client.post(url, json=body, headers=headers)
            status = resp.status_code
            result = resp.json()
        except Exception as e:
            logger.error("Upstream request failed: %s", e)
            return JSONResponse(
                _anthropic_error(502, "api_error", f"Upstream request failed: {e}"),
                status_code=502,
            )

        if status >= 400:
            return JSONResponse(result, status_code=status)

        anthropic_resp = convert_openai_to_anthropic(result, model=model)
        return JSONResponse(anthropic_resp)

    async def _handle_stream(
        self,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
        model: str,
    ) -> JSONResponse | StreamingResponse:
        # Ensure stream is set in the body
        body["stream"] = True
        if "stream_options" not in body:
            body["stream_options"] = {"include_usage": True}

        client = await self.get_client()

        try:
            upstream_resp = await client.send(
                client.build_request("POST", url, json=body, headers=headers),
                stream=True,
            )
        except Exception as e:
            logger.error("Streaming connection failed: %s", e)
            return JSONResponse(
                _anthropic_error(502, "api_error", f"Upstream error: {e}"),
                status_code=502,
            )

        if upstream_resp.status_code >= 400:
            error_text = await upstream_resp.aread()
            await upstream_resp.aclose()
            try:
                error_body = json.loads(error_text)
            except json.JSONDecodeError:
                error_body = {"error": {"message": error_text.decode()}}

            error_msg = "Upstream error"
            err = error_body.get("error")
            if isinstance(err, dict):
                error_msg = err.get("message", error_msg)
            elif isinstance(err, str):
                error_msg = err

            async def error_stream() -> AsyncIterator[bytes]:
                error_event = json.dumps(
                    {
                        "type": "error",
                        "error": {"type": "api_error", "message": error_msg},
                    }
                )
                yield f"event: error\ndata: {error_event}\n\n".encode()

            return StreamingResponse(
                error_stream(),
                status_code=upstream_resp.status_code,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        async def generate() -> AsyncIterator[bytes]:
            try:

                async def iter_openai_chunks() -> AsyncIterator[dict]:
                    async for line in upstream_resp.aiter_lines():
                        line_str = line.strip()
                        if not line_str:
                            continue
                        if line_str.startswith("data: "):
                            data = line_str[6:]
                            if data.strip() == "[DONE]":
                                return
                            try:
                                chunk = json.loads(data)
                                yield chunk
                            except json.JSONDecodeError:
                                continue

                async for sse_event in async_convert_openai_stream_to_anthropic_sse(
                    iter_openai_chunks(), model=model
                ):
                    yield sse_event.encode("utf-8")
            except Exception as e:
                logger.error("Streaming error: %s", e, exc_info=True)
                error_event = json.dumps(
                    {
                        "type": "error",
                        "error": {"type": "api_error", "message": f"Upstream error: {e}"},
                    }
                )
                yield f"event: error\ndata: {error_event}\n\n".encode()
            finally:
                await upstream_resp.aclose()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config: Config) -> FastAPI:
    """Create the FastAPI application with routes."""
    handler = AnthropicMessageHandler(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ARG001
        yield
        await handler.close()

    app = FastAPI(lifespan=lifespan)

    app.post("/v1/messages", response_model=None)(handler.handle_messages)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    # Store handler for test access
    app.state.handler = handler

    return app


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------


_ENV_PREFIX = "A2O_"


def _config_to_env(config: Config) -> dict[str, str]:
    """Serialize config fields to environment variables."""
    import os

    env = os.environ.copy()
    for field_name in config.__dataclass_fields__:
        val = getattr(config, field_name)
        env[f"{_ENV_PREFIX}{field_name.upper()}"] = "" if val is None else str(val)
    return env


def create_app_from_env() -> FastAPI:
    """Factory called by uvicorn workers -- reads config from env vars."""
    import dataclasses
    import os
    import typing

    hints = typing.get_type_hints(Config)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(Config):
        env_val = os.environ.get(f"{_ENV_PREFIX}{f.name.upper()}")
        if env_val is None:
            continue
        if env_val == "":
            kwargs[f.name] = None
        else:
            t = hints.get(f.name)
            is_int = t is int or (hasattr(t, "__args__") and int in t.__args__)
            kwargs[f.name] = int(env_val) if is_int else env_val
    return create_app(Config(**kwargs))


def run_server(config: Config) -> None:
    """Run the proxy server with uvicorn (multi-worker support built-in)."""
    import os

    import uvicorn

    logger.info(
        "Starting server on %s:%s -> %s (workers=%d)",
        config.host,
        config.port,
        config.openai_base_url,
        config.workers,
    )

    if config.workers > 1:
        # Multi-worker: pass app as import string so uvicorn can fork
        os.environ.update(_config_to_env(config))
        uvicorn.run(
            "a2o.server:create_app_from_env",
            factory=True,
            host=config.host,
            port=config.port,
            workers=config.workers,
            log_level="info",
        )
    else:
        uvicorn.run(
            create_app(config),
            host=config.host,
            port=config.port,
            log_level="info",
        )
