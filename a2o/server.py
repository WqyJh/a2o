"""HTTP server that exposes the Anthropic Messages API and proxies to OpenAI."""

from __future__ import annotations

import json
import logging
from typing import Any

from aiohttp import web

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


def _json_response(data: Any, status: int = 200) -> web.Response:
    return web.json_response(data, status=status)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class AnthropicMessageHandler:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._client_session: Any = None

    async def get_client_session(self) -> Any:
        if self._client_session is None or self._client_session.closed:
            import aiohttp

            self._client_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
            )
        return self._client_session

    async def close(self) -> None:
        if self._client_session and not self._client_session.closed:
            await self._client_session.close()

    async def handle_messages(self, request: web.Request) -> web.Response:
        """Handle POST /v1/messages (Anthropic API format)."""
        try:
            raw_body = await request.json()
        except Exception:
            return _json_response(
                _anthropic_error(400, "invalid_request_error", "Invalid JSON body"),
                status=400,
            )

        # Parse Anthropic request
        try:
            anthropic_req = parse_anthropic_request(raw_body)
        except ParseError as e:
            return _json_response(
                _anthropic_error(400, "invalid_request_error", e.message),
                status=400,
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
            return _json_response(
                _anthropic_error(500, "api_error", "No upstream URL configured"),
                status=500,
            )

        # Forward incoming headers that might be useful
        headers = {
            "Content-Type": "application/json",
        }
        # Pass through Authorization header
        auth = request.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth

        if streaming:
            return await self._handle_stream(
                request, upstream_url, openai_body, headers, model
            )
        else:
            return await self._handle_nonstream(
                upstream_url, openai_body, headers, model
            )

    async def _handle_nonstream(
        self,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
        model: str,
    ) -> web.Response:
        session = await self.get_client_session()
        try:
            async with session.post(url, json=body, headers=headers) as resp:
                status = resp.status
                result = await resp.json()
        except Exception as e:
            logger.error("Upstream request failed: %s", e)
            return _json_response(
                _anthropic_error(502, "api_error", f"Upstream request failed: {e}"),
                status=502,
            )

        if status >= 400:
            return _json_response(result, status=status)

        anthropic_resp = convert_openai_to_anthropic(result, model=model)
        return _json_response(anthropic_resp)

    async def _handle_stream(
        self,
        request: web.Request,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
        model: str,
    ) -> web.StreamResponse:
        # Ensure stream is set in the body
        body["stream"] = True
        if "stream_options" not in body:
            body["stream_options"] = {"include_usage": True}

        session = await self.get_client_session()
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

        try:
            async with session.post(url, json=body, headers=headers) as upstream_resp:
                if upstream_resp.status >= 400:
                    error_text = await upstream_resp.text()
                    try:
                        error_body = json.loads(error_text)
                    except json.JSONDecodeError:
                        error_body = {"error": {"message": error_text}}

                    error_msg = "Upstream error"
                    err = error_body.get("error")
                    if isinstance(err, dict):
                        error_msg = err.get("message", error_msg)
                    elif isinstance(err, str):
                        error_msg = err

                    response.set_status(upstream_resp.status)
                    await response.prepare(request)
                    error_event = json.dumps(
                        {
                            "type": "error",
                            "error": {"type": "api_error", "message": error_msg},
                        }
                    )
                    await response.write(
                        f"event: error\ndata: {error_event}\n\n".encode()
                    )
                    await response.write_eof()
                    return response

                await response.prepare(request)

                async def iter_openai_chunks():
                    async for line in upstream_resp.content:
                        line_str = line.decode("utf-8", errors="replace").strip()
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
                    await response.write(sse_event.encode("utf-8"))

                await response.write_eof()
                return response

        except Exception as e:
            logger.error("Streaming error: %s", e, exc_info=True)
            error_event = json.dumps(
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": f"Upstream error: {e}"},
                }
            )
            try:
                await response.prepare(request)
                await response.write(f"event: error\ndata: {error_event}\n\n".encode())
                await response.write_eof()
            except Exception:
                pass
            return response


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config: Config) -> web.Application:
    """Create the aiohttp application with routes."""
    app = web.Application()
    handler = AnthropicMessageHandler(config)

    # Anthropic endpoint
    app.router.add_post("/v1/messages", handler.handle_messages)

    # Health check
    async def health(_request: web.Request) -> web.Response:
        return _json_response({"status": "ok"})

    app.router.add_get("/health", health)

    # Keep a reference for cleanup
    app["handler"] = handler
    app.on_shutdown.append(_on_shutdown)

    return app


async def _on_shutdown(app: web.Application) -> None:
    handler = app.get("handler")
    if isinstance(handler, AnthropicMessageHandler):
        await handler.close()


async def run_server(config: Config) -> None:
    """Run the proxy server."""
    app = create_app(config)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=config.host, port=config.port)
    logger.info(
        "Starting Anthropic proxy on %s:%s -> %s",
        config.host,
        config.port,
        config.openai_base_url,
    )
    await site.start()

    # Wait forever
    import asyncio

    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()
