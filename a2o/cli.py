"""Command-line interface for the Anthropic proxy server."""

from __future__ import annotations

import argparse
import logging
import sys

from a2o.config import Config
from a2o.server import run_server


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="anthropic-proxy",
        description="Lightweight Anthropic→OpenAI API proxy for Claude Code integration.",
    )
    parser.add_argument(
        "--upstream",
        default="",
        help="OpenAI-compatible base URL (e.g. http://host/v1/chat/completions)",
    )
    parser.add_argument("--model", default="", help="Default model name override")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=3578, help="Bind port (default: 3578)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=1000,
        help="Max upstream connections (default: 1000)",
    )
    parser.add_argument(
        "--max-connections-per-host",
        type=int,
        default=500,
        help="Max upstream connections per host (default: 500)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    config = Config(
        openai_base_url=args.upstream,
        default_model=args.model,
        host=args.host,
        port=args.port,
        request_timeout=args.timeout,
        stream_timeout=args.timeout,
        workers=args.workers,
        max_connections=args.max_connections,
        max_connections_per_host=args.max_connections_per_host,
    )

    try:
        run_server(config)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
