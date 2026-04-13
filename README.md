# a2o

[![PyPI version](https://img.shields.io/pypi/v/ant2oai.svg)](https://pypi.org/project/ant2oai/)
[![Python](https://img.shields.io/pypi/pyversions/ant2oai.svg)](https://pypi.org/project/ant2oai/)
[![CI](https://github.com/WqyJh/a2o/actions/workflows/ci.yml/badge.svg)](https://github.com/WqyJh/a2o/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Lightweight **Anthropic-to-OpenAI** API proxy. Run [Claude Code](https://docs.anthropic.com/en/docs/claude-code) with any OpenAI-compatible backend.

## Features

- Translates Anthropic `/v1/messages` requests to OpenAI `/v1/chat/completions` format
- Streaming (SSE) and non-streaming support
- Tool calling (function calling) with full round-trip conversion
- Extended thinking / reasoning content
- Image content (base64 and URL)
- Multi-worker deployment with connection pooling
- Minimal dependencies: FastAPI + httpx

## Installation

```bash
pip install ant2oai
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install ant2oai
```

## Quick Start

Start the proxy pointing to any OpenAI-compatible endpoint:

```bash
a2o --upstream http://your-openai-endpoint/v1/chat/completions --model your-model
```

Then configure Claude Code to use the proxy:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:3578
export ANTHROPIC_API_KEY=any-string   # passed through to upstream

claude
```

## Usage

```
usage: a2o [-h] [--upstream UPSTREAM] [--model MODEL] [--host HOST]
           [--port PORT] [--timeout TIMEOUT] [--debug]
           [--workers WORKERS] [--max-connections MAX_CONNECTIONS]
           [--max-connections-per-host MAX_CONNECTIONS_PER_HOST]

options:
  --upstream UPSTREAM   OpenAI-compatible base URL
  --model MODEL         Default model name override
  --host HOST           Bind host (default: 127.0.0.1)
  --port PORT           Bind port (default: 3578)
  --timeout TIMEOUT     Request timeout in seconds (default: 300)
  --debug               Enable debug logging
  --workers WORKERS     Number of worker processes (default: 1)
  --max-connections N   Max upstream connections (default: 1000)
  --max-connections-per-host N
                        Max per-host connections (default: 500)
```

### Production Deployment

For high-throughput scenarios:

```bash
a2o \
  --upstream http://your-endpoint/v1/chat/completions \
  --model your-model \
  --host 0.0.0.0 \
  --port 3578 \
  --workers 4 \
  --max-connections 2000
```

## How It Works

```
Claude Code ──► a2o proxy ──► OpenAI-compatible backend
  (Anthropic API)    (converts)    (OpenAI API)
```

1. Claude Code sends requests in Anthropic format to the proxy
2. The proxy converts requests to OpenAI chat completion format
3. The upstream response is converted back to Anthropic format
4. Claude Code receives a native-looking Anthropic response

### Conversion Details

| Anthropic | OpenAI |
|-----------|--------|
| `system` (string/blocks) | First `system` message |
| `messages[].content` blocks | `messages[].content` parts |
| `tool_use` content blocks | `tool_calls` on assistant messages |
| `tool_result` content blocks | `tool` role messages |
| `thinking` blocks | `reasoning_content` field |
| `max_tokens` | `max_tokens` |
| `stop_sequences` | `stop` |
| Streaming SSE events | Streaming SSE chunks |

## Development

```bash
# Clone and install
git clone https://github.com/WqyJh/a2o.git
cd a2o
uv sync --all-extras

# Run tests
uv run pytest

# Lint
uv run ruff check .
uv run ruff format --check .
```

## License

[MIT](LICENSE)
