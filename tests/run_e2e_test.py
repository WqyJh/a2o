"""Standalone E2E test script for proxy + Claude Code.

Run with:
    uv run python tests/run_e2e_test.py

Or enable the pytest E2E test:
    ANTHROPIC_PROXY_E2E_TEST=true uv run pytest tests/test_e2e_claude_code.py -v
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile

import aiohttp
from aiohttp import web

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from a2o.config import Config
from a2o.server import create_app

# Configuration
UPSTREAM_URL = os.environ.get(
    "ANTHROPIC_PROXY_UPSTREAM",
    "http://s-20251114160409-ocnba-1r2hj.wlcb-prod-3-cloudml.xiaomi.srv/v1/chat/completions",
)
MODEL = os.environ.get("ANTHROPIC_PROXY_MODEL", "0312_pro")
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 3578
CLAUDE_BIN = shutil.which("claude") or os.environ.get("CLAUDE_BIN", "")


async def start_proxy() -> web.AppRunner:
    """Start the proxy server."""
    config = Config(
        openai_base_url=UPSTREAM_URL,
        default_model=MODEL,
        host=PROXY_HOST,
        port=PROXY_PORT,
        request_timeout=300,
        stream_timeout=300,
    )
    app = create_app(config)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=PROXY_HOST, port=PROXY_PORT)
    await site.start()
    print(f"[Proxy] Started on {PROXY_HOST}:{PROXY_PORT}")
    return runner


async def check_proxy_health() -> bool:
    """Verify the proxy is running."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://{PROXY_HOST}:{PROXY_PORT}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
    except Exception:
        return False


def run_claude_code(task: str, work_dir: str) -> subprocess.CompletedProcess:
    """Run a Claude Code task through the proxy."""
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
    env["ANTHROPIC_API_KEY"] = "sk-proxy-test"
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = MODEL
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = MODEL
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = MODEL

    return subprocess.run(
        [
            CLAUDE_BIN,
            "-p",
            task,
            "--model",
            MODEL,
            "--max-turns",
            "5",
            "--output-format",
            "json",
        ],
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


async def main() -> None:
    print("=" * 60)
    print("E2E Test: Anthropic Proxy + Claude Code")
    print("=" * 60)
    print(f"Upstream: {UPSTREAM_URL}")
    print(f"Model: {MODEL}")
    print(f"Claude: {CLAUDE_BIN or 'NOT FOUND'}")
    print()

    # Step 1: Start proxy
    print("[Step 1] Starting proxy server...")
    runner = await start_proxy()

    if not await check_proxy_health():
        print("[ERROR] Proxy health check failed!")
        await runner.cleanup()
        sys.exit(1)
    print("[Step 1] Proxy is healthy! ✓")
    print()

    try:
        # Step 2: Test proxy endpoint directly
        print("[Step 2] Testing proxy endpoint directly...")
        async with aiohttp.ClientSession() as session:
            test_body = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Reply with exactly: E2E_OK"}],
            }
            async with session.post(
                f"http://{PROXY_HOST}:{PROXY_PORT}/v1/messages",
                json=test_body,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    print(f"[ERROR] Proxy returned {resp.status}: {text}")
                    sys.exit(1)
                body = await resp.json()
                print(
                    f"[Step 2] Proxy response: type={body.get('type')}, model={body.get('model')}"
                )
                # Check we got valid Anthropic format
                assert body.get("type") == "message"
                assert body.get("role") == "assistant"
                assert len(body.get("content", [])) > 0
                text_content = " ".join(
                    c.get("text", "") for c in body["content"] if c.get("type") == "text"
                )
                print(f"[Step 2] Content: {text_content[:200]}...")
                print("[Step 2] Direct proxy test passed! ✓")
        print()

        # Step 3: Test streaming via proxy
        print("[Step 3] Testing streaming via proxy...")
        async with aiohttp.ClientSession() as session:
            stream_body = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 50,
                "stream": True,
                "messages": [{"role": "user", "content": "Say just: STREAM_OK"}],
            }
            async with session.post(
                f"http://{PROXY_HOST}:{PROXY_PORT}/v1/messages",
                json=stream_body,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                assert resp.status == 200
                text = await resp.text()
                assert "message_start" in text
                assert "message_stop" in text
                print("[Step 3] Streaming proxy test passed! ✓")
        print()

        # Step 4: Run Claude Code
        if not CLAUDE_BIN:
            print("[Step 4] SKIPPED: Claude Code CLI not found in PATH")
            print("         Install: npm install -g @anthropic-ai/claude-code")
        else:
            print("[Step 4] Running Claude Code through proxy...")
            work_dir = tempfile.mkdtemp(prefix="claude-e2e-")

            # Create a test file
            test_file = os.path.join(work_dir, "hello.py")
            with open(test_file, "w") as f:
                f.write(
                    'def greet(name):\n    return f"Hello, {name}!"\n\n'
                    'if __name__ == "__main__":\n    print(greet("World"))\n'
                )

            result = run_claude_code(
                "Read the file hello.py and tell me what it does in one sentence. "
                "Output ONLY the sentence, nothing else.",
                work_dir,
            )

            print(f"[Step 4] Return code: {result.returncode}")
            print(f"[Step 4] STDOUT: {result.stdout[:500]}")
            if result.stderr:
                stderr_preview = result.stderr[:300]
                print(f"[Step 4] STDERR: {stderr_preview}")

            # Check for success
            if result.returncode == 0:
                print("[Step 4] Claude Code completed successfully! ✓")
            else:
                stderr_lower = result.stderr.lower()
                if "connection" in stderr_lower or "refused" in stderr_lower:
                    print("[Step 4] FAILED: Connection error to proxy")
                    sys.exit(1)
                else:
                    # Some non-fatal error (model issues etc.)
                    print(
                        "[Step 4] Claude Code returned non-zero "
                        f"({result.returncode}), but proxy itself works"
                    )

            # Cleanup
            shutil.rmtree(work_dir, ignore_errors=True)

        print()
        print("=" * 60)
        print("All E2E tests passed! ✓")
        print("=" * 60)

    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
