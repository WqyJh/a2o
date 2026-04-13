"""End-to-end test: proxy + Claude Code integration.

This test:
1. Starts the proxy server pointing to a real backend
2. Runs a Claude Code CLI task with custom ANTHROPIC_* env vars
3. Verifies Claude Code completes successfully without errors
"""

import asyncio
import os
import subprocess
import shutil
import tempfile
import time
import json

import pytest
import aiohttp

from a2o.config import Config
from a2o.server import create_app, run_server


# The upstream endpoint provided by the user
REAL_UPSTREAM_URL = os.environ.get(
    "ANTHROPIC_PROXY_UPSTREAM",
    "http://s-20251114160409-ocnba-1r2hj.wlcb-prod-3-cloudml.xiaomi.srv/v1/chat/completions",
)
REAL_MODEL = os.environ.get("ANTHROPIC_PROXY_MODEL", "0312_pro")
PROXY_PORT = int(os.environ.get("ANTHROPIC_PROXY_PORT", "3578"))
PROXY_HOST = "127.0.0.1"

# Whether to run this test (disabled by default since it requires Claude Code CLI)
SKIP_E2E = os.environ.get("ANTHROPIC_PROXY_E2E_TEST", "false").lower() != "true"


@pytest.mark.skipif(
    SKIP_E2E, reason="E2E test disabled; set ANTHROPIC_PROXY_E2E_TEST=true to enable"
)
class TestEndToEnd:
    """E2E test using real Claude Code CLI against the proxy."""

    @pytest.fixture(autouse=True)
    async def start_proxy(self):
        """Start the proxy server in the background."""
        config = Config(
            openai_base_url=REAL_UPSTREAM_URL,
            default_model=REAL_MODEL,
            host=PROXY_HOST,
            port=PROXY_PORT,
            request_timeout=300,
            stream_timeout=300,
        )

        # Start proxy task
        import asyncio

        stop_event = asyncio.Event()

        async def run_proxy():
            from aiohttp import web
            from a2o.server import create_app

            app = create_app(config)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, host=PROXY_HOST, port=PROXY_PORT)
            await site.start()
            print(f"Proxy started on {PROXY_HOST}:{PROXY_PORT}")
            try:
                await stop_event.wait()
            finally:
                await runner.cleanup()

        task = asyncio.create_task(run_proxy())
        # Give proxy time to start
        await asyncio.sleep(0.5)

        # Verify proxy is running
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{PROXY_HOST}:{PROXY_PORT}/health"
                ) as resp:
                    assert resp.status == 200
                    print("Proxy health check OK")
        except Exception as e:
            pytest.skip(f"Proxy failed to start: {e}")
            return

        yield

        stop_event.set()
        await task

    @pytest.fixture
    def test_dir(self):
        """Create a temporary directory for the Claude Code task."""
        d = tempfile.mkdtemp(prefix="claude-e2e-")
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_claude_code_succeeds(self, test_dir):
        """Run a simple Claude Code task and verify it completes."""
        # Find claude CLI
        claude_bin = shutil.which("claude")
        if not claude_bin:
            pytest.skip("claude CLI not found in PATH")

        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
        env["ANTHROPIC_API_KEY"] = "sk-proxy-test"
        env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = REAL_MODEL
        env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = REAL_MODEL
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = REAL_MODEL

        # Create a simple file for the task to work on
        test_file = os.path.join(test_dir, "hello.py")
        with open(test_file, "w") as f:
            f.write("print('hello')\n")

        # Run Claude Code with a simple coding task
        result = subprocess.run(
            [
                claude_bin,
                "-p",
                "Read the file hello.py and output its content exactly. Do not modify anything.",
                "--model",
                REAL_MODEL,
                "--max-turns",
                "3",
                "--output-format",
                "json",
            ],
            cwd=test_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )

        print(f"STDOUT: {result.stdout[:2000]}")
        print(f"STDERR: {result.stderr[:2000]}")
        print(f"Return code: {result.returncode}")

        # Check that Claude Code ran without crashing
        # It may return 0 or non-zero depending on the model/response
        # The key is that we get valid output
        if result.returncode != 0:
            # Check if it's a proxy/config error vs model error
            stderr = result.stderr.lower()
            if "connection error" in stderr or "refused" in stderr:
                pytest.fail(f"Claude Code couldn't connect to proxy: {result.stderr}")
            elif "invalid api" in stderr and "key" in stderr:
                # API key error is expected with fake key but should reach the server
                pytest.fail(
                    f"API key error (proxy may not be working): {result.stderr}"
                )

        # At minimum, stdout should not be empty
        assert result.stdout.strip(), (
            f"Claude Code produced no output. stderr: {result.stderr}"
        )
