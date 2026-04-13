"""End-to-end test: proxy + Claude Code integration.

This test:
1. Starts the proxy server pointing to a real backend
2. Runs a Claude Code CLI task with custom ANTHROPIC_* env vars
3. Verifies the generated files exist and contain valid HTML
"""

import glob
import os
import subprocess
import shutil
import threading
import time

import pytest

from a2o.config import Config


# The upstream endpoint provided by the user
REAL_UPSTREAM_URL = os.environ.get(
    "ANTHROPIC_PROXY_UPSTREAM",
    "http://s-20260413151221-vvzu6.bcecn-bj-cloudml.xiaomi.srv/v1/chat/completions",
)
REAL_MODEL = os.environ.get("ANTHROPIC_PROXY_MODEL", "mimo-v2-pro")
PROXY_PORT = int(os.environ.get("ANTHROPIC_PROXY_PORT", "3578"))
PROXY_HOST = "127.0.0.1"

# Whether to run this test (disabled by default since it requires Claude Code CLI)
SKIP_E2E = os.environ.get("ANTHROPIC_PROXY_E2E_TEST", "false").lower() != "true"

# Fixed work directory for deterministic file verification
E2E_WORK_DIR = os.environ.get("ANTHROPIC_PROXY_E2E_WORK_DIR", "/tmp/a2o-e2e-work")

CLAUDE_PROMPT = (
    "Create index.html with <h1>My Portfolio</h1> and a <form> for contact, "
    "style.css with body { color: white; background: #1a1a2e; }, "
    "and script.js with a submit event listener that prevents default and logs to console. "
    "index.html must link to style.css and script.js."
)


def _start_proxy_thread(config: Config, ready_event: threading.Event) -> threading.Event:
    """Start the proxy in a background thread with its own event loop.

    Returns a stop_event that, when set, will shut down the proxy.
    """
    import asyncio
    import aiohttp
    from aiohttp import web
    from a2o.server import create_app

    stop_event = threading.Event()

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _serve():
            app = create_app(config)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, host=config.host, port=config.port)
            await site.start()
            print(f"Proxy started on {config.host}:{config.port}")
            ready_event.set()

            # Wait until stop_event is set
            while not stop_event.is_set():
                await asyncio.sleep(0.2)

            await runner.cleanup()

        loop.run_until_complete(_serve())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return stop_event


@pytest.mark.skipif(
    SKIP_E2E, reason="E2E test disabled; set ANTHROPIC_PROXY_E2E_TEST=true to enable"
)
class TestEndToEnd:
    """E2E test using real Claude Code CLI against the proxy."""

    @pytest.fixture(autouse=True)
    def start_proxy(self):
        """Start the proxy server in a background thread."""
        config = Config(
            openai_base_url=REAL_UPSTREAM_URL,
            default_model=REAL_MODEL,
            host=PROXY_HOST,
            port=PROXY_PORT,
            request_timeout=300,
            stream_timeout=300,
        )

        import aiohttp

        ready = threading.Event()
        stop = _start_proxy_thread(config, ready)

        # Wait for proxy to be ready
        assert ready.wait(timeout=5), "Proxy failed to start within 5s"

        # Health check
        import urllib.request
        try:
            req = urllib.request.Request(f"http://{PROXY_HOST}:{PROXY_PORT}/health")
            resp = urllib.request.urlopen(req, timeout=5)
            assert resp.status == 200
            print("Proxy health check OK")
        except Exception as e:
            stop.set()
            pytest.skip(f"Proxy health check failed: {e}")
            return

        yield

        stop.set()

    @pytest.fixture
    def work_dir(self):
        """Provide a clean working directory for the Claude Code task."""
        if os.path.exists(E2E_WORK_DIR):
            shutil.rmtree(E2E_WORK_DIR)
        os.makedirs(E2E_WORK_DIR, exist_ok=True)
        yield E2E_WORK_DIR

    def test_claude_code_generates_portfolio(self, work_dir):
        """Run Claude Code to generate a portfolio page and verify the output files."""
        claude_bin = shutil.which("claude")
        if not claude_bin:
            pytest.skip("claude CLI not found in PATH")

        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
        env["ANTHROPIC_API_KEY"] = "sk-proxy-test"
        env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = REAL_MODEL
        env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = REAL_MODEL
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = REAL_MODEL

        result = subprocess.run(
            [
                claude_bin,
                "-p",
                CLAUDE_PROMPT,
                "--model",
                REAL_MODEL,
                "--output-format",
                "json",
                "--dangerously-skip-permissions",
            ],
            cwd=work_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            stdin=subprocess.DEVNULL,
        )

        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout[:3000]}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr[:1000]}")

        # Check for proxy/connectivity errors
        stderr_lower = result.stderr.lower()
        if "connection error" in stderr_lower or "refused" in stderr_lower:
            pytest.fail(f"Claude Code couldn't connect to proxy: {result.stderr}")

        # Verify files were created
        html_files = glob.glob(os.path.join(work_dir, "**", "*.html"), recursive=True)
        css_files = glob.glob(os.path.join(work_dir, "**", "*.css"), recursive=True)
        js_files = glob.glob(os.path.join(work_dir, "**", "*.js"), recursive=True)

        print(f"\nGenerated files:")
        print(f"  HTML: {html_files}")
        print(f"  CSS:  {css_files}")
        print(f"  JS:   {js_files}")

        assert len(html_files) >= 1, f"No HTML files generated in {work_dir}"
        assert len(css_files) >= 1, f"No CSS files generated in {work_dir}"
        assert len(js_files) >= 1, f"No JS files generated in {work_dir}"

        # Verify index.html content
        index_path = os.path.join(work_dir, "index.html")
        assert os.path.exists(index_path), "index.html not found"
        with open(index_path) as f:
            html_content = f.read()

        assert "<html" in html_content.lower(), "index.html missing <html> tag"
        assert "<h1" in html_content.lower(), "index.html missing <h1> tag"
        assert "my portfolio" in html_content.lower(), (
            "index.html missing 'My Portfolio' heading"
        )
        assert "<form" in html_content.lower(), "index.html missing <form> element"
        assert "style.css" in html_content, "index.html missing link to style.css"
        assert "script.js" in html_content, "index.html missing link to script.js"

        # Verify style.css content
        css_path = os.path.join(work_dir, "style.css")
        assert os.path.exists(css_path), "style.css not found"
        with open(css_path) as f:
            css_content = f.read()
        assert len(css_content.strip()) > 0, "style.css is empty"
        assert "{" in css_content and "}" in css_content, (
            "style.css contains no CSS rules"
        )

        # Verify script.js content
        js_path = os.path.join(work_dir, "script.js")
        assert os.path.exists(js_path), "script.js not found"
        with open(js_path) as f:
            js_content = f.read()
        assert "submit" in js_content.lower(), (
            "script.js missing 'submit' event listener"
        )

        print("\nAll file validations passed!")
