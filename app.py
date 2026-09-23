"""Start the app and open it in your browser.

    python app.py              # http://localhost:8000
    python app.py --port 9000
    python app.py --no-browser

Everything a first-time user can get wrong is checked here, before the server
starts, so the failure is a sentence that says what to do rather than a stack
trace.
"""

from __future__ import annotations

import argparse
import importlib.util
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN_PYTHON = (3, 11)
REQUIRED = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "faiss": "faiss-cpu",
    "rank_bm25": "rank-bm25",
    "pydantic_settings": "pydantic-settings",
    "multipart": "python-multipart",
    "pypdf": "pypdf",
}


def fail(message: str) -> None:
    print(f"\n  {message}\n", file=sys.stderr)
    sys.exit(1)


def check_environment() -> None:
    if sys.version_info < MIN_PYTHON:
        fail(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is needed; this is "
            f"{sys.version_info.major}.{sys.version_info.minor}. "
            "Install a newer Python from https://www.python.org/downloads/"
        )

    missing = [pkg for module, pkg in REQUIRED.items() if importlib.util.find_spec(module) is None]
    if missing:
        fail(
            "Some packages are missing: " + ", ".join(missing) + "\n"
            "  Install everything with:\n\n"
            "      pip install -r requirements.txt"
        )

    has_fastembed = importlib.util.find_spec("fastembed") is not None
    has_st = importlib.util.find_spec("sentence_transformers") is not None
    if not (has_fastembed or has_st):
        fail(
            "No embedding engine is installed. Install the small default with:\n\n"
            "      pip install fastembed"
        )


def free_port(preferred: int) -> int:
    """The preferred port if it is free, otherwise the next one that is."""
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    fail(f"Ports {preferred}-{preferred + 19} are all in use. Pass another with --port.")
    return preferred  # unreachable


def announce_when_ready(url: str, open_browser: bool) -> None:
    """Wait for the server to answer, then say so - and open the browser.

    Opening it straight away gives a 'connection refused' page, because the
    first start downloads the search model and can take a minute.
    """
    for _ in range(600):  # up to five minutes on a slow connection
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=1):
                print(f"  Ready: {url}\n", flush=True)
                if open_browser:
                    webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.5)
    print("  Still not answering after five minutes - check the messages above.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about your own documents.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Use 0.0.0.0 to reach it from other devices on your network")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed logs")
    args = parser.parse_args()

    check_environment()
    sys.path.insert(0, str(ROOT / "src"))

    port = free_port(args.port)
    url = f"http://localhost:{port}"

    say = lambda text="": print(text, flush=True)  # noqa: E731
    say()
    say("  Ask your documents")
    say(f"  Starting on {url} ...  (Ctrl+C here to stop)")
    say("  The first start downloads a small search model (about 90 MB), so it can take a minute.")
    if not (ROOT / ".env").exists():
        say("  No .env file: answers will quote your documents directly.")
        say("  For written answers, copy .env.example to .env and add a free key.")
    say()

    threading.Thread(
        target=announce_when_ready, args=(url, not args.no_browser), daemon=True
    ).start()

    import logging

    import uvicorn

    level = "info" if args.verbose else "warning"
    # The app's own logs are for developers; someone who just wants to ask
    # questions should see the few lines above, not a dict of index stats.
    logging.getLogger("rag").setLevel(logging.INFO if args.verbose else logging.WARNING)
    uvicorn.run("rag.api:app", host=args.host, port=port, log_level=level)


if __name__ == "__main__":
    main()
