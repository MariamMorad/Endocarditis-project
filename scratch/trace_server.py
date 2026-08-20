"""
trace_server.py  —  Polls the deployed Azure Container App /health endpoint
until the RAG index finishes chunking and is fully ready.

Usage:
    python scratch/trace_server.py
    python scratch/trace_server.py --url https://your-custom-url.azurecontainerapps.io
"""
import sys
import time
import argparse
import urllib.request
import json

# ── Default: your Azure Container App URL ────────────────────────────────────
DEFAULT_URL = "https://rag-backend-api.livelywater-3e0acde8.eastus.azurecontainerapps.io"
HEALTH_PATH = "/api/v1/health"
POLL_INTERVAL_SECONDS = 5
MAX_WAIT_SECONDS = 600  # 10 minutes


def poll_health(base_url: str) -> None:
    url = base_url.rstrip("/") + HEALTH_PATH
    print(f"🔍 Tracing server: {url}")
    print(f"   Polling every {POLL_INTERVAL_SECONDS}s (timeout {MAX_WAIT_SECONDS}s)\n")

    elapsed = 0
    attempt = 0

    while elapsed < MAX_WAIT_SECONDS:
        attempt += 1
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            status       = data.get("status", "unknown")
            index_ready  = data.get("index_ready", False)
            total_chunks = data.get("total_chunks", 0)

            # Progress bar chars
            bar_fill = "█" * min(40, int(elapsed / MAX_WAIT_SECONDS * 40))
            bar_empty = "░" * (40 - len(bar_fill))

            if index_ready:
                print(f"\r✅  [{bar_fill}{bar_empty}] READY — {total_chunks} chunks loaded in {elapsed}s   ")
                print(f"\n🎉 Index is fully ready! ({total_chunks} chunks)")
                return
            else:
                symbol = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"][attempt % 10]
                print(
                    f"\r{symbol}  [{bar_fill}{bar_empty}] Building... "
                    f"status={status} | chunks_loaded={total_chunks} | elapsed={elapsed}s   ",
                    end="",
                    flush=True,
                )

        except urllib.error.HTTPError as e:
            print(f"\r⚠️  HTTP {e.code} from server (attempt {attempt}, elapsed {elapsed}s)   ",
                  end="", flush=True)
        except Exception as e:
            print(f"\r❌  Connection error: {e} (attempt {attempt}, elapsed {elapsed}s)   ",
                  end="", flush=True)

        time.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS

    print(f"\n\n⏱️  Timed out after {MAX_WAIT_SECONDS}s — index still not ready.")
    print("   Check container logs: az containerapp logs show --name rag-backend-api --resource-group depi_demo --follow")
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poll EndoAI server until index is ready")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the deployed API")
    args = parser.parse_args()
    poll_health(args.url)
