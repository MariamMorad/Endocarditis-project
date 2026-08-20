import sys
import time
import argparse
import urllib.request
import json

# Force UTF-8 on Windows to avoid cp1252 errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_URL = "https://rag-backend-api.livelywater-3e0acde8.eastus.azurecontainerapps.io"
HEALTH_PATH = "/api/v1/health"
POLL_INTERVAL = 5       # seconds between polls
MAX_WAIT = 600          # 10-minute timeout


def poll_health(base_url: str) -> None:
    url = base_url.rstrip("/") + HEALTH_PATH
    print(f"[TRACE] Polling: {url}")
    print(f"        Every {POLL_INTERVAL}s | Timeout {MAX_WAIT}s\n")

    elapsed = 0
    attempt = 0

    while elapsed < MAX_WAIT:
        attempt += 1
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            status       = data.get("status", "unknown")
            index_ready  = data.get("index_ready", False)
            total_chunks = data.get("total_chunks", 0)

            filled = "#" * min(40, int(elapsed / MAX_WAIT * 40))
            empty  = "-" * (40 - len(filled))

            if index_ready:
                print(f"\r[DONE] [{filled}{empty}] READY | {total_chunks} chunks | {elapsed}s   ")
                print(f"\n[OK] Index is fully ready! ({total_chunks} chunks loaded)")
                return
            else:
                spin = ["|", "/", "-", "\\"][attempt % 4]
                print(
                    f"\r[{spin}]  [{filled}{empty}] Building..."
                    f"  status={status} | chunks={total_chunks} | {elapsed}s elapsed   ",
                    end="", flush=True,
                )

        except urllib.error.HTTPError as e:
            print(f"\r[WARN] HTTP {e.code} (attempt {attempt}, {elapsed}s)   ", end="", flush=True)
        except Exception as e:
            print(f"\r[ERR]  {e} (attempt {attempt}, {elapsed}s)   ", end="", flush=True)

        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    print(f"\n\n[TIMEOUT] Index not ready after {MAX_WAIT}s.")
    print("  az containerapp logs show --name rag-backend-api --resource-group depi_demo --follow")
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poll EndoAI server until RAG index is ready")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the deployed API")
    args = parser.parse_args()
    poll_health(args.url)
