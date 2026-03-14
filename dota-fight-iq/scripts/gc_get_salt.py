"""
GC Salt Retriever — queries Valve's Game Coordinator for match replay URL.

Usage: python scripts/gc_get_salt.py <match_id>
Output: JSON to stdout: {"match_id": ..., "cluster": ..., "replay_salt": ..., "replay_url": "..."}
Exit 0 on success, 1 on failure.

This runs as a subprocess because the steam/dota2 libraries use gevent
which conflicts with asyncio (FastAPI/uvicorn). The call takes ~2 seconds.

Rate limit: 100 requests per 24 hours per Steam account.
"""

import sys
import os
import json

# gevent monkey-patch MUST happen before any other imports
import gevent
from gevent import monkey
monkey.patch_all()

from steam.client import SteamClient
from dota2.client import Dota2Client


TIMEOUT_SECONDS = 15


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: gc_get_salt.py <match_id>"}))
        sys.exit(1)

    match_id = int(sys.argv[1])
    steam_user = os.environ.get("STEAM_USER", "")
    steam_pass = os.environ.get("STEAM_PASS", "")

    if not steam_user or not steam_pass:
        print(json.dumps({"error": "STEAM_USER and STEAM_PASS env vars required"}))
        sys.exit(1)

    client = SteamClient()
    dota = Dota2Client(client)
    result = {"match_id": match_id}
    got_response = gevent.event.Event()

    @client.on("logged_on")
    def on_logged_on():
        dota.launch()

    @client.on("error")
    def on_error(eresult):
        result["error"] = f"Steam login failed: EResult={eresult}"
        got_response.set()

    @dota.on("ready")
    def on_dota_ready():
        dota.request_match_details(match_id)

    @dota.on("match_details")
    def on_match_details(mid, eresult, match):
        if eresult != 1:
            result["error"] = f"GC error: eresult={eresult}"
        else:
            cluster = match.cluster
            salt = match.replay_salt
            result["cluster"] = cluster
            result["replay_salt"] = salt
            if cluster and salt:
                result["replay_url"] = (
                    f"http://replay{cluster}.valve.net/570/{match_id}_{salt}.dem.bz2"
                )
        got_response.set()

    # Connect and wait
    login_result = client.login(steam_user, steam_pass)

    if login_result != 1:  # EResult.OK
        # Login itself might fail but the error handler above may have fired
        if "error" not in result:
            result["error"] = f"Steam login returned EResult={login_result}"
        print(json.dumps(result))
        sys.exit(1)

    # Wait for GC response with timeout
    got_response.wait(timeout=TIMEOUT_SECONDS)

    try:
        client.disconnect()
    except Exception:
        pass

    if "error" in result:
        print(json.dumps(result))
        sys.exit(1)

    if "replay_url" not in result:
        result["error"] = "No cluster/salt returned by GC"
        print(json.dumps(result))
        sys.exit(1)

    print(json.dumps(result))
    sys.exit(0)


if __name__ == "__main__":
    main()