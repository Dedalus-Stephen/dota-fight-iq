"""
Parser Worker — downloads and parses Dota 2 replays via odota/parser's /blob endpoint.

The /blob endpoint handles everything: download, decompress, parse, and assemble
into the same structured JSON that OpenDota's API returns (teamfights, players, etc.)

Architecture:
  - Cloud Tasks sends {match_id, replay_url, job_id} to POST /parse
  - This worker forwards replay_url to the odota/parser container's /blob endpoint
  - The assembled JSON is uploaded to GCS
  - parse_jobs table is updated for frontend status polling
"""

import os, json, sys
from datetime import datetime, timezone
import httpx
from google.cloud import storage
from supabase import create_client


# The odota/parser container runs as a sidecar or separate service
# In production on Cloud Run, this would be an internal service URL
PARSER_URL = os.environ.get("PARSER_URL", "http://localhost:5600")
PARSER_TIMEOUT = int(os.environ.get("PARSER_TIMEOUT", "300"))


def update_job_status(sb, job_id: str, status: str, error: str | None = None):
    payload = {"status": status}
    if error:
        payload["error"] = error[:500]
    if status in ("complete", "failed"):
        payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    sb.table("parse_jobs").update(payload).eq("job_id", job_id).execute()


def main():
    match_id     = int(os.environ["MATCH_ID"])
    replay_url   = os.environ["REPLAY_URL"]
    job_id       = os.environ.get("JOB_ID", "")
    gcs_bucket   = os.environ["GCS_BUCKET"]
    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_SERVICE_KEY"]

    gcs_output_key = f"parsed/{match_id}.json"

    sb = create_client(supabase_url, supabase_key)
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(gcs_bucket)

    update_job_status(sb, job_id, "parsing")

    try:
        # Call the parser's /blob endpoint — it handles download, decompress, parse, assembly
        blob_url = f"{PARSER_URL}/blob?replay_url={replay_url}"
        print(f"Requesting parse for match {match_id}: {blob_url}")

        with httpx.Client(timeout=PARSER_TIMEOUT) as client:
            resp = client.get(blob_url)
            resp.raise_for_status()

        parsed = resp.json()

        if not parsed or not parsed.get("teamfights"):
            raise RuntimeError("Parser returned empty or invalid data")

        # Inject match_id (parser doesn't know it from the replay alone)
        parsed["match_id"] = match_id

        print(f"Parse complete: {len(parsed.get('teamfights', []))} teamfights, "
              f"{len(parsed.get('players', []))} players")

        # Upload to GCS
        blob = bucket.blob(gcs_output_key)
        blob.upload_from_string(
            json.dumps(parsed),
            content_type="application/json"
        )
        print(f"Uploaded to gs://{gcs_bucket}/{gcs_output_key}")

        update_job_status(sb, job_id, "complete")

    except Exception as e:
        print(f"Parse failed: {e}", file=sys.stderr)
        update_job_status(sb, job_id, "failed", error=str(e))
        raise


if __name__ == "__main__":
    main()