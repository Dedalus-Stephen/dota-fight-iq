import os, subprocess, json, sys, tempfile
import httpx
from google.cloud import storage
from supabase import create_client


def update_job_status(sb, job_id: str, status: str, error: str | None = None):
    payload = {"status": status}
    if error:
        payload["error"] = error[:500]
    if status in ("complete", "failed"):
        from datetime import datetime, timezone
        payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    sb.table("parse_jobs").update(payload).eq("job_id", job_id).execute()


def main():
    # Read env vars inside main() — NOT at module level
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
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_path = f"{tmpdir}/{match_id}.dem.bz2"
            dem_path    = f"{tmpdir}/{match_id}.dem"

            # 1. Download replay from Valve CDN
            print(f"Downloading replay for match {match_id}...")
            with httpx.stream("GET", replay_url, follow_redirects=True, timeout=120) as r:
                r.raise_for_status()
                with open(replay_path, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=65536):
                        f.write(chunk)

            # 2. Decompress
            subprocess.run(["bunzip2", "-k", replay_path], check=True)

            # 3. Parse with odota/parser JAR
            print("Parsing replay...")
            result = subprocess.run(
                ["java", "-jar", "/app/parser.jar", dem_path],
                capture_output=True, text=True, timeout=600 # <--- This might be too short
            )
            if result.returncode != 0:
                raise RuntimeError(f"Parser JAR failed: {result.stderr[:500]}")

            # 4. Assemble NDJSON into structured match object
            events = [
                json.loads(line)
                for line in result.stdout.strip().splitlines()
                if line
            ]
            parsed = assemble_parsed_match(events)

            # 5. Upload to GCS
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


def assemble_parsed_match(events: list[dict]) -> dict:
    result = {"teamfights": [], "players": [], "objectives": [], "chat": []}
    for event in events:
        t = event.get("type")
        if t == "teamfight":
            result["teamfights"].append(event)
        elif t == "objectives":
            result["objectives"].append(event)
        elif t == "chat":
            result["chat"].append(event)
    return result


if __name__ == "__main__":
    main()