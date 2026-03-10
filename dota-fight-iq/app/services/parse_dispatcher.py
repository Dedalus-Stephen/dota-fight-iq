import json, uuid
from google.cloud import tasks_v2
from app.core.config import get_settings

async def enqueue_parse_job(match_id: int, replay_url: str) -> str:
    """
    Enqueues a Cloud Tasks message that triggers the Cloud Run Job parser worker.
    Returns a job_id for status polling.
    """
    settings = get_settings()
    client = tasks_v2.CloudTasksClient()

    job_id = str(uuid.uuid4())
    queue_path = client.queue_path(
        settings.gcp_project_id,
        settings.gcp_region,         # e.g. "us-central1"
        "replay-parse-queue"
    )

    payload = json.dumps({
        "match_id": match_id,
        "replay_url": replay_url,
        "job_id": job_id,
    }).encode()

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{settings.parser_worker_url}/parse",  # your Cloud Run Job trigger endpoint
            "headers": {"Content-Type": "application/json"},
            "body": payload,
            "oidc_token": {
                "service_account_email": settings.cloud_run_sa_email,
            },
        }
    }

    response = client.create_task(request={"parent": queue_path, "task": task})
    return job_id