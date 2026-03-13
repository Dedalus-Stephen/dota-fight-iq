import os, threading
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import worker  # imports worker.py from same directory

app = FastAPI()


class ParseRequest(BaseModel):
    match_id: int
    replay_url: str
    job_id: str


@app.post("/parse")
def parse_endpoint(req: ParseRequest):
    os.environ["MATCH_ID"]   = str(req.match_id)
    os.environ["REPLAY_URL"] = req.replay_url
    os.environ["JOB_ID"]     = req.job_id
    worker.main()  # blocks until done — Cloud Tasks waits up to 30 min by default
    return {"status": "complete", "job_id": req.job_id}


@app.get("/health")
def health():
    return {"status": "ok"}