import httpx
import subprocess
import uuid
import json

# Your Cloud Run URL
URL = "https://dota-parser-worker-2nmdgyqovq-uc.a.run.app/parse"

test_payload = {
    "match_id": 8719764820, 
    "replay_url": "http://replay181.valve.net/570/8719764820_1585639439.dem.bz2",
    "job_id": "00000000-0000-0000-0000-000000000000" # Use the ID we just inserted
}

def get_gcloud_id_token():
    """Shells out to gcloud to get an ID token."""
    try:
        # Added shell=True for Windows compatibility with .cmd files
        token = subprocess.check_output(
            ["gcloud", "auth", "print-identity-token"], 
            text=True,
            shell=True 
        ).strip()
        return token
    except subprocess.CalledProcessError as e:
        print("Error: Make sure you have run 'gcloud auth login'")
        raise e

def trigger():
    print("Fetching ID token from gcloud...")
    token = get_gcloud_id_token()
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    print(f"Sending request to {URL}...")
    # Cloud Run might take a second to spin up if it's at 0 instances
    with httpx.Client(timeout=600) as client:
        response = client.post(URL, json=test_payload, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {response.json()}")
    except:
        print(f"Raw Response: {response.text}")

if __name__ == "__main__":
    trigger()