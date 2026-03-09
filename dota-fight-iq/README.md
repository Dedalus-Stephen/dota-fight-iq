# Dota Fight IQ

ML-Powered Teamfight Performance Analyzer — benchmarks your Dota 2 gameplay against 7000+ MMR players.

## Architecture

**Infrastructure:** Google Cloud Platform (Cloud Run, Cloud Storage, Cloud Scheduler)
**Backend:** Python FastAPI with XGBoost, DBSCAN, scikit-learn
**Frontend:** Next.js + Tailwind CSS v3 on Vercel
**Database:** Supabase (Postgres + pgvector + Auth + Realtime)
**Data Sources:** OpenDota REST API + STRATZ GraphQL API

## Quick Start

### 1. Prerequisites
- Python 3.12+
- Docker & Docker Compose
- Supabase account (free tier)
- GCP account (for production deployment)

### 2. Setup

```bash
# Clone and enter
cd dota-fight-iq

# Copy env template and fill in your credentials
cp .env.example .env
# Edit .env with your Supabase credentials + API tokens

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 3. Database Setup

1. Go to your Supabase Dashboard → SQL Editor
2. Copy the contents of `migrations/001_initial_schema.sql`
3. Run it

### 4. Run Locally

```bash
# Option A: Docker (recommended)
docker-compose up

# Option B: Direct
uvicorn app.main:app --reload --port 8000
```

Local development uses `STORAGE_BACKEND=local` — files are stored in `./data/`. No GCP credentials needed for local dev.

### 5. Start Collecting Data

```bash
# Discover high-MMR matches from STRATZ
python -m scripts.collect_matches --discover --count 20

# Fetch parsed match data from OpenDota
python -m scripts.collect_matches --fetch-pending --limit 20
```

### 6. Test It

```bash
# Analyze a specific match
curl -X POST http://localhost:8000/api/analyze/8120171790

# Check status
curl http://localhost:8000/api/analyze/8120171790/status

# Get fight data
curl http://localhost:8000/api/fights/8120171790
```

## Deploy to GCP (Cloud Run)

### One-time setup

```bash
# Install gcloud CLI: https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  cloudscheduler.googleapis.com \
  cloudtasks.googleapis.com \
  storage.googleapis.com \
  secretmanager.googleapis.com

# Create GCS bucket for match data + model artifacts
gsutil mb -l us-central1 gs://dota-fight-iq-data

# Store secrets in Secret Manager
echo -n 'https://your-project.supabase.co' | gcloud secrets create supabase-url --data-file=-
echo -n 'your-anon-key' | gcloud secrets create supabase-anon-key --data-file=-
echo -n 'your-service-key' | gcloud secrets create supabase-service-key --data-file=-
echo -n 'postgresql://...' | gcloud secrets create database-url --data-file=-
echo -n 'your-stratz-token' | gcloud secrets create stratz-token --data-file=-
echo -n 'your-opendota-key' | gcloud secrets create opendota-key --data-file=-
```

### Deploy

```bash
chmod +x deploy.sh setup_scheduler.sh

# Deploy the API to Cloud Run
./deploy.sh

# Set up cron jobs (data collection, benchmark refresh, model retraining)
./setup_scheduler.sh
```

Cloud Run scales to zero when idle — you pay nothing when nobody is using the API.

## Project Structure

```
dota-fight-iq/
├── app/
│   ├── main.py              # FastAPI application
│   ├── core/
│   │   ├── config.py         # Settings (env vars, GCP config)
│   │   ├── database.py       # Supabase client + DB operations
│   │   └── storage.py        # Storage abstraction (local / GCS)
│   ├── clients/
│   │   ├── opendota.py       # OpenDota API client
│   │   └── stratz.py         # STRATZ GraphQL client
│   ├── services/
│   │   └── match_processor.py # Core pipeline: fetch → extract → store
│   ├── api/                  # Route modules
│   ├── ml/                   # ML pipeline (XGBoost, DBSCAN, scoring)
│   ├── models/               # Pydantic models
│   └── schemas/              # Response schemas
├── migrations/
│   └── 001_initial_schema.sql
├── scripts/
│   ├── collect_matches.py    # Data collection pipeline
│   └── train_models.py       # ML training script
├── deploy.sh                 # Cloud Run deployment
├── setup_scheduler.sh        # Cloud Scheduler cron jobs
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `POST /api/analyze/{match_id}` | Submit match for analysis |
| `GET /api/analyze/{match_id}/status` | Check processing status |
| `GET /api/matches/{match_id}/overview` | Match stats |
| `GET /api/fights/{match_id}` | All teamfights |
| `GET /api/fights/{match_id}/{index}` | Fight deep-dive |
| `GET /api/fights/{match_id}/{index}/minimap` | Position data for fight replay |
| `GET /api/heroes/{hero_id}/benchmarks` | Hero benchmark data |
| `GET /health` | Health check |

## Cost

| Service | MVP (10 users) | Monthly Cost |
|---|---|---|
| Cloud Run (API) | Scale-to-zero | $0 |
| Cloud Storage | ~50-100GB | $2-3 |
| Cloud Scheduler | 3 cron jobs | $0 |
| Supabase | Free tier | $0 |
| Vercel | Free tier | $0 |
| **Total** | | **~$2-3/month** |