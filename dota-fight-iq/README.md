# Dota Fight IQ

ML-Powered Teamfight Performance Analyzer — benchmarks your Dota 2 gameplay against 7000+ MMR players.

## Quick Start

### 1. Prerequisites
- Python 3.12+
- Docker & Docker Compose
- Supabase account (free tier)
- AWS account (S3 access)

### 2. Setup

```bash
# Clone and enter
cd dota-fight-iq

# Copy env template and fill in your credentials
cp .env.example .env
# Edit .env with your Supabase + AWS credentials

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

## Project Structure

```
dota-fight-iq/
├── app/
│   ├── main.py              # FastAPI application
│   ├── core/
│   │   ├── config.py         # Settings (env vars)
│   │   ├── database.py       # Supabase client + DB operations
│   │   └── s3.py             # S3 client for raw data + models
│   ├── clients/
│   │   ├── opendota.py       # OpenDota API client
│   │   └── stratz.py         # STRATZ GraphQL client
│   ├── services/
│   │   └── match_processor.py # Core pipeline: fetch → extract → store
│   ├── api/                  # Route modules (Phase 3)
│   ├── models/               # Pydantic models (Phase 2)
│   ├── schemas/              # Response schemas
│   └── workers/              # Celery tasks (Phase 2)
├── migrations/
│   └── 001_initial_schema.sql
├── scripts/
│   └── collect_matches.py    # Data collection pipeline
├── tests/
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
