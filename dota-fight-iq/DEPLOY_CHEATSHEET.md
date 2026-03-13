# Dota Fight IQ — Deploy Cheatsheet (PowerShell)

## Architecture

```
dota2iq.com → Firebase Hosting
  ├─ /api/**  → Cloud Run: dota-fight-iq      (FastAPI backend)
  └─ /**      → Cloud Run: dota-fight-iq-web   (Next.js frontend)
```

---

## Backend (FastAPI → `dota-fight-iq`)

**Repo:** `dota-fight-iq/`

```powershell
cd dota-fight-iq

$PROJECT_ID = gcloud config get-value project

# Build & push
gcloud builds submit --tag "gcr.io/$PROJECT_ID/dota-fight-iq" .

# Deploy
gcloud run deploy dota-fight-iq `
  --image "gcr.io/$PROJECT_ID/dota-fight-iq" `
  --region us-central1 `
  --platform managed `
  --allow-unauthenticated `
  --memory 512Mi --cpu 1 `
  --min-instances 0 --max-instances 3 `
  --timeout 300 `
  --set-env-vars "ENV=production,STORAGE_BACKEND=gcs,GCS_BUCKET=dota-fight-iq-data,GCP_PROJECT_ID=$PROJECT_ID" `
  --update-secrets "SUPABASE_URL=supabase-url:latest,SUPABASE_ANON_KEY=supabase-anon-key:latest,SUPABASE_SERVICE_KEY=supabase-service-key:latest,DATABASE_URL=database-url:latest,STRATZ_API_TOKEN=stratz-token:latest,OPENDOTA_API_KEY=opendota-key:latest"
```

**Verify:** `Invoke-RestMethod https://dota2iq.com/api/health`

---

## Frontend (Next.js → `dota-fight-iq-web`)

**Repo:** `dota-fight-iq-front/`

```powershell
cd dota-fight-iq-front

$PROJECT_ID = gcloud config get-value project

# Build & push
gcloud builds submit --tag "gcr.io/$PROJECT_ID/dota-fight-iq-web" .

# Deploy
gcloud run deploy dota-fight-iq-web `
  --image "gcr.io/$PROJECT_ID/dota-fight-iq-web" `
  --region us-central1 `
  --platform managed `
  --allow-unauthenticated `
  --memory 256Mi --cpu 1 `
  --min-instances 0 --max-instances 3
```

> Frontend needs a `Dockerfile` in the repo root (Node.js, `npm run build`, `npm start` on port 3000 or `$PORT`).

---

## Firebase Hosting (routing layer)

**Repo:** whichever repo has `firebase.json` + `.firebaserc`

```powershell
# Deploy routing config only (no code — just rewrites)
firebase deploy --only hosting
```

**`firebase.json`** routes:
- `/api/**` → Cloud Run `dota-fight-iq` (backend)
- `/**` → Cloud Run `dota-fight-iq-web` (frontend)

---

## Quick Reference

```powershell
# ── Deploy ──
cd dota-fight-iq;       gcloud builds submit --tag "gcr.io/$PROJECT_ID/dota-fight-iq" .
cd dota-fight-iq-front; gcloud builds submit --tag "gcr.io/$PROJECT_ID/dota-fight-iq-web" .

# ── Logs ──
gcloud run logs read dota-fight-iq     --limit=50 --region=us-central1
gcloud run logs read dota-fight-iq-web --limit=50 --region=us-central1

# ── Health ──
Invoke-RestMethod https://dota2iq.com/api/health
```

---

## Local Dev

```powershell
# Backend
cd dota-fight-iq
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd dota-fight-iq-front
npm run dev    # localhost:3000, proxies /api/* → localhost:8000
```

---

## Secrets (one-time setup)

All in GCP Secret Manager. To update a secret:

```powershell
"new-value" | gcloud secrets versions add SECRET_NAME --data-file=-
```

Secrets: `supabase-url`, `supabase-anon-key`, `supabase-service-key`, `database-url`, `stratz-token`, `opendota-key`

After updating secrets, redeploy backend for it to pick up new values (Cloud Run caches secrets at startup).
