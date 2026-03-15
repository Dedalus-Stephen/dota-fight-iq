#!/bin/bash
# ── Deploy Dota Fight IQ Backend to Cloud Run ──
set -euo pipefail

PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
SERVICE_NAME="dota-fight-iq"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "═══════════════════════════════════════════"
echo "  Deploying Backend to Cloud Run"
echo "  Project: ${PROJECT_ID}"
echo "  Region:  ${REGION}"
echo "═══════════════════════════════════════════"

# ── Build ──
echo ""
echo "▸ Building container image..."
gcloud builds submit --tag "${IMAGE}" .

# ── Deploy ──
echo ""
echo "▸ Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --no-cpu-throttling \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 300 \
  --set-env-vars "ENV=production,STORAGE_BACKEND=gcs,GCS_BUCKET=dota-fight-iq-data,GCP_PROJECT_ID=${PROJECT_ID},PARSER_URL=https://dota-parser-53745627806.us-central1.run.app" \
  --update-secrets "SUPABASE_URL=supabase-url:latest,SUPABASE_ANON_KEY=supabase-anon-key:latest,SUPABASE_SERVICE_KEY=supabase-service-key:latest,DATABASE_URL=database-url:latest,STRATZ_API_TOKEN=stratz-token:latest,OPENDOTA_API_KEY=opendota-key:latest,STEAM_USER=steam-user:latest,STEAM_PASS=steam-pass:latest"

# ── Verify ──
echo ""
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format "value(status.url)")
echo "═══════════════════════════════════════════"
echo "  ✓ Backend deployed!"
echo "  URL:    ${SERVICE_URL}"
echo "  Health: ${SERVICE_URL}/health"
echo "═══════════════════════════════════════════"