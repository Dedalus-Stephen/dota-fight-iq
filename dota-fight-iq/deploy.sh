#!/bin/bash
# ── Deploy Dota Fight IQ to Cloud Run ──
#
# Prerequisites:
#   1. gcloud CLI installed and authenticated: `gcloud auth login`
#   2. Project set: `gcloud config set project YOUR_PROJECT_ID`
#   3. APIs enabled (run once):
#        gcloud services enable \
#          run.googleapis.com \
#          cloudbuild.googleapis.com \
#          cloudscheduler.googleapis.com \
#          cloudtasks.googleapis.com \
#          storage.googleapis.com
#   4. GCS bucket created (run once):
#        gsutil mb -l us-central1 gs://dota-fight-iq-data
#   5. .env file configured with Supabase + API credentials
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh

set -euo pipefail

# ── Configuration ──
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
SERVICE_NAME="dota-fight-iq"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "═══════════════════════════════════════════"
echo "  Deploying Dota Fight IQ to Cloud Run"
echo "  Project: ${PROJECT_ID}"
echo "  Region:  ${REGION}"
echo "═══════════════════════════════════════════"

# ── Step 1: Build and push container image ──
echo ""
echo "▸ Building container image..."
gcloud builds submit --tag "${IMAGE}" .

# ── Step 2: Deploy to Cloud Run ──
echo ""
echo "▸ Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 300 \
  --set-env-vars "ENV=production,STORAGE_BACKEND=gcs,GCS_BUCKET=dota-fight-iq-data,GCP_PROJECT_ID=${PROJECT_ID}" \
  --update-secrets "SUPABASE_URL=supabase-url:latest,SUPABASE_ANON_KEY=supabase-anon-key:latest,SUPABASE_SERVICE_KEY=supabase-service-key:latest,DATABASE_URL=database-url:latest,STRATZ_API_TOKEN=stratz-token:latest,OPENDOTA_API_KEY=opendota-key:latest"

# ── Step 3: Get the deployed URL ──
echo ""
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format "value(status.url)")
echo "═══════════════════════════════════════════"
echo "  ✓ Deployed successfully!"
echo "  URL: ${SERVICE_URL}"
echo "  Health: ${SERVICE_URL}/health"
echo "═══════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Store secrets in Secret Manager (if not already done):"
echo "     gcloud secrets create supabase-url --data-file=- <<< 'https://your-project.supabase.co'"
echo "     gcloud secrets create supabase-anon-key --data-file=- <<< 'your-anon-key'"
echo "     gcloud secrets create supabase-service-key --data-file=- <<< 'your-service-key'"
echo "     gcloud secrets create database-url --data-file=- <<< 'postgresql://...'"
echo "     gcloud secrets create stratz-token --data-file=- <<< 'your-stratz-token'"
echo "     gcloud secrets create opendota-key --data-file=- <<< 'your-opendota-key'"
echo ""
echo "  2. Grant Secret Manager access to Cloud Run service account:"
echo "     gcloud secrets add-iam-policy-binding supabase-url \\"
echo "       --member='serviceAccount:${PROJECT_ID}@appspot.gserviceaccount.com' \\"
echo "       --role='roles/secretmanager.secretAccessor'"
echo "     (repeat for each secret)"
echo ""
echo "  3. Set up Cloud Scheduler for data collection (see setup_scheduler.sh)"