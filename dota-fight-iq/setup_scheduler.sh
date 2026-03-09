#!/bin/bash
# ── Set up Cloud Scheduler cron jobs for Dota Fight IQ ──
#
# These replace EventBridge cron rules from the AWS spec.
# Each job hits an endpoint on your Cloud Run service, which triggers
# the corresponding background work.
#
# Prerequisites:
#   - Cloud Run service deployed (run deploy.sh first)
#   - Cloud Scheduler API enabled
#
# Usage:
#   chmod +x setup_scheduler.sh
#   ./setup_scheduler.sh

set -euo pipefail

PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
SERVICE_NAME="dota-fight-iq"

# Get the Cloud Run service URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region "${REGION}" \
  --format "value(status.url)")

echo "═══════════════════════════════════════════"
echo "  Setting up Cloud Scheduler jobs"
echo "  Target: ${SERVICE_URL}"
echo "═══════════════════════════════════════════"

# ── Job 1: Daily match discovery ──
# Runs at 03:00 UTC daily. Triggers STRATZ API to discover new Immortal-rank matches.
echo ""
echo "▸ Creating: collect-high-mmr-matches (daily 03:00 UTC)"
gcloud scheduler jobs create http collect-high-mmr-matches \
  --location="${REGION}" \
  --schedule="0 3 * * *" \
  --uri="${SERVICE_URL}/api/jobs/collect-matches" \
  --http-method=POST \
  --oidc-service-account-email="${PROJECT_ID}@appspot.gserviceaccount.com" \
  --oidc-token-audience="${SERVICE_URL}" \
  --time-zone="UTC" \
  --description="Daily: discover new Immortal-rank matches via STRATZ" \
  --attempt-deadline="600s" \
  2>/dev/null || \
gcloud scheduler jobs update http collect-high-mmr-matches \
  --location="${REGION}" \
  --schedule="0 3 * * *" \
  --uri="${SERVICE_URL}/api/jobs/collect-matches" \
  --http-method=POST \
  --oidc-service-account-email="${PROJECT_ID}@appspot.gserviceaccount.com" \
  --oidc-token-audience="${SERVICE_URL}"

# ── Job 2: Weekly benchmark refresh ──
# Runs at 04:00 UTC every Monday. Recomputes percentile benchmarks from latest data.
echo ""
echo "▸ Creating: refresh-benchmarks (weekly Monday 04:00 UTC)"
gcloud scheduler jobs create http refresh-benchmarks \
  --location="${REGION}" \
  --schedule="0 4 * * 1" \
  --uri="${SERVICE_URL}/api/jobs/refresh-benchmarks" \
  --http-method=POST \
  --oidc-service-account-email="${PROJECT_ID}@appspot.gserviceaccount.com" \
  --oidc-token-audience="${SERVICE_URL}" \
  --time-zone="UTC" \
  --description="Weekly: recompute hero benchmark percentiles" \
  --attempt-deadline="900s" \
  2>/dev/null || \
gcloud scheduler jobs update http refresh-benchmarks \
  --location="${REGION}" \
  --schedule="0 4 * * 1" \
  --uri="${SERVICE_URL}/api/jobs/refresh-benchmarks" \
  --http-method=POST \
  --oidc-service-account-email="${PROJECT_ID}@appspot.gserviceaccount.com" \
  --oidc-token-audience="${SERVICE_URL}"

# ── Job 3: Bi-weekly model retraining ──
# Runs at 05:00 UTC on the 1st and 15th of each month.
echo ""
echo "▸ Creating: retrain-models (bi-weekly, 1st & 15th at 05:00 UTC)"
gcloud scheduler jobs create http retrain-models \
  --location="${REGION}" \
  --schedule="0 5 1,15 * *" \
  --uri="${SERVICE_URL}/api/jobs/retrain-models" \
  --http-method=POST \
  --oidc-service-account-email="${PROJECT_ID}@appspot.gserviceaccount.com" \
  --oidc-token-audience="${SERVICE_URL}" \
  --time-zone="UTC" \
  --description="Bi-weekly: retrain XGBoost models and refresh DBSCAN clusters" \
  --attempt-deadline="1800s" \
  2>/dev/null || \
gcloud scheduler jobs update http retrain-models \
  --location="${REGION}" \
  --schedule="0 5 1,15 * *" \
  --uri="${SERVICE_URL}/api/jobs/retrain-models" \
  --http-method=POST \
  --oidc-service-account-email="${PROJECT_ID}@appspot.gserviceaccount.com" \
  --oidc-token-audience="${SERVICE_URL}"

echo ""
echo "═══════════════════════════════════════════"
echo "  ✓ All scheduler jobs configured!"
echo ""
echo "  View jobs:"
echo "    gcloud scheduler jobs list --location=${REGION}"
echo ""
echo "  Test a job manually:"
echo "    gcloud scheduler jobs run collect-high-mmr-matches --location=${REGION}"
echo "═══════════════════════════════════════════"