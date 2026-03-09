FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Cloud Run sets PORT env var (defaults to 8080)
# For local dev, override with -p 8000:8000 or set PORT=8000
ENV PORT=8080

EXPOSE ${PORT}

# Production: no --reload, use workers based on CPU
# Cloud Run provides 1-2 vCPUs typically, so 2 workers is safe.
# For local dev, override CMD with: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 2