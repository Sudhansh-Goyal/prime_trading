# ── Stage 1: base image ───────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /code

# Install OS-level deps (none needed beyond slim defaults for pandas/numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: dependencies ─────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 3: application ──────────────────────────────────────────────────
COPY datasets/ ./datasets/
COPY app/ ./app/

EXPOSE 8000

# Single-worker is fine; data is loaded once and cached in-process
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
