# ── Build stage ───────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir ".[train]"

# ── Runtime stage ─────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages and project source
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build/src /app/src

# Copy runtime artifacts
COPY configs/  /app/configs/
COPY models/   /app/models/

# Create output directories
RUN mkdir -p /app/data/{01_raw,02_interim,03_processed} \
             /app/results \
             /app/logs

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "src.models.pipeline"]
CMD ["--help"]
