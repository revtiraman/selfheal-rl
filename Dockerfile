# SelfHealRL — OpenEnv Docker image
# Build:  docker build -t selfheal-rl .
# Run:    docker run -p 8000:8000 selfheal-rl

FROM python:3.10.14-slim-bullseye

# ── System deps ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Python deps ───────────────────────────────────────────────
COPY requirements.txt .

# Install CPU-only torch (much smaller than default GPU build)
# Try pytorch CDN first; fall back to PyPI if CDN is unavailable
RUN pip install --no-cache-dir --retries 5 torch \
        --index-url https://download.pytorch.org/whl/cpu \
    || pip install --no-cache-dir --retries 5 torch

# Install remaining deps
RUN pip install --no-cache-dir --retries 5 -r requirements.txt

# ── Copy project files ────────────────────────────────────────
COPY config.py .
COPY models.py .
COPY openenv.yaml .
COPY pyproject.toml .
COPY run.py .
COPY core/     core/
COPY env/      env/
COPY server/   server/
COPY training/ training/
COPY models/   models/

# ── Environment ───────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# ── Expose port ───────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────
HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
