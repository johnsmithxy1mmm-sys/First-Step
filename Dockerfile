# Polymarket bot — container image.
# Default CMD runs paper mode (safe). Override for live:
#   docker run ... polymarket-bot --mode live --i-understand-the-risk
FROM python:3.11-slim

WORKDIR /app

# Install deps first for layer caching.
COPY polymarket_bot/requirements.txt polymarket_bot/requirements.txt
RUN pip install --no-cache-dir -r polymarket_bot/requirements.txt

COPY . .

# Persisted state (ledger, logs) lives here — mount a volume over it.
VOLUME ["/app/polymarket_bot/data"]

# Prometheus /metrics + /health (enable ops.metrics_enabled in config).
EXPOSE 9090
# Metrics disabled (connection refused) counts as healthy; a served /health
# reporting non-200 counts as unhealthy. Set ops.metrics_bind: "0.0.0.0" in the
# container config for the compose port map to work.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "deploy/healthcheck.py"]

ENTRYPOINT ["python", "-m", "polymarket_bot"]
CMD ["--mode", "paper"]
