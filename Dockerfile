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
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:9090/health', timeout=3).status==200 else 1)" || exit 1

ENTRYPOINT ["python", "-m", "polymarket_bot"]
CMD ["--mode", "paper"]
