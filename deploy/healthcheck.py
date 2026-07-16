"""Container healthcheck: healthy if /health returns 200 OR metrics are disabled.

Metrics off -> connection refused -> exit 0 (nothing to check).
Metrics on but /health broken or reporting a halt -> exit 1.
"""
import sys
import urllib.request

try:
    resp = urllib.request.urlopen("http://localhost:9090/health", timeout=3)
    sys.exit(0 if resp.status == 200 else 1)
except ConnectionRefusedError:
    sys.exit(0)
except Exception:
    sys.exit(1)
