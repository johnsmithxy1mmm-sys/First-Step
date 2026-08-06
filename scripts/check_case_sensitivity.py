"""One-off repro for OPEN-QUESTIONS C7: is the Info API case-sensitive on `user`?

Run from a network where api.hyperliquid.xyz is reachable:

    python scripts/check_case_sensitivity.py

Two different status codes confirm case sensitivity; two 200s refute it.
Report the result back into OPEN-QUESTIONS.md C7 either way.
"""

import json
import urllib.error
import urllib.request

ADDRESS = "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"  # any real account, checksummed


def post(user: str) -> int:
    body = json.dumps({"type": "clearinghouseState", "user": user}).encode()
    req = urllib.request.Request(
        "https://api.hyperliquid.xyz/info",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 — fixed https URL above
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


if __name__ == "__main__":
    for u in (ADDRESS, ADDRESS.lower()):
        print(post(u), u)
