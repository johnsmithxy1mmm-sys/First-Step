"""Two-way Telegram: control the bot from your phone, no shell needed.

A background thread long-polls getUpdates and dispatches slash commands to
handlers the bot supplies (/status, /positions, /pnl, /pause, /resume, /help).
Security: only messages from the configured TELEGRAM_CHAT_ID are honored —
anyone else who finds the bot is ignored. Read-only by default; the only
state-changing commands are pause/resume of the shared kill-switch, which can
only ever make the bot SAFER (stop trading), never place an order.

Dispatch (handle_update) is pure and network-free, so it is unit-tested; the
run loop is a thin getUpdates wrapper around it.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable

import httpx

log = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramControl(threading.Thread):
    daemon = True

    def __init__(self, handlers: dict[str, Callable[[], str]],
                 poll_timeout: int = 25):
        super().__init__(name="tg-control")
        self._handlers = handlers
        self._poll_timeout = poll_timeout
        self._token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        self._offset = 0
        self._halt = threading.Event()

    @property
    def enabled(self) -> bool:
        return bool(self._token and self._chat_id)

    # --- pure dispatch (no network) ---

    def handle_update(self, update: dict) -> str | None:
        """One getUpdates item -> reply text (or None to stay silent).

        Ignores anything not from the configured chat, and anything that is not
        a known slash command.
        """
        msg = update.get("message") or update.get("edited_message") or {}
        chat = str((msg.get("chat") or {}).get("id", ""))
        if not chat or chat != str(self._chat_id):
            return None                       # not our owner — ignore silently
        text = (msg.get("text") or "").strip()
        if not text.startswith("/"):
            return None
        cmd = text[1:].split()[0].split("@")[0].lower()   # "/status@bot foo" -> "status"
        handler = self._handlers.get(cmd)
        if handler is None:
            known = ", ".join("/" + k for k in sorted(self._handlers))
            return f"Unknown command. Try: {known}"
        try:
            return handler()
        except Exception as exc:            # a handler must never kill the poller
            log.exception("telegram command /%s", cmd)
            return f"/{cmd} failed: {exc}"

    def _advance_offset(self, updates: list[dict]) -> None:
        for u in updates:
            self._offset = max(self._offset, int(u.get("update_id", 0)) + 1)

    # --- network ---

    def _get_updates(self) -> list[dict]:
        try:
            resp = httpx.get(
                _API.format(token=self._token, method="getUpdates"),
                params={"offset": self._offset, "timeout": self._poll_timeout},
                timeout=self._poll_timeout + 10)
            if resp.status_code != 200:
                return []
            return resp.json().get("result", []) or []
        except httpx.HTTPError as exc:
            log.debug("telegram getUpdates: %s", exc)
            return []

    def _reply(self, text: str) -> None:
        try:
            httpx.post(_API.format(token=self._token, method="sendMessage"),
                       json={"chat_id": self._chat_id, "text": text[:4000]},
                       timeout=10)
        except httpx.HTTPError as exc:
            log.debug("telegram sendMessage: %s", exc)

    def run(self) -> None:  # pragma: no cover — network loop over tested parts
        if not self.enabled:
            return
        log.info("telegram control: listening for commands")
        while not self._halt.is_set():
            updates = self._get_updates()
            self._advance_offset(updates)
            for u in updates:
                reply = self.handle_update(u)
                if reply:
                    self._reply(reply)

    def stop(self) -> None:
        self._halt.set()
