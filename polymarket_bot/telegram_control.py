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

        Every field is shape-checked before use. This is an EXTERNAL payload:
        the previous version called `.get()` on whatever sat in `message` and
        `chat`, and `.strip()` on whatever sat in `text`, so a non-object there
        raised AttributeError — and the caller treats a raised update as an
        unprocessable batch, which stalls the whole control channel.
        """
        if not isinstance(update, dict):
            return None
        msg = update.get("message") or update.get("edited_message") or {}
        if not isinstance(msg, dict):
            return None
        chat_obj = msg.get("chat")
        if not isinstance(chat_obj, dict):
            return None
        chat = str(chat_obj.get("id", ""))
        if not chat or chat != str(self._chat_id):
            return None                       # not our owner — ignore silently
        raw_text = msg.get("text")
        if not isinstance(raw_text, str):
            return None
        text = raw_text.strip()
        if not text.startswith("/"):
            return None
        parts = text[1:].split()
        if not parts:
            return None                   # a bare "/" is not a command
        cmd = parts[0].split("@")[0].lower()   # "/status@bot foo" -> "status"
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
        """Acknowledge a batch. MUST make forward progress, always.

        Telegram replays every update until the offset moves past it, so a
        single unreadable `update_id` used to wedge the control channel
        permanently: `int("abc")` raised, the run loop swallowed it, the offset
        stayed put, and the next poll returned the same poisoned batch — for
        ever. Commands after it in the batch were never dispatched, which means
        /pause, the operator's remote kill, became unreachable.

        Note this runs BEFORE the chat-id check in handle_update, so it must
        survive payloads from anyone, not just the owner.
        """
        if not updates:
            return
        highest: int | None = None
        for u in updates:
            raw = u.get("update_id") if isinstance(u, dict) else None
            try:
                update_id = int(raw)          # type: ignore[arg-type]
            except (TypeError, ValueError, OverflowError):
                # OverflowError is int(inf) — a float that survives every other
                # check. Caught by this module's own property test.
                log.warning("telegram: unreadable update_id %r — skipping it", raw)
                continue
            highest = update_id if highest is None else max(highest, update_id)
        if highest is not None:
            self._offset = max(self._offset, highest + 1)
        else:
            # Nothing in the batch was readable. Step past it anyway: replaying
            # an unreadable batch for ever is strictly worse than losing it.
            self._offset += 1
            log.error("telegram: no readable update_id in a batch of %d — "
                      "advancing the offset to keep control responsive",
                      len(updates))

    # --- network ---

    def _redact(self, exc: Exception) -> str:
        """httpx error strings can embed the URL — which contains the token."""
        return str(exc).replace(self._token, "***") if self._token else str(exc)

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
            log.debug("telegram getUpdates: %s", self._redact(exc))
            return []

    def _reply(self, text: str) -> None:
        try:
            httpx.post(_API.format(token=self._token, method="sendMessage"),
                       json={"chat_id": self._chat_id, "text": text[:4000]},
                       timeout=10)
        except httpx.HTTPError as exc:
            log.debug("telegram sendMessage: %s", self._redact(exc))

    def run(self) -> None:  # pragma: no cover — network loop over tested parts
        if not self.enabled:
            return
        log.info("telegram control: listening for commands")
        while not self._halt.is_set():
            try:
                updates = self._get_updates()
                self._advance_offset(updates)
                for u in updates:
                    reply = self.handle_update(u)
                    if reply:
                        self._reply(reply)
            except Exception:               # one bad update must not kill control
                log.exception("telegram control loop")

    def stop(self) -> None:
        self._halt.set()
