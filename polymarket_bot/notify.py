"""Уведомления в Telegram (опционально): бот сам сообщает о ставках и выигрышах.

Достаточно задать переменные окружения:
  TELEGRAM_BOT_TOKEN — токен бота от @BotFather
  TELEGRAM_CHAT_ID   — ID чата (узнать у @userinfobot)
Если переменные не заданы, уведомления тихо отключены.
"""

from __future__ import annotations

import os

import requests


def notify(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:4000]},
            timeout=10,
        )
        return resp.status_code == 200
    except requests.RequestException:
        return False  # уведомление не должно ронять торговый цикл
