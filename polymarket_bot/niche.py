"""Стратегия №5: информационный edge в нише.

Исторически частные плюсы на prediction-рынках — это знание узкого домена
глубже толпы. Автоматизировать чужую экспертизу нельзя, но можно убрать
задержку между «появился рынок в моей нише» и «я его увидел»: модуль следит
за новыми рынками по нишевым вотчлистам и немедленно алертит — дальше
решает человек, который читает первоисточники быстрее западной толпы.

Стратегия №4 (rules lawyering) поддерживается здесь же: алерт включает
правила резолюции, чтобы расхождение заголовка и буквы правил было видно
сразу.
"""

from __future__ import annotations

import logging
import re

from .config import BotConfig
from .ledger import Ledger
from .models import Market
from .monitor import alert

log = logging.getLogger(__name__)


class NicheWatcher:
    def __init__(self, cfg: BotConfig, ledger: Ledger):
        self._cfg = cfg.niche
        self._ledger = ledger
        # Ключевые слова матчим по границам слов: иначе "eth" ловит "Hegseth".
        self._patterns: list[tuple[str, re.Pattern]] = []
        for watchlist in self._cfg.watchlists:
            keywords = [k.lower() for k in watchlist.keywords if k.strip()]
            if not keywords:
                continue
            joined = "|".join(re.escape(k) for k in keywords)
            self._patterns.append((
                watchlist.name,
                re.compile(rf"(?<![a-z0-9])(?:{joined})(?![a-z0-9])"),
            ))

    def _match(self, market: Market) -> str | None:
        text = f"{market.question} {market.event_title}".lower()
        for name, pattern in self._patterns:
            if pattern.search(text):
                return name
        return None

    def _alert_market(self, name: str, m: Market) -> None:
        days = m.days_to_resolution()
        days_text = f"{days:.0f} дн." if days is not None else "дата не указана"
        text = (
            f"НОВЫЙ РЫНОК В НИШЕ [{name}]\n"
            f"{m.question}\n"
            f"Цена Yes: {m.outcome_prices[0] if m.outcome_prices else '?'} | "
            f"объём 24h: ${m.volume_24h_usd:,.0f} | "
            f"до резолюции: {days_text}\n"
            f"Источник резолюции: {m.resolution_source or 'НЕ УКАЗАН'}\n"
            f"Правила: {m.description[:500]}\n"
            f"https://polymarket.com/market/{m.slug}"
        )
        log.info(text)
        alert(text)

    def cycle(self, markets: list[Market]) -> list[tuple[str, Market]]:
        """Новые рынки в нишах: алерт один раз на рынок."""
        if not self._cfg.enabled or not self._patterns:
            return []
        seen = self._ledger.seen_market_ids()
        hits: list[tuple[str, Market]] = []
        fresh_ids: list[str] = []

        for m in markets:
            if m.id in seen or m.closed:
                continue
            fresh_ids.append(m.id)
            name = self._match(m)
            if name is None:
                continue
            hits.append((name, m))
            try:
                self._alert_market(name, m)
            except Exception:
                # Один кривой рынок не должен срывать цикл и пометку «увиден».
                log.exception("niche: алерт по рынку %s", m.id)

        # Помечаем все новые рынки (не только нишевые), чтобы не сканировать заново.
        if fresh_ids:
            self._ledger.mark_markets_seen(fresh_ids)
        return hits
