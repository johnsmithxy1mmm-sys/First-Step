"""LLM-сигнал: оценка вероятности хвостового исхода моделью Claude.

Модель получает вопрос рынка, правила резолюции, цену и срок — и возвращает
структурированный JSON (p_est, confidence, rationale) через официальный SDK
(client.messages.parse + pydantic-схема). Ответы кэшируются на сутки,
частота вызовов ограничена per-cycle лимитом.

Внешний новостной фон подключается через NewsProvider: по умолчанию он пуст
(модель опирается на собственные знания), но интерфейс позволяет подать
заголовки из любого источника.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from ..config import BotConfig
from ..models import Candidate, Signal

log = logging.getLogger(__name__)


class NewsProvider(Protocol):
    def headlines(self, query: str, limit: int = 5) -> list[str]: ...


class NullNewsProvider:
    def headlines(self, query: str, limit: int = 5) -> list[str]:
        return []


class LLMEstimate(BaseModel):
    """Схема структурированного ответа модели."""

    p_est: float = Field(ge=0.0, le=1.0,
                         description="Оценка вероятности исхода Yes, 0..1")
    direction: str = Field(description="'higher' если рынок недооценивает, "
                                       "'lower' если переоценивает, 'fair' если цена честная")
    confidence: float = Field(ge=0.0, le=1.0,
                              description="Уверенность в оценке, 0..1")
    rationale: str = Field(description="Краткое обоснование, 1-3 предложения")


SYSTEM_PROMPT = (
    "You are a calibrated superforecaster evaluating prediction-market tail outcomes. "
    "Estimate the true probability of the stated outcome resolving YES. "
    "Anchor on base rates, adjust for specific evidence, avoid narrative bias. "
    "Cheap tail outcomes are usually cheap for a reason: only deviate from the market "
    "price when you have concrete grounds. Be honest about uncertainty via the "
    "confidence field."
)


class LLMSignal:
    name = "llm"

    def __init__(self, cfg: BotConfig, news: NewsProvider | None = None):
        self._cfg = cfg.estimator.llm
        self._cache_path = Path(cfg.runtime.llm_cache_path)
        self._news = news or NullNewsProvider()
        self._calls_this_cycle = 0
        self._client = None
        self._cache: dict[str, dict] = {}
        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            except ValueError:
                self._cache = {}

    @property
    def available(self) -> bool:
        return self._cfg.enabled and bool(os.environ.get("ANTHROPIC_API_KEY"))

    def start_cycle(self) -> None:
        self._calls_this_cycle = 0

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def _cached(self, key: str) -> dict | None:
        entry = self._cache.get(key)
        if not entry:
            return None
        if time.time() - entry.get("ts", 0) > self._cfg.cache_ttl_hours * 3600:
            return None
        return entry.get("result")

    def _store(self, key: str, result: dict) -> None:
        self._cache[key] = {"ts": time.time(), "result": result}
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False), encoding="utf-8"
        )

    def evaluate(self, candidate: Candidate) -> Signal | None:
        if not self.available or candidate.outcome_index != 0:
            return None

        key = f"{candidate.market.id}:{candidate.outcome_index}"
        result = self._cached(key)
        if result is None:
            if self._calls_this_cycle >= self._cfg.max_calls_per_cycle:
                return None
            result = self._query(candidate)
            if result is None:
                return None
            self._store(key, result)

        p_est = float(result["p_est"])
        confidence = float(result["confidence"])
        if result.get("direction") == "fair":
            confidence *= 0.5  # честная цена = слабый сигнал, почти воздержание
        return Signal(
            name=self.name, p_est=p_est,
            confidence=min(confidence, 0.7),  # LLM не перевешивает структурные сигналы
            rationale=str(result.get("rationale", ""))[:300],
        )

    def _query(self, candidate: Candidate) -> dict | None:
        import anthropic

        m = candidate.market
        days = m.days_to_resolution()
        headlines = self._news.headlines(m.question)
        news_block = ("\nRecent headlines:\n" + "\n".join(f"- {h}" for h in headlines)) \
            if headlines else ""
        prompt = (
            f"Market question: {m.question}\n"
            f"Resolution rules: {m.description[:1200] or 'n/a'}\n"
            f"Resolution source: {m.resolution_source or 'n/a'}\n"
            f"Days until resolution: {days:.0f}\n"
            f"Current market price of YES: {candidate.p_mkt:.4f}\n"
            f"{news_block}\n"
            "Estimate the true probability that this market resolves YES."
        )
        self._calls_this_cycle += 1
        try:
            response = self._get_client().messages.parse(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_format=LLMEstimate,
            )
        except anthropic.APIError as exc:
            log.warning("llm: %s", exc)
            return None
        parsed = response.parsed_output
        if parsed is None:
            return None
        return parsed.model_dump()
