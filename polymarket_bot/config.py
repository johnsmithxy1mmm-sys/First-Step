"""Конфигурация бота: значения по умолчанию + JSON-файл + переопределения из CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class BotConfig:
    # --- Стратегия ---
    # Максимальная цена исхода в долларах: 0.01 = потенциал x100 при победе.
    max_price: float = 0.01
    # Минимальная цена: исходы за 0.000 обычно означают мёртвый рынок без стакана.
    min_price: float = 0.001
    # Сколько долларов вкладывать в одну ставку.
    stake_usd: float = 15.0
    # Общий бюджет на серию ставок; бот не превысит его.
    total_budget_usd: float = 1500.0
    # Максимум ставок за один запуск (дополнительный предохранитель к бюджету).
    max_bets: int = 100

    # --- Фильтры рынков ---
    # Отсекаем неликвид: на пустых рынках дешёвая цена — иллюзия, купить не получится.
    min_liquidity_usd: float = 1000.0
    min_volume_usd: float = 5000.0
    # Окно резолюции: слишком близкие рынки почти решены, слишком далёкие морозят деньги.
    min_days_to_resolution: float = 1.0
    max_days_to_resolution: float = 120.0
    # Ключевые слова по вопросу рынка (без учёта регистра). Пустой include = берём всё.
    include_keywords: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)
    # Не больше одной ставки на рынок, чтобы не коррелировать риски.
    max_bets_per_market: int = 1

    # --- Исполнение ---
    # Перед ставкой сверять реальный лучший ask в стакане CLOB (медленнее, но честнее).
    verify_orderbook: bool = True
    # Пауза между запросами к API, чтобы не упереться в rate limit.
    request_delay_sec: float = 0.25

    # --- Подключение ---
    clob_host: str = "https://clob.polymarket.com"
    gamma_host: str = "https://gamma-api.polymarket.com"
    data_api_host: str = "https://data-api.polymarket.com"
    chain_id: int = 137  # Polygon
    # 0 — обычный кошелёк (EOA), 1 — аккаунт через email (Magic),
    # 2 — аккаунт через браузерный кошелёк (прокси Polymarket).
    signature_type: int = 0

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides) -> "BotConfig":
        """Собирает конфиг: файл (если есть) поверх дефолтов, CLI поверх файла."""
        data: dict = {}
        if path is not None:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            known = {f.name for f in fields(cls)}
            unknown = set(raw) - known
            if unknown:
                raise ValueError(f"Неизвестные ключи в конфиге: {sorted(unknown)}")
            data.update(raw)
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    def validate(self) -> None:
        if not 0 < self.max_price < 1:
            raise ValueError("max_price должен быть между 0 и 1")
        if self.stake_usd <= 0 or self.total_budget_usd <= 0:
            raise ValueError("stake_usd и total_budget_usd должны быть положительными")
        if self.stake_usd > self.total_budget_usd:
            raise ValueError("stake_usd больше общего бюджета")
