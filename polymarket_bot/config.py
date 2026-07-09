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
    # Сколько долларов вкладывать в одну ставку (базовый размер, см. stake_scaling).
    stake_usd: float = 15.0
    # Общий бюджет на серию ставок; бот не превысит его.
    total_budget_usd: float = 1500.0
    # Дневной лимит трат (0 = выключен). Защита от «слить всё за один цикл».
    daily_budget_usd: float = 0.0
    # Максимум ставок за один запуск (дополнительный предохранитель к бюджету).
    max_bets: int = 100
    # Масштабировать размер ставки по скору кандидата: 0.5x–1.5x от stake_usd.
    stake_scaling: bool = True

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
    # Минимальный скор кандидата (0..1); 0 = брать всех, ранжируя по скору.
    min_score: float = 0.0

    # --- Исполнение ---
    # maker — свой ордер у бида (дешевле, но может не исполниться);
    # taker — покупка по лучшему ask (дороже, но сразу).
    entry_mode: str = "maker"
    # Перед ставкой сверять реальный стакан CLOB (медленнее, но честнее).
    verify_orderbook: bool = True
    # Пауза между запросами к API, чтобы не упереться в rate limit.
    request_delay_sec: float = 0.25

    # --- Автопилот (команда auto) ---
    # Период цикла в минутах.
    auto_interval_min: float = 30.0
    # Снимать неисполненные ордера старше N часов — не морозить бюджет.
    max_order_age_hours: float = 12.0
    # Автофиксация: когда цена позиции вырастает в N раз от входа...
    take_profit_multiple: float = 10.0
    # ...продать эту долю позиции (0.5 = половину; остаток «едет бесплатно»).
    take_profit_fraction: float = 0.5

    # --- Арбитраж neg-risk событий (сумма всех исходов < $1) ---
    arb_scan: bool = True
    # Минимальная гарантированная маржа, чтобы считать событие арбитражем.
    arb_min_edge: float = 0.02
    # Автоматически исполнять арбитражи (экспериментально; есть риск частичного входа).
    arb_execute: bool = False
    arb_stake_usd: float = 50.0

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
        if self.entry_mode not in ("maker", "taker"):
            raise ValueError("entry_mode должен быть 'maker' или 'taker'")
        if not 0 < self.take_profit_fraction <= 1:
            raise ValueError("take_profit_fraction должен быть в (0, 1]")
        if self.take_profit_multiple <= 1:
            raise ValueError("take_profit_multiple должен быть больше 1")
        if not 0 <= self.arb_min_edge < 1:
            raise ValueError("arb_min_edge должен быть в [0, 1)")
