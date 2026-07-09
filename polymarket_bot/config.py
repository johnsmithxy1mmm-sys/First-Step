"""Конфигурация: config.yaml → pydantic-модели. Секреты — только из .env."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

PACKAGE_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.yaml"


class ScannerConfig(BaseModel):
    price_min: float = 0.002
    price_max: float = 0.05
    min_volume_24h_usd: float = 5_000.0
    min_book_depth_usd: float = 500.0
    book_depth_pct_from_mid: float = 0.20
    min_days_to_resolution: float = 3.0
    max_days_to_resolution: float = 120.0
    # Неоднозначная резолюция: требуем источник резолюции или внятное описание правил.
    require_resolution_clarity: bool = True
    min_description_chars: int = 80
    exclude_keywords: list[str] = Field(default_factory=list)
    include_keywords: list[str] = Field(default_factory=list)
    verify_book_depth: bool = True
    interval_minutes: float = 20.0
    max_candidates_per_cycle: int = 200


class LLMConfig(BaseModel):
    enabled: bool = False              # включать после калибровки остальных сигналов
    model: str = "claude-sonnet-5"
    max_calls_per_cycle: int = 10
    cache_ttl_hours: float = 24.0
    max_tokens: int = 1024


class EstimatorConfig(BaseModel):
    # Порог мисспрайсинга: p_est / p_mkt ≥ min_edge_ratio при p_mkt ≤ max_p_mkt.
    min_edge_ratio: float = 2.0
    max_p_mkt: float = 0.05
    base_rates_file: str = "base_rates.yaml"
    # Вес рыночной цены как якоря в ансамбле (0.9 = почти доверяем рынку;
    # ниже — сигналы легче перевешивают рынок, edge находить проще, но шумнее).
    market_anchor_confidence: float = 0.85
    llm: LLMConfig = Field(default_factory=LLMConfig)


class PortfolioConfig(BaseModel):
    bankroll_usd: float = 5_000.0
    kelly_fraction: float = 0.15       # λ дробного Келли (0.1–0.25)
    max_market_pct: float = 0.01       # ≤ 1% банка на рынок
    max_category_pct: float = 0.10     # ≤ 10% на категорию (с учётом корреляций)
    max_total_exposure_pct: float = 0.30
    max_drawdown_pct: float = 0.25     # стоп всей системы
    take_profit_multiple: float = 7.0  # частичная фиксация на 5–10x
    take_profit_fraction: float = 0.6  # продаём 50–70%

    @field_validator("kelly_fraction")
    @classmethod
    def _kelly_sane(cls, v: float) -> float:
        if not 0 < v <= 0.5:
            raise ValueError("kelly_fraction должен быть в (0, 0.5]")
        return v


class ExecutorConfig(BaseModel):
    max_reprices: int = 3
    fill_timeout_sec: float = 90.0
    poll_interval_sec: float = 5.0
    max_child_order_usd: float = 200.0


class ArbitrageConfig(BaseModel):
    """Стратегия №1: структурный арбитраж neg-risk корзин (YES и NO)."""
    enabled: bool = True
    execute: bool = False              # обнаружение+алерт по умолчанию; ордера — явно
    interval_sec: float = 60.0
    min_profit_pct: float = 0.015      # минимум 1.5% на комплект (покрывает риск ноги)
    prefilter_tolerance: float = 0.01  # порог отбора по ценам Gamma (грубее реального)
    max_stake_usd: float = 300.0
    min_sets: int = 5
    max_legs: int = 20
    max_events_per_cycle: int = 10
    min_leg_volume_24h_usd: float = 500.0


class MarketMakerConfig(BaseModel):
    """Стратегия №3: маркет-мейкинг + liquidity rewards — база денежного потока."""
    enabled: bool = False              # включать осознанно: требует капитала на котировки
    interval_sec: float = 45.0
    max_markets: int = 8
    price_lo: float = 0.10             # средние рынки, не хвосты
    price_hi: float = 0.90
    min_volume_24h_usd: float = 20_000.0
    min_days_to_resolution: float = 2.0
    half_spread: float = 0.01          # полуспред котировки
    quote_size_usd: float = 50.0       # долларов на каждую сторону каждого рынка
    inventory_cap_usd: float = 150.0   # кэп перекоса Yes/No на рынок
    guard_price_move: float = 0.03     # mid сдвинулся сильнее — снять котировки
    guard_cooldown_cycles: int = 3
    guard_volume_ratio: float = 0.5    # 24h-объём > 50% всего оборота = новостной шок


class CrossMarketConfig(BaseModel):
    """Стратегия №2: расхождения с другими площадками (алерты, без автоторговли)."""
    enabled: bool = True
    interval_min: float = 15.0
    min_divergence: float = 0.04       # от 4 п.п.
    min_similarity: float = 0.65       # порог совпадения заголовков (Жаккар)
    min_volume_24h_usd: float = 10_000.0
    max_alerts_per_cycle: int = 10


class Watchlist(BaseModel):
    name: str
    keywords: list[str] = Field(default_factory=list)


class NicheConfig(BaseModel):
    """Стратегия №5: мгновенные алерты о новых рынках в ваших нишах."""
    enabled: bool = True
    watchlists: list[Watchlist] = Field(default_factory=lambda: [
        Watchlist(name="post-soviet", keywords=[
            "russia", "ukraine", "belarus", "kazakhstan", "armenia", "azerbaijan",
            "georgia", "moldova", "putin", "zelensky", "lukashenko", "kremlin",
            "donbas", "crimea", "baltic", "latvia", "lithuania", "estonia",
        ]),
        Watchlist(name="crypto", keywords=[
            "bitcoin", "btc", "ethereum", "eth", "solana", "crypto", "stablecoin",
            "binance", "coinbase", "tether", "sec etf", "halving", "defi",
        ]),
    ])


class RuntimeConfig(BaseModel):
    gamma_host: str = "https://gamma-api.polymarket.com"
    clob_host: str = "https://clob.polymarket.com"
    data_api_host: str = "https://data-api.polymarket.com"
    chain_id: int = 137
    db_path: str = str(PACKAGE_DIR / "data" / "ledger.sqlite")
    log_path: str = str(PACKAGE_DIR / "data" / "bot.jsonl")
    llm_cache_path: str = str(PACKAGE_DIR / "data" / "llm_cache.json")
    request_timeout_sec: float = 30.0
    max_retries: int = 4


class BacktestConfig(BaseModel):
    max_markets: int = 300
    lookback_days_before_end: float = 21.0  # смотрим цену за N дней до резолюции


class BotConfig(BaseModel):
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    estimator: EstimatorConfig = Field(default_factory=EstimatorConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    arbitrage: ArbitrageConfig = Field(default_factory=ArbitrageConfig)
    market_maker: MarketMakerConfig = Field(default_factory=MarketMakerConfig)
    crossmarket: CrossMarketConfig = Field(default_factory=CrossMarketConfig)
    niche: NicheConfig = Field(default_factory=NicheConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "BotConfig":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)

    def base_rates_path(self) -> Path:
        p = Path(self.estimator.base_rates_file)
        return p if p.is_absolute() else PACKAGE_DIR / p
