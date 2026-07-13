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
    min_profit_pct: float = 0.015      # минимум 1.5% ЧИСТОГО edge (после комиссий)
    suspicious_gross_edge: float = 0.05  # gross выше -> вероятно неполная корзина, не исполнять
    prefilter_tolerance: float = 0.01  # порог отбора по ценам Gamma (грубее реального)
    max_stake_usd: float = 300.0
    min_sets: int = 5
    max_legs: int = 20
    max_events_per_cycle: int = 10
    min_leg_volume_24h_usd: float = 500.0


class MarketMakerConfig(BaseModel):
    """Ядро (80% капитала): маркет-мейкинг + liquidity rewards farming."""
    enabled: bool = False              # включать осознанно: требует капитала на котировки
    interval_sec: float = 30.0
    max_markets: int = 5
    # Отбор рынков (скоринг-модуль): объём, горизонт, rewards, стабильность.
    min_volume_24h_usd: float = 50_000.0
    min_days_to_resolution: float = 30.0
    require_rewards_program: bool = True
    max_daily_midpoint_move: float = 0.05   # реализованная волатильность midpoint
    # Котирование.
    half_spread: float = 0.01
    quote_size_usd: float = 10.0
    inventory_skew_k: float = 0.5      # сдвиг fair против инвентаря
    # Requote-гистерезис: лишний churn ест rate limit и rewards-сэмплинг.
    requote_threshold_ticks: float = 2.0
    requote_timer_sec: float = 120.0
    # Adverse selection guard.
    guard_price_move: float = 0.03
    guard_cooldown_cycles: int = 3
    guard_volume_ratio: float = 0.5


class RiskLimitsConfig(BaseModel):
    """Абсолютные лимиты риск-фреймворка (жёсткие требования мастер-промпта)."""
    max_position_per_market_usd: float = 50.0
    max_global_exposure_usd: float = 300.0
    max_daily_loss_usd: float = 25.0        # дневной стоп → halt до ручного рестарта
    max_drawdown_pct: float = 0.15          # от high-water mark → полный halt
    min_edge_after_fees: float = 0.01
    reconcile_interval_sec: float = 60.0
    ws_staleness_kill_sec: float = 10.0


class FeesConfig(BaseModel):
    """Fee Structure V2 (март 2026). Проверяйте актуальность на docs.polymarket.com."""
    taker: dict[str, float] = Field(default_factory=lambda: {
        "crypto": 0.07, "sports": 0.03, "finance": 0.04, "politics": 0.04,
        "tech": 0.04, "economics": 0.05, "culture": 0.05, "weather": 0.05,
        "geopolitics": 0.0, "other": 0.04,
    })
    maker_rebate_frac: float = 0.35    # rebate 20-50% от taker fee; середина


class WSConfig(BaseModel):
    enabled: bool = True
    url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    ping_interval_sec: float = 10.0


class RateLimitConfig(BaseModel):
    """Собственный token-bucket даже при повышенных лимитах V2."""
    orders_per_sec: float = 20.0
    orders_burst: float = 100.0
    reads_per_sec: float = 30.0
    reads_burst: float = 100.0


class SatelliteConfig(BaseModel):
    """Сателлит (20%): T-10s TA на 5-мин BTC up/down. По умолчанию ВЫКЛЮЧЕН."""
    enabled: bool = False
    slug_template: str = "btc-updown-5m-{ts}"
    window_sec: int = 300
    entry_from_sec: float = 30.0       # входим за 10-30 сек до закрытия окна
    entry_to_sec: float = 10.0
    edge_threshold: float = 0.05       # P(model) - implied - fee > порога
    kelly_fraction: float = 0.25       # quarter-Kelly
    max_bet_usd: float = 10.0
    candles_url: str = ("https://api.binance.com/api/v3/klines"
                        "?symbol=BTCUSDT&interval=1m&limit=30")


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
        Watchlist(name="ai", keywords=[
            "ai", "openai", "chatgpt", "gpt", "anthropic", "claude", "gemini",
            "deepmind", "agi", "llm", "grok", "xai", "nvidia", "sora",
            "artificial intelligence", "large language model",
        ]),
        Watchlist(name="football-eu", keywords=[
            "champions league", "premier league", "la liga", "serie a", "bundesliga",
            "ligue 1", "europa league", "uefa", "ballon", "real madrid", "barcelona",
            "man city", "manchester city", "manchester united", "man united",
            "liverpool", "arsenal", "chelsea", "tottenham", "bayern", "dortmund",
            "psg", "juventus", "inter milan", "ac milan", "napoli", "atletico madrid",
        ]),
    ])


class SmartMoneyConfig(BaseModel):
    """Трекер «умных денег»: алерты, когда сильные кошельки заходят в рынок.

    Использует публичный Data API Polymarket (/positions по адресу).
    watch_wallets — адреса, за которыми следим (сильные игроки, которых вы
    нашли на leaderboard'е polymarket.com). Пусто = трекер молчит.
    """
    enabled: bool = False
    interval_min: float = 10.0
    watch_wallets: list[str] = Field(default_factory=list)
    min_position_usd: float = 50.0     # игнорируем пыль
    tail_max_price: float = 0.10       # вход дешевле — помечаем как хвостовой
    only_niche_or_tail: bool = False   # true = алертить лишь про нишу/хвост


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
    risk: RiskLimitsConfig = Field(default_factory=RiskLimitsConfig)
    fees: FeesConfig = Field(default_factory=FeesConfig)
    ws: WSConfig = Field(default_factory=WSConfig)
    ratelimit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    satellite: SatelliteConfig = Field(default_factory=SatelliteConfig)
    crossmarket: CrossMarketConfig = Field(default_factory=CrossMarketConfig)
    niche: NicheConfig = Field(default_factory=NicheConfig)
    smart_money: SmartMoneyConfig = Field(default_factory=SmartMoneyConfig)
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
