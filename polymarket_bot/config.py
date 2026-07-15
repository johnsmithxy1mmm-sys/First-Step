"""Configuration: config.yaml -> pydantic models. Secrets come only from .env."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

PACKAGE_DIR = Path(__file__).parent
# config.yaml is the user's personal config (outside git). config.example.yaml
# is the repo template; used when there is no config.yaml of your own yet.
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.yaml"
EXAMPLE_CONFIG_PATH = PACKAGE_DIR / "config.example.yaml"


class ScannerConfig(BaseModel):
    price_min: float = 0.002
    price_max: float = 0.05
    min_volume_24h_usd: float = 5_000.0
    min_book_depth_usd: float = 500.0
    book_depth_pct_from_mid: float = 0.20
    min_days_to_resolution: float = 3.0
    max_days_to_resolution: float = 120.0
    # Ambiguous resolution: require a resolution source or a clear rules description.
    require_resolution_clarity: bool = True
    min_description_chars: int = 80
    exclude_keywords: list[str] = Field(default_factory=list)
    include_keywords: list[str] = Field(default_factory=list)
    verify_book_depth: bool = True
    interval_minutes: float = 20.0
    max_candidates_per_cycle: int = 200


class LLMConfig(BaseModel):
    enabled: bool = False              # enable after calibrating the other signals
    model: str = "claude-sonnet-5"
    max_calls_per_cycle: int = 10
    cache_ttl_hours: float = 24.0
    max_tokens: int = 1024


class EstimatorConfig(BaseModel):
    # Mispricing threshold: p_est / p_mkt >= min_edge_ratio when p_mkt <= max_p_mkt.
    min_edge_ratio: float = 2.0
    max_p_mkt: float = 0.05
    base_rates_file: str = "base_rates.yaml"
    # Weight of the market price as an anchor in the ensemble (0.9 = almost trust
    # the market; lower -> signals outweigh the market more easily, edge is easier
    # to find but noisier).
    market_anchor_confidence: float = 0.85
    llm: LLMConfig = Field(default_factory=LLMConfig)


class PortfolioConfig(BaseModel):
    bankroll_usd: float = 5_000.0
    kelly_fraction: float = 0.15       # lambda of fractional Kelly (0.1-0.25)
    max_market_pct: float = 0.01       # <= 1% of bankroll per market
    max_category_pct: float = 0.10     # <= 10% per category (accounting for correlations)
    max_total_exposure_pct: float = 0.30
    max_drawdown_pct: float = 0.25     # stop for the whole system
    take_profit_multiple: float = 7.0  # partial take at 5-10x
    take_profit_fraction: float = 0.6  # sell 50-70%

    @field_validator("kelly_fraction")
    @classmethod
    def _kelly_sane(cls, v: float) -> float:
        if not 0 < v <= 0.5:
            raise ValueError("kelly_fraction must be in (0, 0.5]")
        return v


class ExecutorConfig(BaseModel):
    max_reprices: int = 3
    fill_timeout_sec: float = 90.0
    poll_interval_sec: float = 5.0
    max_child_order_usd: float = 200.0


class FadeConfig(BaseModel):
    """Fading overpriced tails: buy NO when the YES tail is overpriced.

    The profitable side of favorite-longshot bias: the crowd inflates the price
    of cheap outcomes, we systematically harvest that skew by buying NO.
    Positive on average (the bias is real), but each bet is asymmetric — a rare
    large loss when the tail hits — hence hard caps and diversification.
    """
    enabled: bool = False
    # Systematic correction: treat the tail as overpriced by at least this fraction
    # (0.30 = "cheap outcomes are on average 30% above fair price").
    bias_discount: float = 0.30
    fade_max_price: float = 0.10       # fade only tails cheaper than this YES price
    min_tail_price: float = 0.005      # below this — illiquid / noise
    min_edge_after_fees: float = 0.005  # at least 0.5% net edge on the NO side
    # Veto: if the estimator sees a REAL longshot (p_est >= ratio * p_mkt — the same
    # threshold the longshot strategy BUYS Yes on), don't fade against our own
    # signal. A slight drift of p_est above market is anchor noise, no obstacle.
    longshot_veto_ratio: float = 2.0
    # Horizon cap: only fade tails resolving within this many days. The fade edge
    # (~bias) is fixed per bet, so a distant resolution means a tiny IRR (capital
    # locked for months to earn a few %). Near-term tails recycle capital fast.
    max_days_to_resolution: float = 45.0


class ArbitrageConfig(BaseModel):
    """Strategy #1: structural arbitrage of neg-risk baskets (YES and NO)."""
    enabled: bool = True
    execute: bool = False              # detect+alert by default; orders explicitly
    interval_sec: float = 60.0
    min_profit_pct: float = 0.015      # at least 1.5% NET edge (after fees)
    suspicious_gross_edge: float = 0.05  # gross above this -> likely incomplete basket, don't execute
    prefilter_tolerance: float = 0.01  # selection threshold on Gamma prices (coarser than real)
    max_stake_usd: float = 300.0
    min_sets: int = 5
    max_legs: int = 20
    max_events_per_cycle: int = 10
    min_leg_volume_24h_usd: float = 500.0


class MarketMakerConfig(BaseModel):
    """Core (80% of capital): market making + liquidity rewards farming."""
    enabled: bool = False              # enable deliberately: needs capital for quotes
    interval_sec: float = 30.0
    max_markets: int = 5
    # Market selection (scoring module): volume, horizon, rewards, stability.
    min_volume_24h_usd: float = 50_000.0
    min_days_to_resolution: float = 30.0
    require_rewards_program: bool = True
    max_daily_midpoint_move: float = 0.05   # realized midpoint volatility
    # Quoting.
    half_spread: float = 0.01
    quote_size_usd: float = 10.0
    inventory_skew_k: float = 0.5      # shift fair against inventory
    # MM 2.0 (default off = classic behavior):
    vol_spread_k: float = 0.0          # A-S: widen half-spread by k * realized sigma
    rewards_weighting: bool = False    # allocate quote size across markets by score
    # Requote hysteresis: excess churn eats rate limit and rewards sampling.
    requote_threshold_ticks: float = 2.0
    requote_timer_sec: float = 120.0
    # Adverse selection guard.
    guard_price_move: float = 0.03
    guard_cooldown_cycles: int = 3
    guard_volume_ratio: float = 0.5


class RiskLimitsConfig(BaseModel):
    """Absolute risk-framework limits (hard requirements from the master prompt)."""
    max_position_per_market_usd: float = 50.0
    max_global_exposure_usd: float = 300.0
    max_daily_loss_usd: float = 25.0        # daily stop -> halt until manual restart
    max_drawdown_pct: float = 0.15          # from the high-water mark -> full halt
    min_edge_after_fees: float = 0.01
    reconcile_interval_sec: float = 60.0
    ws_staleness_kill_sec: float = 10.0


class FeesConfig(BaseModel):
    """Fee Structure V2 (March 2026). Verify it's current at docs.polymarket.com."""
    taker: dict[str, float] = Field(default_factory=lambda: {
        "crypto": 0.07, "sports": 0.03, "finance": 0.04, "politics": 0.04,
        "tech": 0.04, "economics": 0.05, "culture": 0.05, "weather": 0.05,
        "geopolitics": 0.0, "other": 0.04,
    })
    maker_rebate_frac: float = 0.35    # rebate 20-50% of the taker fee; midpoint


class WSConfig(BaseModel):
    enabled: bool = True
    url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    ping_interval_sec: float = 10.0


class RateLimitConfig(BaseModel):
    """Own token bucket even under the raised V2 limits."""
    orders_per_sec: float = 20.0
    orders_burst: float = 100.0
    reads_per_sec: float = 30.0
    reads_burst: float = 100.0


class SatelliteConfig(BaseModel):
    """Satellite (20%): T-10s TA on 5-min BTC up/down. DISABLED by default."""
    enabled: bool = False
    slug_template: str = "btc-updown-5m-{ts}"
    window_sec: int = 300
    entry_from_sec: float = 30.0       # enter 10-30 sec before the window closes
    entry_to_sec: float = 10.0
    edge_threshold: float = 0.05       # P(model) - implied - fee > threshold
    kelly_fraction: float = 0.25       # quarter-Kelly
    max_bet_usd: float = 10.0
    candles_url: str = ("https://api.binance.com/api/v3/klines"
                        "?symbol=BTCUSDT&interval=1m&limit=30")


class CrossMarketConfig(BaseModel):
    """Strategy #2: divergences with other venues (alerts, no auto-trading)."""
    enabled: bool = True
    interval_min: float = 15.0
    min_divergence: float = 0.04       # from 4 pp
    min_similarity: float = 0.65       # title match threshold (Jaccard)
    min_volume_24h_usd: float = 10_000.0
    max_alerts_per_cycle: int = 10


class Watchlist(BaseModel):
    name: str
    keywords: list[str] = Field(default_factory=list)


class NicheConfig(BaseModel):
    """Strategy #5: instant alerts on new markets in your niches."""
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
    """Smart-money tracker: alerts when strong wallets enter a market.

    Uses Polymarket's public Data API (/positions by address). watch_wallets are
    the addresses we follow (strong players you found on the polymarket.com
    leaderboard). Empty = the tracker stays silent.
    """
    enabled: bool = False
    interval_min: float = 10.0
    watch_wallets: list[str] = Field(default_factory=list)
    min_position_usd: float = 50.0     # ignore dust
    tail_max_price: float = 0.10       # entry cheaper than this — flag as a tail
    only_niche_or_tail: bool = False   # true = alert only on niche/tail


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
    lookback_days_before_end: float = 21.0  # look at price N days before resolution


class BotConfig(BaseModel):
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    estimator: EstimatorConfig = Field(default_factory=EstimatorConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    arbitrage: ArbitrageConfig = Field(default_factory=ArbitrageConfig)
    fade: FadeConfig = Field(default_factory=FadeConfig)
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
        if path is not None:
            chosen = Path(path)
        elif DEFAULT_CONFIG_PATH.exists():
            chosen = DEFAULT_CONFIG_PATH          # user's personal config
        elif EXAMPLE_CONFIG_PATH.exists():
            chosen = EXAMPLE_CONFIG_PATH          # fresh clone with no config.yaml of its own
        else:
            return cls()
        if not chosen.exists():
            return cls()
        raw = yaml.safe_load(chosen.read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)

    def base_rates_path(self) -> Path:
        p = Path(self.estimator.base_rates_file)
        return p if p.is_absolute() else PACKAGE_DIR / p
