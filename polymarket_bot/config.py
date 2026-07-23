"""Configuration: config.yaml -> pydantic models. Secrets come only from .env."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).parent
# config.example.yaml (in git) is the BASE layer: every default + comments,
# kept current by upstream. config.yaml (yours, outside git) is an OVERLAY —
# it needs to hold only what you want to change. At load time the overlay is
# deep-merged onto the base, so any new block added upstream is live after a
# `git pull` with zero copying, while your explicit values always win.
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.yaml"
EXAMPLE_CONFIG_PATH = PACKAGE_DIR / "config.example.yaml"


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively overlay `overlay` onto `base`; overlay wins at the leaves.

    Nested dicts merge key-by-key; every non-dict value (scalars AND lists) is
    replaced wholesale by the overlay — a config list (watchlists, keywords)
    is a deliberate choice, not something to silently concatenate.
    """
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _top_level_keys(text: str) -> list[str]:
    """Ordered top-level YAML section keys (col-0 `key:` lines)."""
    return [m.group(1) for line in text.splitlines()
            if (m := re.match(r"^([A-Za-z_][\w]*):", line))]


def example_section_blocks() -> list[tuple[str, str]]:
    """The example split into (section_key, raw_text) blocks, comments kept.

    A block owns the contiguous comment lines directly above its `key:` header,
    so materializing one into config.yaml carries its documentation along.
    """
    text = EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    headers = [i for i, ln in enumerate(lines) if re.match(r"^[A-Za-z_][\w]*:", ln)]
    if not headers:
        return []
    starts = []
    for h in headers:
        s = h
        while s - 1 >= 0 and lines[s - 1].lstrip().startswith("#"):
            s -= 1                    # pull in the comment run directly above
        starts.append(s)
    blocks = []
    for idx, h in enumerate(headers):
        block_start = starts[idx]
        block_end = starts[idx + 1] if idx + 1 < len(headers) else len(lines)
        key = re.match(r"^([A-Za-z_][\w]*):", lines[h]).group(1)
        blocks.append((key, "".join(lines[block_start:block_end])))
    return blocks


def missing_sections(user_path: Path | None = None) -> list[str]:
    """Top-level sections present in the example but absent from config.yaml."""
    user_path = user_path or DEFAULT_CONFIG_PATH
    if not user_path.exists():
        return _top_level_keys(EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))
    have = set((yaml.safe_load(user_path.read_text(encoding="utf-8")) or {}).keys())
    return [k for k, _ in example_section_blocks() if k not in have]


def sync_config(user_path: Path | None = None) -> list[str]:
    """Append example sections missing from config.yaml (comments and all).

    Append-only: never touches a section you already have, so your values and
    edits are preserved. Creates config.yaml from the full example if absent.
    Returns the section keys that were added.
    """
    user_path = user_path or DEFAULT_CONFIG_PATH
    if not user_path.exists():
        user_path.write_text(EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8"),
                             encoding="utf-8")
        return _top_level_keys(EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))
    missing = set(missing_sections(user_path))
    if not missing:
        return []
    current = user_path.read_text(encoding="utf-8")
    additions = [f"\n{text.rstrip()}\n" for key, text in example_section_blocks()
                 if key in missing]
    if not current.endswith("\n"):
        current += "\n"
    header = ("\n# --- synced from config.example.yaml (edit freely; "
              "your values are never overwritten) ---\n")
    user_path.write_text(current + header + "".join(additions), encoding="utf-8")
    return [k for k, _ in example_section_blocks() if k in missing]


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
    # Platt recalibration of p_est against realized outcomes (fitted by the
    # calibration job; identity until enough resolutions accumulate).
    platt_enabled: bool = False
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


class ResolutionConfig(BaseModel):
    """Resolution alpha: buy a near-resolved side for a small near-riskless gain.

    When a market's top outcome trades at 0.95-0.985 with fresh volume and is
    at/near its end date, the outcome is effectively decided but not yet paid.
    Buying it (taker) earns (1 - price) over hours. Honest residual risk: a UMA
    dispute — reserved via dispute_haircut. Detect+alert by default.
    """
    enabled: bool = False
    execute: bool = False
    interval_sec: float = 120.0
    near_min: float = 0.95             # top outcome in this band = likely resolved-pending
    near_max: float = 0.985
    min_net_edge: float = 0.01         # after taker fee + dispute haircut
    dispute_haircut: float = 0.02      # reserve for UMA dispute risk
    min_volume_24h_usd: float = 20_000.0
    volume_spike_ratio: float = 0.10   # 24h volume >= this fraction of total = fresh activity
    max_days_to_resolution: float = 7.0
    max_stake_usd: float = 200.0
    max_alerts_per_cycle: int = 10


class RedemptionConfig(BaseModel):
    """Auto-redemption of won positions (live mode): a winning token pays $1
    only after redeemPositions is called on-chain — until then the capital is
    frozen in an already-decided market. Alert-first; on-chain execution is
    opt-in (execute + web3 + POLYGON_RPC_URL). Neg-risk positions are always a
    manual claim (different adapter — a wrong automated call loses money).
    """
    enabled: bool = True
    execute: bool = False
    interval_sec: float = 300.0
    min_claim_usd: float = 1.0     # ignore dust (gas would eat it)
    rpc_url: str = ""              # empty -> POLYGON_RPC_URL env


class RulesLawyerConfig(BaseModel):
    """Strategy #4 support: LLM compares the headline with the resolution rules."""
    enabled: bool = False
    model: str = "claude-sonnet-5"
    max_calls_per_cycle: int = 5
    min_volume_24h_usd: float = 20_000.0
    max_tokens: int = 512


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


class ChainArbConfig(BaseModel):
    """Strategy: near-riskless arbitrage across logically nested ('ladder')
    sibling markets in the same event -- e.g. "Fed cuts by June" vs "Fed cuts
    by July" (a later ABSORBING deadline can never be less likely), or "BTC
    reaches $150k by <date>" vs "$200k by <date>" (a higher continuous-path
    threshold can never be MORE likely). The relationship is a logical fact of
    how the two markets are WORDED, not a probability estimate.

    Unlike neg-risk (an official Polymarket flag), the pairing here is
    INFERRED from question text (chainarb.classify_pair) -- classification_
    haircut reserves margin against a bad match turning "riskless" into a
    real bet. Detect + alert by default (execute: false).
    """
    enabled: bool = True
    execute: bool = False
    interval_sec: float = 90.0
    min_net_edge: float = 0.03          # at least 3% NET edge (after fee + haircut)
    classification_haircut: float = 0.02  # reserve for a bad ladder-pair match
    prefilter_tolerance: float = 0.01   # Gamma-price threshold before hitting real books
    max_stake_usd: float = 200.0
    min_sets: int = 5                   # dust depth at the ask can fake an edge
    min_leg_volume_24h_usd: float = 1_000.0
    max_events_per_cycle: int = 10


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
    # Queue preservation: if the new level is within one tick of the old and the
    # current order's fill probability exceeds this, keep the order (its queue
    # position is worth more than the tick). 0 = off (classic behavior).
    requote_min_fill_prob: float = 0.0
    # Adverse selection guard.
    guard_price_move: float = 0.03
    guard_cooldown_cycles: int = 3
    guard_volume_ratio: float = 0.5


class SprintMakerConfig(MarketMakerConfig):
    """Short-dated MM (fast capital turnover): quote LIQUID markets that resolve
    within hours, earn spread + maker rebate, and recycle the capital at
    settlement 1-2 days later — instead of the 30d+ horizon of the core MM.

    Same quoting engine, but a tighter risk profile: near-resolution markets
    have higher adverse selection (info arrives, price jumps), so we lean harder
    against inventory, widen the base spread, pull quotes on smaller shocks, and
    refuse the final settlement window (min_hours_to_resolution) where direction
    dominates. Income is spread + rebate only (short markets are rarely in the
    rewards program) — still +EV as long as the captured spread beats the
    adverse-selection cost, which the fee-break-even floor and the guard enforce.
    """
    enabled: bool = False
    interval_sec: float = 20.0             # faster loop — short markets move
    min_volume_24h_usd: float = 50_000.0   # need real two-sided flow for fast in/out
    require_rewards_program: bool = False  # short markets rarely in rewards; spread+rebate still +EV
    max_hours_to_resolution: float = 48.0  # only quote markets resolving this soon
    min_hours_to_resolution: float = 2.0   # but skip the settlement window (direction dominates)
    half_spread: float = 0.015             # a touch wider to pay for adverse selection
    quote_size_usd: float = 25.0
    inventory_skew_k: float = 1.0          # lean HARDER against inventory than core MM
    vol_spread_k: float = 8.0              # widen spread with realized volatility
    guard_price_move: float = 0.02         # pull quotes on a smaller shock
    guard_cooldown_cycles: int = 3
    requote_timer_sec: float = 45.0        # refresh quotes more often


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
    as_signal: bool = False            # feed strong-wallet entries into the ensemble
    signal_max_confidence: float = 0.40  # cap on the smart-money signal weight
    signal_min_pnl_usd: float = 1_000.0  # wallet needs this realized PnL to count


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


class TicksConfig(BaseModel):
    """Continuous market-data recording — the raw material for every learned
    model (correlations, fill calibration). Cheap to keep on; the learning
    loop is only as good as the history it has."""
    enabled: bool = True
    db_path: str = str(PACKAGE_DIR / "data" / "ticks.sqlite")
    retention_days: float = 30.0
    flush_sec: float = 2.0


class AllocatorConfig(BaseModel):
    """Sharpe allocator that ACTS: digest-time Sharpe weights move per-strategy
    sizing multipliers inside a clamped corridor. Hard risk caps always apply
    AFTER the multiplier, so it can tilt capital but never break a limit."""
    enabled: bool = True
    floor: float = 0.7             # multiplier corridor
    ceil: float = 1.3
    smoothing: float = 0.5         # EMA weight of the new target per update


class OpsConfig(BaseModel):
    """Prometheus /metrics + /health server (stdlib, no extra deps)."""
    metrics_enabled: bool = False
    metrics_port: int = 9090
    # Bind localhost by default: /metrics exposes equity/PnL. Set "0.0.0.0"
    # only behind a firewall or inside Docker (compose needs it for the port map).
    metrics_bind: str = "127.0.0.1"


class BacktestConfig(BaseModel):
    max_markets: int = 300
    lookback_days_before_end: float = 21.0  # look at price N days before resolution


class BotConfig(BaseModel):
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    estimator: EstimatorConfig = Field(default_factory=EstimatorConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    arbitrage: ArbitrageConfig = Field(default_factory=ArbitrageConfig)
    resolution: ResolutionConfig = Field(default_factory=ResolutionConfig)
    redemption: RedemptionConfig = Field(default_factory=RedemptionConfig)
    ticks: TicksConfig = Field(default_factory=TicksConfig)
    allocator: AllocatorConfig = Field(default_factory=AllocatorConfig)
    ruleslawyer: RulesLawyerConfig = Field(default_factory=RulesLawyerConfig)
    fade: FadeConfig = Field(default_factory=FadeConfig)
    market_maker: MarketMakerConfig = Field(default_factory=MarketMakerConfig)
    sprint_mm: SprintMakerConfig = Field(default_factory=SprintMakerConfig)
    chain_arb: ChainArbConfig = Field(default_factory=ChainArbConfig)
    risk: RiskLimitsConfig = Field(default_factory=RiskLimitsConfig)
    fees: FeesConfig = Field(default_factory=FeesConfig)
    ws: WSConfig = Field(default_factory=WSConfig)
    ratelimit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    satellite: SatelliteConfig = Field(default_factory=SatelliteConfig)
    crossmarket: CrossMarketConfig = Field(default_factory=CrossMarketConfig)
    niche: NicheConfig = Field(default_factory=NicheConfig)
    smart_money: SmartMoneyConfig = Field(default_factory=SmartMoneyConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    ops: OpsConfig = Field(default_factory=OpsConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "BotConfig":
        """config.example.yaml (base) deep-merged under config.yaml (overlay).

        Your file only needs the values you changed; every section you did NOT
        write comes live from the example — so a `git pull` that ships a new
        block activates it with the tuned example values, zero hand-copying.
        Anything you wrote always wins over the example.
        """
        base = _read_yaml(EXAMPLE_CONFIG_PATH)
        user_file = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        overlay = _read_yaml(user_file)
        if overlay and base:
            adopted = sorted(k for k in base if k not in overlay)
            if adopted:
                log.info("config: sections inherited from config.example.yaml "
                         "(absent in %s): %s — run `--mode sync-config` to "
                         "materialize them into your file for editing",
                         user_file.name, ", ".join(adopted))
        merged = _deep_merge(base, overlay)
        if not merged:
            return cls()
        return cls.model_validate(merged)

    def base_rates_path(self) -> Path:
        p = Path(self.estimator.base_rates_file)
        return p if p.is_absolute() else PACKAGE_DIR / p
