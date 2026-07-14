"""Polymarket longshot bot — a barbell on the mispricing of tail outcomes.

Mispricing scanner: buys a longshot only when its own probability estimate is
materially above the market price (p_est / p_mkt >= threshold).
Modules: scanner -> estimator (base rates, coherence, LLM, momentum)
-> portfolio (fractional Kelly + limits) -> executor (maker limit orders)
-> ledger (sqlite, PnL attribution) -> monitor (rich + Telegram).
"""

__version__ = "1.0.0"
