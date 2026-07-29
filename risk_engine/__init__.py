"""Pre-trade portfolio risk engine for Hyperliquid perpetuals.

The engine estimates the distribution of outcomes for a book of positions
under stated assumptions. It does not forecast prices (see
`model/drift.py`), does not give advice, and never returns a point estimate
without an interval around it (see `domain.types.RiskEstimate`).
"""

from risk_engine.version import MODEL_VERSION

__all__ = ["MODEL_VERSION"]
