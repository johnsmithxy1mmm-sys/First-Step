"""Probability estimator: an ensemble of signals over scanner candidates."""

from __future__ import annotations

import logging

from ..config import BotConfig
from ..models import Candidate, Estimate, Market
from .base_rates import BaseRatesSignal, load_base_rates
from .coherence import CoherenceSignal
from .ensemble import combine
from .llm import LLMSignal
from .momentum import MomentumSignal

log = logging.getLogger(__name__)


class Estimator:
    def __init__(self, cfg: BotConfig, llm: LLMSignal | None = None):
        self._cfg = cfg
        self._base_rates = BaseRatesSignal(load_base_rates(cfg.base_rates_path()))
        self._momentum = MomentumSignal()
        self._llm = llm if llm is not None else LLMSignal(cfg)

    def estimate_all(self, candidates: list[Candidate],
                     all_markets: list[Market]) -> list[Estimate]:
        """Estimates candidates; sorts by descending edge."""
        coherence = CoherenceSignal(all_markets)
        self._llm.start_cycle()

        estimates: list[Estimate] = []
        for c in candidates:
            signals = []
            for producer in (
                lambda: coherence.evaluate(c),          # priority #1
                lambda: self._base_rates.evaluate(c),
                lambda: self._momentum.evaluate(c),
                lambda: self._llm.evaluate(c),
            ):
                try:
                    signal = producer()
                except Exception as exc:  # one signal must not break the estimate
                    log.warning("signal error on %s: %s", c.market.id, exc)
                    signal = None
                if signal is not None:
                    signals.append(signal)
            est = combine(c, signals, self._cfg.estimator.market_anchor_confidence)
            estimates.append(est)
            log.debug(
                "estimate market=%s p_mkt=%.4f p_est=%.4f edge=%.2f signals=%s",
                c.market.id, est.p_mkt, est.p_est, est.edge_ratio,
                [(s.name, s.p_est, s.confidence) for s in est.signals],
            )

        estimates.sort(key=lambda e: e.edge_ratio, reverse=True)
        return estimates

    def qualifies(self, estimate: Estimate) -> bool:
        """Mispricing threshold: edge >= 2.0 at p_mkt <= 0.05 (configurable)."""
        e = self._cfg.estimator
        return (estimate.p_mkt <= e.max_p_mkt
                and estimate.edge_ratio >= e.min_edge_ratio)
