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
    def __init__(self, cfg: BotConfig, llm: LLMSignal | None = None,
                 extra_signals: list | None = None, platt=None):
        self._cfg = cfg
        self._base_rates = BaseRatesSignal(load_base_rates(cfg.base_rates_path()))
        self._momentum = MomentumSignal()
        self._llm = llm if llm is not None else LLMSignal(cfg)
        # Extra signal sources with an .evaluate(candidate) -> Signal|None method
        # (e.g. the smart-money signal). Each contributes to the ensemble.
        self._extra = extra_signals or []
        # Optional PlattCalibrator: recalibrates the ensemble p_est against
        # realized outcomes (identity until fitted with enough resolutions).
        self._platt = platt

    def estimate_all(self, candidates: list[Candidate],
                     all_markets: list[Market]) -> list[Estimate]:
        """Estimates candidates; sorts by descending edge."""
        coherence = CoherenceSignal(all_markets)
        self._llm.start_cycle()

        estimates: list[Estimate] = []
        for c in candidates:
            signals = []
            # `c=c` binds the loop variable explicitly. These lambdas are
            # invoked within this same iteration so late binding does not bite
            # today — but the next line already binds `s=s`, and an asymmetry
            # like that is exactly what a later refactor (deferring the calls)
            # turns into a silent bug.
            producers = [
                lambda c=c: coherence.evaluate(c),      # priority #1
                lambda c=c: self._base_rates.evaluate(c),
                lambda c=c: self._momentum.evaluate(c),
                lambda c=c: self._llm.evaluate(c),
            ]
            producers += [(lambda s=s, c=c: s.evaluate(c)) for s in self._extra]
            for producer in producers:
                try:
                    signal = producer()
                except Exception as exc:  # one signal must not break the estimate
                    log.warning("signal error on %s: %s", c.market.id, exc)
                    signal = None
                if signal is not None:
                    signals.append(signal)
            est = combine(c, signals, self._cfg.estimator.market_anchor_confidence)
            if self._platt is not None:
                est = est.model_copy(update={"p_est": self._platt.calibrate(est.p_est)})
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
