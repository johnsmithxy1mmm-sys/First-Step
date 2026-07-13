"""Стратегия: фейдинг переоценённых хвостов (прибыльная сторона longshot-bias).

Дешёвые исходы на prediction-рынках систематически ПЕРЕоценены (толпа любит
покупать «лотерейные билеты»). Наивная покупка таких хвостов минусова — а
вот обратная сторона плюсовая: покупать NO, когда YES-хвост стоит дороже
честной цены.

Механика на бинарном рынке [Yes, No], цены [p, 1-p]:
  - хвост Yes переоценён: p_mkt > честной вероятности;
  - тогда No недооценён: 1-p_mkt < 1 - p_fair;
  - покупаем No по ~(1-p_mkt), при резолюции получаем $1.

Честная вероятность Yes: min(оценка ансамбля, p_mkt·(1-bias_discount)) —
берём меньшее из «что говорит оценщик» и «систематическая поправка на bias».
Даже без активного сигнала bias_discount делает хвост фейдабельным.

Переиспользует оценщик, портфельные лимиты (Kelly + кэпы) и исполнитель.
Только maker-лимитки (fee 0). Идемпотентность — на стороне executor.
Позиции держатся до резолюции (выигрыш маленький и частый).
"""

from __future__ import annotations

import logging

from .config import BotConfig
from .executor import Executor
from .ledger import Ledger
from .models import Candidate, Estimate, TradePlan
from .portfolio import Portfolio, classify_category

log = logging.getLogger(__name__)


class FadeStrategy:
    def __init__(self, cfg: BotConfig, ledger: Ledger, portfolio: Portfolio,
                 executor: Executor, mode: str):
        self._cfg = cfg.fade
        self._ledger = ledger
        self._portfolio = portfolio
        self._executor = executor
        self._mode = mode

    def plan(self, estimate: Estimate) -> TradePlan | None:
        """Строит план покупки NO для переоценённого YES-хвоста (или None)."""
        cfg = self._cfg
        c = estimate.candidate
        market = c.market
        p_mkt_yes = estimate.p_mkt

        if not cfg.min_tail_price <= p_mkt_yes <= cfg.fade_max_price:
            return None
        if c.outcome_index not in (0, 1) or len(market.clob_token_ids) < 2:
            return None
        # Если оценщик считает хвост НЕдооценённым — это лонгшот-покупка, не фейд.
        # Допуск на float-шум: без сигнала якорь даёт p_est ≈ p_mkt.
        if estimate.p_est > p_mkt_yes + 1e-9:
            return None

        # Честная вероятность Yes с поправкой на систематический bias.
        p_fair_yes = min(estimate.p_est, p_mkt_yes * (1.0 - cfg.bias_discount))
        p_fair_no = 1.0 - p_fair_yes
        entry_no = 1.0 - p_mkt_yes                    # рыночная цена No
        no_index = 1 - c.outcome_index
        no_token = market.clob_token_ids[no_index]

        # Edge на No-стороне. Фейд ставится maker-лимиткой (fee ~0, возможен
        # rebate), поэтому комиссию не вычитаем; min_edge_after_fees — буфер
        # под проскальзывание и неполный fill.
        edge = p_fair_no - entry_no                   # = p_mkt_yes - p_fair_yes
        if edge < cfg.min_edge_after_fees:
            return None

        category = classify_category(market.question, market.category)
        size = self._portfolio.size_usd(category, p_fair_no, entry_no,
                                        market.min_order_size * entry_no)
        if size is None:
            return None

        # Синтетическая оценка No-стороны для сайзинга/леджера.
        no_est = Estimate(
            candidate=Candidate(market=market, outcome_index=no_index,
                                token_id=no_token, p_mkt=entry_no),
            p_mkt=entry_no, p_est=p_fair_no, signals=estimate.signals,
        )
        # Не платить за No выше, чем остаётся минимальный edge.
        price_cap = min(p_fair_no - cfg.min_edge_after_fees, 0.99)
        return TradePlan(estimate=no_est, category=category,
                         size_usd=size, limit_price_cap=price_cap)

    def cycle(self, estimates: list[Estimate]) -> int:
        """Фейдит переоценённые хвосты среди оценённых кандидатов. -> входов."""
        if not self._cfg.enabled:
            return 0
        entered = 0
        for est in estimates:
            plan = self.plan(est)
            if plan is None:
                continue
            result = self._executor.execute(plan, strategy="fade")
            if result.status == "filled":
                entered += 1
                m = plan.estimate.candidate.market
                log.info("ФЕЙД No %.3f x %.0f = $%.2f (честн. Yes %.3f vs рынок %.3f) [%s]",
                         result.avg_price, result.filled_size,
                         result.avg_price * result.filled_size,
                         1 - plan.estimate.p_est, est.p_mkt, m.question[:50])
        if entered:
            log.info("фейдов за цикл: %d", entered)
        return entered
