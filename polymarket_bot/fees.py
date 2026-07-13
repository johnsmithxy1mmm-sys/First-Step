"""Комиссии Fee Structure V2 (март 2026): taker fees по категориям, maker rebate.

Maker платит 0 и получает rebate 20-50% от taker fees — вся экономика бота
строится на maker-стороне. Ставки читаются из config.yaml (fees:), НЕ
хардкодятся в логике: при изменении структуры комиссий правится конфиг.
"""

from __future__ import annotations

from .config import FeesConfig

# Маппинг наших категорий портфеля на категории fee-таблицы.
_CATEGORY_TO_FEE_KEY = {
    "crypto": "crypto",
    "sports": "sports",
    "elections": "politics",
    "economy": "economics",
    "nature": "weather",
    "geopolitics": "geopolitics",
    "other": "other",
}


class FeeModel:
    def __init__(self, cfg: FeesConfig):
        self._cfg = cfg

    def fee_key(self, category: str, gamma_category: str = "") -> str:
        g = gamma_category.lower()
        # От длинных ключей к коротким: иначе "geopolitics" ложно матчит "politics".
        for key in sorted(self._cfg.taker, key=len, reverse=True):
            if key != "other" and key in g:
                return key
        return _CATEGORY_TO_FEE_KEY.get(category, "other")

    def taker_fee(self, category: str, gamma_category: str = "") -> float:
        key = self.fee_key(category, gamma_category)
        return self._cfg.taker.get(key, self._cfg.taker.get("other", 0.04))

    def maker_rebate(self, category: str, gamma_category: str = "") -> float:
        """Rebate мейкеру как доля от taker fee этой категории."""
        return self.taker_fee(category, gamma_category) * self._cfg.maker_rebate_frac

    def net_taker_edge(self, gross_edge: float, category: str,
                       gamma_category: str = "") -> float:
        """Edge агрессивной (taker) ноги после комиссии."""
        return gross_edge - self.taker_fee(category, gamma_category)

    def mm_min_half_spread(self, category: str, gamma_category: str = "",
                           min_edge_after_fees: float = 0.01) -> float:
        """Минимальный полуспред котировки, чтобы пара bid+ask была прибыльна.

        Прибыль пары = полный спред + rebate обеих сторон; требуем
        >= min_edge_after_fees. Maker fee = 0.
        """
        rebate = self.maker_rebate(category, gamma_category)
        return max((min_edge_after_fees - 2 * rebate) / 2, 0.0)
