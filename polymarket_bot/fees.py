"""Fee Structure V2 (March 2026): per-category taker fees, maker rebate.

The maker pays 0 and earns a 20-50% rebate on taker fees — the whole bot
economy is built on the maker side. Rates are read from config.yaml (fees:),
NOT hardcoded in logic: if the fee structure changes, edit the config.
"""

from __future__ import annotations

from .config import FeesConfig

# Mapping of our portfolio categories to the fee-table categories.
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
        # Longest keys first: otherwise "geopolitics" falsely matches "politics".
        for key in sorted(self._cfg.taker, key=len, reverse=True):
            if key != "other" and key in g:
                return key
        return _CATEGORY_TO_FEE_KEY.get(category, "other")

    def taker_fee(self, category: str, gamma_category: str = "") -> float:
        key = self.fee_key(category, gamma_category)
        return self._cfg.taker.get(key, self._cfg.taker.get("other", 0.04))

    def maker_rebate(self, category: str, gamma_category: str = "") -> float:
        """Maker rebate as a fraction of this category taker fee."""
        return self.taker_fee(category, gamma_category) * self._cfg.maker_rebate_frac

    def net_taker_edge(self, gross_edge: float, category: str,
                       gamma_category: str = "") -> float:
        """Edge of the aggressive (taker) leg after the fee."""
        return gross_edge - self.taker_fee(category, gamma_category)

    def mm_min_half_spread(self, category: str, gamma_category: str = "",
                           min_edge_after_fees: float = 0.01) -> float:
        """Minimum quote half-spread for a bid+ask pair to be profitable.

        Pair profit = full spread + rebate on both sides; require
        >= min_edge_after_fees. Maker fee = 0.
        """
        rebate = self.maker_rebate(category, gamma_category)
        return max((min_edge_after_fees - 2 * rebate) / 2, 0.0)
