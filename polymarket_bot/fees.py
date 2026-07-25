"""Polymarket fee model: per-category taker coefficients and maker rebates.

The official taker fee is

    fee_usd = theta * shares * price * (1 - price)

with `theta` a per-category coefficient (config `fees.taker`). The maker pays
0 and earns back a share of collected taker fees (config
`fees.maker_rebate_share`) -- the whole bot economy is built on the maker side.

WHY THIS MODULE IS THE ONLY PLACE THAT TURNS A COEFFICIENT INTO MONEY:
`theta` is NOT a fraction of notional. Using it as one overstates the fee by
1/(1 - price): 2x at 50c, 10x at 90c, 20x at 95c, 67x at 98.5c. That error is
invisible on cheap longshots and fatal on near-$1 strategies (resolution carry,
ladder and basket arbitrage), where it turns a real edge into a fake loss. So
callers ask for dollars-per-share and never multiply a coefficient themselves.

Note the fee is symmetric in price: theta*p*(1-p) is unchanged by p -> 1-p, so
the YES and NO sides of the same market always cost the same to take.

Rates are read from config.yaml (fees:), NOT hardcoded in logic: if the fee
structure changes, edit the config.
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

    def taker_coef(self, category: str, gamma_category: str = "") -> float:
        """The category coefficient `theta` -- NOT a fee. Use taker_fee_per_share
        for money; this is for display and for passing into a model field."""
        key = self.fee_key(category, gamma_category)
        return self._cfg.taker.get(key, self._cfg.taker.get("other", 0.04))

    def rebate_share(self, category: str, gamma_category: str = "") -> float:
        """Fraction of collected taker fees rebated to makers in this category."""
        shares = self._cfg.maker_rebate_share
        key = self.fee_key(category, gamma_category)
        return shares.get(key, shares.get("default", 0.25))

    # --- money: dollars per share ---

    @staticmethod
    def fee_per_share_from_coef(coef: float, price: float) -> float:
        """theta * p * (1 - p) -- the official per-share taker fee in dollars.

        Kept static so model classes holding a coefficient (BasketArb,
        ChainPair) can price their own legs without carrying a FeeModel.
        """
        p = min(max(price, 0.0), 1.0)
        return coef * p * (1.0 - p)

    def taker_fee_per_share(self, category: str, price: float,
                            gamma_category: str = "") -> float:
        """Dollars of taker fee per share bought at `price`."""
        return self.fee_per_share_from_coef(
            self.taker_coef(category, gamma_category), price)

    def maker_rebate_per_share(self, category: str, price: float,
                               gamma_category: str = "") -> float:
        """Dollars rebated per share filled as a MAKER at `price` (fee paid: 0)."""
        return (self.taker_fee_per_share(category, price, gamma_category)
                * self.rebate_share(category, gamma_category))

    def net_taker_edge(self, gross_edge: float, category: str, price: float,
                       gamma_category: str = "") -> float:
        """Edge of an aggressive (taker) leg after the fee, in dollars per share.
        `gross_edge` must also be dollars per share."""
        return gross_edge - self.taker_fee_per_share(category, price, gamma_category)

    def mm_min_half_spread(self, category: str, price: float,
                           gamma_category: str = "",
                           min_edge_after_fees: float = 0.01) -> float:
        """Minimum quote half-spread for a bid+ask pair to be profitable.

        Pair profit = full spread + rebate on both sides; require
        >= min_edge_after_fees. Maker fee = 0. The rebate is evaluated at the
        quoted price, so a near-$1 market -- where the rebate is nearly nothing
        -- correctly demands almost the whole spread from the spread itself.
        """
        rebate = self.maker_rebate_per_share(category, price, gamma_category)
        return max((min_edge_after_fees - 2 * rebate) / 2, 0.0)
