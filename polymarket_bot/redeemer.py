"""Auto-redemption of won positions: turn resolved winnings back into USDC.

A winning outcome token pays $1 only after `redeemPositions` is called on the
Conditional Tokens contract — until someone does that, the capital sits frozen
in an already-decided position. The data-api marks such positions with
`redeemable: true`. This module:

  1. scans the wallet's positions (live mode only — paper has no on-chain
     funds; the ledger side of paper settlement is handled by
     `_settle_resolutions`),
  2. ALERTS with the claimable total (deduped: only when the total changes),
  3. optionally (execute: true + web3 installed + key present) sends the
     `redeemPositions` transaction itself.

Honest limits, on purpose:
  * Neg-risk positions redeem through the NegRiskAdapter with a different
    call; automating a wrong contract call loses real money, so those are
    ALERTED for a manual claim in the UI, not auto-redeemed.
  * On-chain execution is best-effort: the first live claim should be
    verified by hand before trusting the automation.
"""

from __future__ import annotations

import logging
import os

from pydantic import BaseModel

from .config import BotConfig
from .monitor import alert

log = logging.getLogger(__name__)

# Polygon mainnet (chain id 137).
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
PARENT_COLLECTION_ID = "0x" + "00" * 32
# redeemPositions(address collateralToken, bytes32 parentCollectionId,
#                 bytes32 conditionId, uint256[] indexSets)
REDEEM_ABI = [{
    "name": "redeemPositions", "type": "function",
    "stateMutability": "nonpayable", "outputs": [],
    "inputs": [
        {"name": "collateralToken", "type": "address"},
        {"name": "parentCollectionId", "type": "bytes32"},
        {"name": "conditionId", "type": "bytes32"},
        {"name": "indexSets", "type": "uint256[]"},
    ],
}]


class Redeemable(BaseModel):
    condition_id: str
    title: str = ""
    size: float = 0.0            # winning shares -> $1 each at redemption
    neg_risk: bool = False

    @property
    def value_usd(self) -> float:
        return self.size


class Redeemer:
    def __init__(self, cfg: BotConfig, trader, mode: str):
        self._cfg = cfg.redemption
        self._trader = trader          # None outside live mode
        self._mode = mode
        self._last_alerted_usd = -1.0  # dedupe: alert only when the total changes

    # --- detection ---

    def scan(self) -> list[Redeemable]:
        """Redeemable positions from the wallet (data-api `redeemable` flag)."""
        if self._trader is None:
            return []
        out: list[Redeemable] = []
        for p in self._trader.api_positions():
            if not p.get("redeemable"):
                continue
            try:
                size = float(p.get("size") or 0.0)
            except (TypeError, ValueError):
                continue
            cond = str(p.get("conditionId") or "")
            if size <= 0 or not cond:
                continue
            item = Redeemable(
                condition_id=cond,
                title=str(p.get("title") or p.get("slug") or "")[:80],
                size=size,
                neg_risk=bool(p.get("negativeRisk") or p.get("negRisk")),
            )
            if item.value_usd >= self._cfg.min_claim_usd:
                out.append(item)
        return out

    # --- on-chain execution (optional) ---

    def redeem(self, item: Redeemable) -> bool:
        """Sends redeemPositions for one condition. Returns True on success."""
        if item.neg_risk:
            # Different adapter + calldata; a wrong call burns gas or worse.
            log.info("redeem: %s is neg-risk — manual claim (UI)", item.title)
            return False
        private_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
        rpc_url = self._cfg.rpc_url or os.environ.get("POLYGON_RPC_URL", "")
        if not private_key or not rpc_url:
            log.warning("redeem: POLYMARKET_PRIVATE_KEY / POLYGON_RPC_URL missing")
            return False
        try:
            from web3 import Web3
        except ImportError:
            log.warning("redeem: pip install web3 for on-chain redemption")
            return False
        try:
            w3 = Web3(Web3.HTTPProvider(rpc_url))
            account = w3.eth.account.from_key(private_key)
            ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF_ADDRESS),
                                  abi=REDEEM_ABI)
            tx = ctf.functions.redeemPositions(
                Web3.to_checksum_address(USDC_ADDRESS),
                bytes.fromhex(PARENT_COLLECTION_ID[2:]),
                bytes.fromhex(item.condition_id.removeprefix("0x")),
                [1, 2],        # both partitions: redeems whatever we hold
            ).build_transaction({
                "from": account.address,
                "nonce": w3.eth.get_transaction_count(account.address),
            })
            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            ok = receipt.get("status") == 1
            (log.info if ok else log.error)(
                "redeem %s: tx %s status=%s", item.title, tx_hash.hex(), receipt.get("status"))
            return ok
        except Exception as exc:
            log.error("redeem failed %s: %s", item.title[:40], exc)
            return False

    # --- cycle ---

    def cycle(self, allow_execute: bool = True) -> list[Redeemable]:
        if not self._cfg.enabled:
            return []
        items = self.scan()
        total = sum(i.value_usd for i in items)
        if not items:
            self._last_alerted_usd = -1.0
            return []
        if abs(total - self._last_alerted_usd) > 0.01:
            self._last_alerted_usd = total
            manual = sum(1 for i in items if i.neg_risk)
            note = (" (auto-claim on)" if self._cfg.execute and allow_execute
                    else " — claim in the UI or set redemption.execute: true")
            extra = f"; {manual} neg-risk need a manual claim" if manual else ""
            alert(f"REDEEMABLE: ${total:,.2f} across {len(items)} resolved "
                  f"positions{extra}{note}")
        if self._cfg.execute and allow_execute:
            for item in items:
                if not item.neg_risk and self.redeem(item):
                    alert(f"REDEEMED ${item.value_usd:,.2f} — {item.title}")
        return items
