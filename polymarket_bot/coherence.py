"""Market coherence graph: logical constraints across related markets.

A generalization of chain arbitrage from a single pair to the whole implication
GRAPH of an event. Within an event, `chainarb.classify_pair` tells us when one
market's YES logically implies another's (a subset -> superset edge: "reach
$250k" implies "reach $200k"; "cut by June" implies "cut by July"). Stacked,
those edges impose a monotone constraint on the true probabilities:

    subset implies superset  =>  P(subset) <= P(superset)      (always)

Two things fall out, both deterministic (LLM-proposed cross-template edges can
extend the graph later; the checker never trusts a proposal it can't verify):

  * VIOLATIONS — where market prices break the ordering (P(subset) priced
    ABOVE P(superset)). Every such gap is a chain-arb, now found across the
    whole chain including transitive pairs, not just adjacent ones.
  * PROJECTION — snapping the observed prices onto the nearest monotone set
    (isotonic regression, pool-adjacent-violators) yields, for EVERY market on
    the chain, a probability coherent with all its siblings. That projected
    number uses cross-market information the single-market price ignores — a
    strictly better p_est, for free.

All pure functions of (markets, prices); tested offline.
"""

from __future__ import annotations

from itertools import combinations

from pydantic import BaseModel

from .chainarb import classify_pair
from .models import Market


class ChainNode(BaseModel):
    market_id: str
    question: str
    p_market: float             # YES price as the market shows it
    p_coherent: float           # after isotonic projection along the chain


class Incoherence(BaseModel):
    event_title: str
    max_gap: float              # largest P(subset) - P(superset) on the chain
    nodes: list[ChainNode]


def _implication_edges(group: list[Market]) -> dict[str, set[str]]:
    """subset_id -> {superset_id} for every classifiable pair in the group."""
    succ: dict[str, set[str]] = {m.id: set() for m in group}
    for a, b in combinations(group, 2):
        r = classify_pair(a, b)
        if r is not None:
            subset, superset, _kind = r
            succ[subset.id].add(superset.id)
    return succ


def _transitive_closure(succ: dict[str, set[str]]) -> dict[str, set[str]]:
    closure = {k: set(v) for k, v in succ.items()}
    changed = True
    while changed:
        changed = False
        for node, outs in closure.items():
            new = set(outs)
            for m in outs:
                new |= closure.get(m, set())
            if new != outs:
                closure[node] = new
                changed = True
    return closure


def event_chain(group: list[Market]) -> list[Market] | None:
    """Order a group into a single implication chain (rarest -> commonest), or
    None if the markets are not a clean total order (a subset would need a
    strict out-degree ranking; ties mean it is a partial order we won't guess)."""
    if len(group) < 2:
        return None
    closure = _transitive_closure(_implication_edges(group))
    outdeg = {m.id: len(closure[m.id]) for m in group}
    if len(set(outdeg.values())) != len(group):
        return None                         # not a strict total order
    return sorted(group, key=lambda m: outdeg[m.id], reverse=True)


def isotonic_nondecreasing(y: list[float]) -> list[float]:
    """Pool-adjacent-violators: nearest non-decreasing fit (equal weights)."""
    # Each block: (value, count). Merge while the last block exceeds the new one.
    blocks: list[list[float]] = []
    for v in y:
        blocks.append([v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, c2 = blocks.pop()
            v1, c1 = blocks.pop()
            merged = (v1 * c1 + v2 * c2) / (c1 + c2)
            blocks.append([merged, c1 + c2])
    out: list[float] = []
    for value, count in blocks:
        out.extend([value] * int(count))
    return out


def _yes_price(m: Market) -> float:
    return m.outcome_prices[0] if m.outcome_prices else 0.0


def analyze_event(group: list[Market]) -> Incoherence | None:
    """Chain view of one event: coherent projection + the worst ordering gap."""
    chain = event_chain(group)
    if chain is None:
        return None
    raw = [_yes_price(m) for m in chain]        # non-decreasing if coherent
    fit = isotonic_nondecreasing(raw)
    max_gap = max((raw[i] - raw[i + 1] for i in range(len(raw) - 1)), default=0.0)
    nodes = [ChainNode(market_id=m.id, question=m.question,
                       p_market=round(raw[i], 4), p_coherent=round(fit[i], 4))
             for i, m in enumerate(chain)]
    return Incoherence(event_title=chain[0].event_title or chain[0].question[:40],
                       max_gap=round(max_gap, 4), nodes=nodes)


def coherent_probs(markets: list[Market]) -> dict[str, float]:
    """market_id -> coherence-projected YES probability (only chained markets)."""
    out: dict[str, float] = {}
    for group in _by_event(markets):
        inc = analyze_event(group)
        if inc is not None:
            for n in inc.nodes:
                out[n.market_id] = n.p_coherent
    return out


def incoherences(markets: list[Market], min_gap: float = 0.02) -> list[Incoherence]:
    """Events whose chain prices violate monotonicity by at least min_gap."""
    out = []
    for group in _by_event(markets):
        inc = analyze_event(group)
        if inc is not None and inc.max_gap >= min_gap:
            out.append(inc)
    out.sort(key=lambda i: i.max_gap, reverse=True)
    return out


def _by_event(markets: list[Market]) -> list[list[Market]]:
    groups: dict[str, list[Market]] = {}
    for m in markets:
        if (m.event_id and not m.closed and m.outcome_prices
                and len(m.clob_token_ids) >= 2):
            groups.setdefault(m.event_id, []).append(m)
    return [g for g in groups.values() if len(g) >= 2]


def run_coherence(cfg) -> None:  # pragma: no cover — I/O glue
    from rich.console import Console
    from rich.table import Table

    from .gamma import GammaClient

    console = Console()
    markets = GammaClient(cfg).fetch_active_markets()
    rows = incoherences(markets, min_gap=0.0)
    if not rows:
        console.print("[dim]No chained events found (need multiple ladder-related "
                      "markets in one event).[/dim]")
        return
    shown = [r for r in rows if r.max_gap > 0][:15]
    console.print(f"[bold]Coherence graph[/bold] — {len(rows)} chained events, "
                  f"{len(shown)} with a price-ordering violation\n")
    for inc in shown or rows[:10]:
        t = Table(title=f"{inc.event_title[:60]}  (max gap {inc.max_gap:.3f})")
        for col in ("Market (rarest -> commonest)", "Market P", "Coherent P"):
            t.add_column(col)
        for n in inc.nodes:
            flag = " ⚠" if abs(n.p_market - n.p_coherent) > 0.01 else ""
            t.add_row(n.question[:50], f"{n.p_market:.3f}",
                      f"{n.p_coherent:.3f}{flag}")
        console.print(t)
    console.print("[dim]Coherent P is the isotonic projection onto P(subset) <= "
                  "P(superset). A market far from its coherent P is either a "
                  "chain-arb (if the gap is executable) or a better p_est than "
                  "its own quote.[/dim]")
