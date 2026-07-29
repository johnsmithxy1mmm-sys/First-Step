'use client';

/**
 * §5.4 requires an honest disclaimer at agent-key creation and a revoke
 * control. Neither is live in Phase 3 — there are no agent keys yet — but
 * the text is written now, because the thing it has to say is uncomfortable
 * and is exactly the sort of copy that gets softened when it is written
 * later, next to a conversion funnel.
 */

export function AgentKeyDisclaimer() {
  return (
    <section className="mt-5 rounded-lg border border-neutral-300 bg-neutral-100 p-5">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-neutral-600">
        About agent keys (Phase 4, not yet active)
      </h2>
      <div className="mt-3 space-y-2 text-sm leading-relaxed text-neutral-700">
        <p>
          When execution arrives, placing an order needs an agent key. Here is what one can
          and cannot do, stated plainly:
        </p>
        <ul className="ml-5 list-disc space-y-1 text-sm">
          <li>
            It <strong>cannot withdraw</strong> your funds. It has no transfer permission.
          </li>
          <li>
            It <strong>can trade</strong>. That is not the same as being harmless: an agent
            key that leaked could drain the account through losing positions without ever
            moving a coin off it.
          </li>
          <li>Each user gets their own key. Keys are never shared and never logged.</li>
          <li>You can revoke it at any time, and revocation takes effect immediately.</li>
        </ul>
        <p className="text-xs text-neutral-500">
          The fee is 0.02% of notional, charged by the protocol on execution only. There is
          no subscription. If a risk estimate is stale, execution is blocked — the product
          does not take a fee on a trade it cannot currently price the risk of.
        </p>
      </div>
      <button
        disabled
        className="mt-4 cursor-not-allowed rounded border border-neutral-400 px-3 py-1.5 text-sm text-neutral-500"
        title="No agent key exists yet; this becomes active in Phase 4"
      >
        Revoke agent key
      </button>
    </section>
  );
}
