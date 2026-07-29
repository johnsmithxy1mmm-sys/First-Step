'use client';

/**
 * Wallet connection and the system-health strip.
 *
 * Phase 3 is read-only, so this reads an address and nothing more. The
 * wagmi/viem wiring and `approveBuilderFee` belong to Phase 4; putting a
 * signing path in now would mean shipping a code path to production that
 * nothing yet gates (§5.4's invariant is that exactly one handler can sign).
 *
 * The health strip is here rather than tucked away because §6 makes the
 * state of the data a first-class part of the screen: if the engine is down,
 * that is the most important thing on the page.
 */

import { useState } from 'react';

export function WalletBar({
  health,
}: {
  health: { ok: boolean; syntheticData?: boolean; matrixAgeMs?: number | null } | null;
}) {
  const [address, setAddress] = useState('');

  return (
    <div className="space-y-3">
      {health && !health.ok ? (
        <div className="rounded border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-900">
          <strong>Risk engine unavailable.</strong> No risk numbers are being shown, and
          execution is blocked, until it returns.
        </div>
      ) : null}
      {health?.syntheticData ? (
        <div className="rounded border border-blue-300 bg-blue-50 px-4 py-2 text-xs text-blue-900">
          Running on <strong>synthetic market data</strong>. These numbers describe a
          simulated market, not Hyperliquid.
        </div>
      ) : null}

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">Portfolio risk</h1>
          <p className="text-xs text-neutral-500">
            Read-only. Nothing here can move your funds.
          </p>
        </div>
        <label className="text-sm">
          <span className="mr-2 text-xs text-neutral-500">Account address</span>
          <input
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            placeholder="0x… (your main account, not an agent address)"
            className="w-80 rounded border border-neutral-300 px-2 py-1 font-mono text-xs"
          />
        </label>
      </div>
    </div>
  );
}
