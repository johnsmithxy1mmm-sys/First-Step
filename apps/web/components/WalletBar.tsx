'use client';

/**
 * Wallet connection and the system-health strip.
 *
 * Phase 3 is read-only, so this reads an address and nothing more — via the
 * wallet's `eth_requestAccounts` or by pasting one, and the paste path is a
 * feature rather than a fallback: a risk readout is legitimate for any
 * address you can see on-chain, and it works in a browser with no wallet at
 * all. Signing (`approveBuilderFee`, agent keys) is Phase 4; there is
 * deliberately zero signing code in this app until then (§5.4).
 *
 * The health strip is here rather than tucked away because §6 makes the
 * state of the data a first-class part of the screen: if the engine is down,
 * that is the most important thing on the page.
 */

import { useEffect, useMemo, useState } from 'react';

import { ADDRESS_RE, connectWallet, detectProvider, onAccountsChanged } from '@/lib/wallet';

export function WalletBar({
  health,
  address,
  onAddress,
}: {
  health: { ok: boolean; syntheticData?: boolean; matrixAgeMs?: number | null } | null;
  /** The address whose book the page is showing; '' = demo book. */
  address: string;
  onAddress: (address: string) => void;
}) {
  // The raw input text is local; only a well-formed address is reported up,
  // so the page never sends half-typed input to the backend.
  const [text, setText] = useState(address);
  const [walletError, setWalletError] = useState<string | null>(null);
  const provider = useMemo(() => detectProvider(), []);

  useEffect(() => {
    setText(address);
  }, [address]);

  useEffect(() => {
    if (!provider) return;
    // Follow account switches; an empty list is a disconnect, which drops
    // the page back to the demo book rather than showing the last account's
    // risk under a wallet that no longer vouches for it.
    return onAccountsChanged(provider, (next) => onAddress(next ?? ''));
  }, [provider, onAddress]);

  const connect = async () => {
    if (!provider) return;
    setWalletError(null);
    try {
      const account = await connectWallet(provider);
      if (account) onAddress(account);
      else setWalletError('the wallet returned no usable account');
    } catch (error) {
      // A rejected prompt is a normal outcome, not a failure to escalate.
      setWalletError(error instanceof Error ? error.message : String(error));
    }
  };

  const applyTyped = () => {
    const trimmed = text.trim().toLowerCase();
    if (trimmed === '') {
      onAddress('');
    } else if (ADDRESS_RE.test(trimmed)) {
      onAddress(trimmed);
    }
  };
  const typedInvalid = text.trim() !== '' && !ADDRESS_RE.test(text.trim());

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
          simulated market, not Hyperliquid — and a real address&apos;s book cannot be
          fetched in this mode.
        </div>
      ) : null}

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">Portfolio risk</h1>
          <p className="text-xs text-neutral-500">
            Read-only. Nothing here can move your funds.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <label className="text-sm">
            <span className="mr-2 text-xs text-neutral-500">Account address</span>
            <input
              value={text}
              onChange={(e) => setText(e.target.value)}
              onBlur={applyTyped}
              onKeyDown={(e) => {
                if (e.key === 'Enter') applyTyped();
              }}
              placeholder="0x… (your main account, not an agent address)"
              className={`w-80 rounded border px-2 py-1 font-mono text-xs ${
                typedInvalid ? 'border-amber-400 bg-amber-50' : 'border-neutral-300'
              }`}
            />
          </label>
          {provider ? (
            <button
              onClick={() => void connect()}
              className="rounded border border-neutral-300 bg-white px-3 py-1 text-xs font-medium hover:bg-neutral-50"
            >
              {address ? 'Reconnect wallet' : 'Connect wallet'}
            </button>
          ) : null}
          {address ? (
            <button
              onClick={() => onAddress('')}
              title="Back to the demo book"
              className="rounded border border-neutral-300 bg-white px-3 py-1 text-xs text-neutral-500 hover:bg-neutral-50"
            >
              Clear
            </button>
          ) : null}
        </div>
      </div>

      {typedInvalid ? (
        <p className="text-xs text-amber-700">
          Not a valid address yet — 42 characters, 0x then 40 hex digits. The demo book
          stays on screen until one is.
        </p>
      ) : null}
      {walletError ? <p className="text-xs text-red-700">Wallet: {walletError}</p> : null}
      <p className="text-xs text-neutral-400">
        {address ? (
          <>
            Showing the live book of <span className="font-mono">{address}</span>. If this
            account looks unexpectedly flat, you may have connected an agent address — the
            venue answers for those with an empty book indistinguishable from a flat one.
          </>
        ) : (
          <>Showing a demo book. Connect a wallet or paste an address to see a real one.</>
        )}
      </p>
    </div>
  );
}
