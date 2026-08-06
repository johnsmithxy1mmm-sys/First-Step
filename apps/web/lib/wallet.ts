/**
 * Minimal EIP-1193 wallet access. Read-only on purpose.
 *
 * Phase 3 needs exactly one thing from a wallet: the account address, so the
 * engine can fetch that account's live book. `eth_requestAccounts` is the
 * whole surface — no signing, no transactions, no chain switching, and no
 * wagmi/viem dependency for what is one `request` call and one event
 * listener. The signing path (`approveBuilderFee`, agent keys) is Phase 4,
 * gated on shadow validation, and deliberately does not exist here: §5.4's
 * invariant is that exactly one handler can sign, and the cheapest way to
 * hold that invariant today is for the frontend to contain zero signing
 * code.
 *
 * A Hyperliquid account is an EVM address, so the connected address is
 * usable directly. The one caveat the UI must carry (§5.1): an AGENT
 * address returns a well-formed empty book identical to a flat account's,
 * and nothing anywhere can detect the difference — the user has to connect
 * their main account.
 */

export interface Eip1193Provider {
  request(args: { method: string; params?: unknown[] }): Promise<unknown>;
  on?(event: string, handler: (...args: unknown[]) => void): void;
  removeListener?(event: string, handler: (...args: unknown[]) => void): void;
}

export function detectProvider(): Eip1193Provider | null {
  if (typeof window === 'undefined') return null;
  const eth = (window as { ethereum?: Eip1193Provider }).ethereum;
  return eth ?? null;
}

export const ADDRESS_RE = /^0x[0-9a-fA-F]{40}$/;

/** Prompt the wallet for access; resolves to the selected address or null. */
export async function connectWallet(provider: Eip1193Provider): Promise<string | null> {
  const accounts = (await provider.request({
    method: 'eth_requestAccounts',
  })) as string[] | null;
  const first = accounts?.[0] ?? null;
  return first && ADDRESS_RE.test(first) ? first.toLowerCase() : null;
}

/**
 * Follow account switches in the wallet. Returns an unsubscribe function.
 * An empty accounts list means the user disconnected the site.
 */
export function onAccountsChanged(
  provider: Eip1193Provider,
  handler: (address: string | null) => void,
): () => void {
  if (!provider.on) return () => {};
  const listener = (...args: unknown[]) => {
    const accounts = args[0] as string[] | undefined;
    const first = accounts?.[0] ?? null;
    handler(first && ADDRESS_RE.test(first) ? first.toLowerCase() : null);
  };
  provider.on('accountsChanged', listener);
  return () => provider.removeListener?.('accountsChanged', listener);
}
