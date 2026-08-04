/**
 * §6 on the DISPLAY side.
 *
 * The backend's contract decides freshness when it is asked. This file is
 * about what happens when nobody can ask it — the case §9's Phase 3
 * acceptance never covered, because it was verified by stopping the RISK
 * SERVICE (which the backend survives and reports) rather than by cutting
 * the browser off from the BACKEND.
 *
 * This package had no test runner at all until 2026-08-04, while carrying a
 * rule §10 names as not to be softened.
 */

import { describe, expect, it } from 'vitest';

import {
  CLIENT_HIDE_AFTER_MS,
  expireLocally,
  hasValue,
  type Guarded,
} from '../lib/contract';

const fresh = (): Guarded<{ p: number }> => ({
  freshness: 'fresh',
  value: { p: 0.1 },
  computedAt: '2026-08-04T10:00:00.000Z',
  ageMs: 5_000,
  degraded: false,
  execution: { allowed: true, reasons: [] },
});

describe('a value the client cannot refresh', () => {
  it('is shown while the refresh gap is inside the limit', () => {
    const t0 = 1_000_000;
    const out = expireLocally(fresh(), t0, t0 + CLIENT_HIDE_AFTER_MS - 1);
    expect(out && hasValue(out)).toBe(true);
  });

  it('is withheld once the gap reaches it', () => {
    const t0 = 1_000_000;
    const out = expireLocally(fresh(), t0, t0 + CLIENT_HIDE_AFTER_MS);
    expect(out && hasValue(out)).toBe(false);
    expect(out?.freshness).toBe('unavailable');
  });

  it('closes the execution gate with it', () => {
    /**
     * The gate has to be read off the EXPIRED payload, not the raw one. A
     * gate computed from data nobody can vouch for is a gate answering the
     * wrong question — and it answers `allowed: true`, which is §6's
     * forbidden direction.
     */
    const t0 = 1_000_000;
    const out = expireLocally(fresh(), t0, t0 + CLIENT_HIDE_AFTER_MS + 1);
    expect(out?.execution.allowed).toBe(false);
  });

  it('says why, rather than just going blank', () => {
    const t0 = 1_000_000;
    const out = expireLocally(fresh(), t0, t0 + 600_000);
    expect((out as { reason: string }).reason).toMatch(/could not be refreshed/);
  });

  it('leaves a payload alone before the first successful poll', () => {
    // `receivedAt === null` means nothing has landed yet; there is no gap to
    // measure and inventing one would withhold a value that was never shown.
    expect(expireLocally(fresh(), null, 9e12)).not.toBeNull();
    expect(expireLocally(null, 1_000, 9e12)).toBeNull();
  });

  it('matches the backend limit it stands in for', () => {
    // The backend hides at 300s. A client that held on longer would be the
    // §9 criterion failing by a different route than the one it was tested
    // against.
    expect(CLIENT_HIDE_AFTER_MS).toBe(300_000);
  });
});
