/**
 * §6 — the degradation contract.
 *
 * §9's Phase 3 acceptance criterion lives here and in
 * `degradation.test.ts`: values disappear within five minutes of the risk
 * service stopping, are marked stale between 60 and 300 seconds, and the
 * execution button is blocked on anything but fresh data.
 */

import { describe, expect, it } from 'vitest';

import {
  HIDE_AFTER_MS,
  MATRIX_STALE_AFTER_MS,
  STALE_AFTER_MS,
  executionGate,
  guard,
  hasValue,
  unavailable,
} from '../src/staleness/contract.js';

const T0 = new Date('2026-07-29T12:00:00.000Z');
const at = (secondsLater: number) => T0.getTime() + secondsLater * 1000;

describe('freshness boundaries (§6)', () => {
  it('is fresh below 60 seconds', () => {
    const p = guard({ v: 1 }, T0, { now: at(59) });
    expect(p.freshness).toBe('fresh');
    expect(p.degraded).toBe(false);
    expect(hasValue(p)).toBe(true);
  });

  it('is stale from 60 seconds, and says how old', () => {
    const p = guard({ v: 1 }, T0, { now: at(120) });
    expect(p.freshness).toBe('stale');
    expect(p.degraded).toBe(true);
    expect(hasValue(p) && p.ageMs).toBe(120_000);
  });

  it('is still stale — not hidden — at 299 seconds', () => {
    expect(guard({ v: 1 }, T0, { now: at(299) }).freshness).toBe('stale');
  });

  it('is hidden from 300 seconds', () => {
    const p = guard({ v: 1 }, T0, { now: at(300) });
    expect(p.freshness).toBe('hidden');
  });

  it('exposes no value once hidden, so nothing stale can be rendered', () => {
    const p = guard({ v: 1 }, T0, { now: at(600) });
    expect(hasValue(p)).toBe(false);
    // §6: values are HIDDEN, not greyed out. A greyed-out number is still a
    // number on screen, and it is the number a stressed trader reads.
    expect(p).not.toHaveProperty('value');
    expect((p as { reason: string }).reason).toContain('withheld');
  });

  it('uses the engine timestamp, so a slow hop cannot launder staleness', () => {
    // The value arrives now, but the engine computed it four minutes ago.
    const p = guard({ v: 1 }, T0, { now: at(240) });
    expect(p.freshness).toBe('stale');
  });

  it('pins the thresholds §6 actually states', () => {
    expect(STALE_AFTER_MS).toBe(60_000);
    expect(HIDE_AFTER_MS).toBe(300_000);
  });
});

describe('execution gate (§6)', () => {
  const fresh = guard({ v: 1 }, T0, { now: at(10) });
  const stale = guard({ v: 1 }, T0, { now: at(90) });
  const hidden = guard({ v: 1 }, T0, { now: at(400) });

  it('allows execution only on fresh, published data with a warm matrix', () => {
    const gate = executionGate({ risk: fresh, publishable: true, matrixAgeMs: 120_000 });
    expect(gate.allowed).toBe(true);
    expect(gate.reasons).toEqual([]);
  });

  it('blocks on stale data, not merely warns', () => {
    // §6: "the product has no right to earn a fee on an execution it cannot
    // price the risk of. Do not soften this for conversion."
    const gate = executionGate({ risk: stale, publishable: true, matrixAgeMs: 0 });
    expect(gate.allowed).toBe(false);
    expect(gate.reasons[0]).toContain('execution requires fresh data');
  });

  it('blocks when values are hidden', () => {
    expect(executionGate({ risk: hidden, publishable: true, matrixAgeMs: 0 }).allowed).toBe(
      false,
    );
  });

  it('blocks when the engine is unavailable', () => {
    const gate = executionGate({
      risk: unavailable('risk service unreachable'),
      publishable: false,
      matrixAgeMs: null,
    });
    expect(gate.allowed).toBe(false);
    expect(gate.reasons.join(' ')).toContain('unreachable');
  });

  it('blocks on an under-resolved estimate (§2.5)', () => {
    const gate = executionGate({ risk: fresh, publishable: false, matrixAgeMs: 0 });
    expect(gate.allowed).toBe(false);
    expect(gate.reasons.join(' ')).toContain('confidence interval');
  });

  it('blocks on a cold correlation matrix', () => {
    const gate = executionGate({
      risk: fresh,
      publishable: true,
      matrixAgeMs: MATRIX_STALE_AFTER_MS + 1,
    });
    expect(gate.allowed).toBe(false);
    expect(gate.reasons.join(' ')).toContain('correlation matrix');
  });

  it('does not apply the 60s book clock to the matrix', () => {
    // OPEN-QUESTIONS D4: the matrix rebuilds every five minutes by design, so
    // thresholding it on the book clock would mark the product permanently
    // stale and block execution forever.
    const gate = executionGate({ risk: fresh, publishable: true, matrixAgeMs: 299_000 });
    expect(gate.allowed).toBe(true);
  });

  it('reports every reason it blocked, not just the first', () => {
    const gate = executionGate({ risk: stale, publishable: false, matrixAgeMs: null });
    expect(gate.reasons.length).toBe(3);
  });
});
