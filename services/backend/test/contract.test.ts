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
  guardInputs,
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


describe('a clock that disagrees cannot delete the contract', () => {
  /**
   * `ageMs = now - computedAt`, and under this contract's own model
   * `computedAt` is always in the past — so a NEGATIVE age is not a very
   * fresh number, it is evidence the two clocks disagree.
   *
   * It read as fresh. A stamp an hour ahead gave `freshness: 'fresh'`,
   * `ageMs: -3600000` and an execution gate that opened with no reasons,
   * whatever the data's real age. `guard`'s own docstring says a slow hop
   * must not launder a stale number into a fresh one; a skewed clock did not
   * launder the number, it removed the mechanism — and §6's execution block
   * is the rule §10 forbids softening.
   */
  it('withholds a value stamped in the future rather than calling it fresh', () => {
    const now = T0.getTime();
    for (const skewMs of [60_000, 3_600_000, 86_400_000]) {
      const g = guard({ p: 1 }, new Date(now + skewMs), { now });
      expect(g.freshness).toBe('unavailable');
      expect(hasValue(g)).toBe(false);
      expect((g as { reason: string }).reason).toMatch(/FUTURE/);
    }
  });

  it('tolerates the sub-second disagreement two healthy containers have', () => {
    const now = T0.getTime();
    expect(guard({ p: 1 }, new Date(now + 500), { now }).freshness).toBe('fresh');
  });

  it('closes the execution gate on a future-stamped value', () => {
    const now = T0.getTime();
    const risk = guard({ p: 1 }, new Date(now + 3_600_000), { now });
    expect(executionGate({ risk, publishable: true, matrixAgeMs: 0 }).allowed).toBe(false);
  });

  it('closes it for a future-stamped MATRIX too', () => {
    const now = T0.getTime();
    const risk = guard({ p: 1 }, new Date(now - 1_000), { now });
    const gate = executionGate({ risk, publishable: true, matrixAgeMs: -3_600_000 });
    expect(gate.allowed).toBe(false);
    expect(gate.reasons.join(' ')).toMatch(/future/);
  });

  it('catches a skewed input before severity ranking can hide it', () => {
    /**
     * A future-stamped input scores as the FRESHEST of the set, so ranking by
     * severity would hand the verdict to some other input and let this one
     * through unremarked. It has to be checked before the ranking rather than
     * inside it.
     */
    const now = T0.getTime();
    const g = guardInputs({ p: 1 }, [
      { label: 'book', at: new Date(now + 3_600_000), staleAfterMs: 60_000, hideAfterMs: 300_000 },
      { label: 'prices', at: new Date(now - 5_000), staleAfterMs: 60_000, hideAfterMs: 300_000 },
    ], { now });
    expect(g.freshness).toBe('unavailable');
    expect((g as { reason: string }).reason).toMatch(/book/);
  });
});
