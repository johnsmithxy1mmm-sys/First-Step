/**
 * Reproducer: the signal track record measures forward return at "whenever the
 * engine next ran", not at the signal's stated horizon.
 *
 * A signal with a 24h horizon that becomes due while the process is down is
 * scored against the price at restart — days or weeks later — and the result is
 * published as the 24h forward return for that signal type.
 */
import BetterSqlite3 from "better-sqlite3";

const BASE = "/home/user/Polymarket-Mint-Bot/hypersignal-mcp/dist";
const { SignalStore } = await import(`${BASE}/store/signalStore.js`);
const { AlertEngine } = await import(`${BASE}/alerts/engine.js`);

const db = new BetterSqlite3(":memory:");
const signals = new SignalStore(db);

const DAY = 86_400_000;
const t0 = Date.parse("2026-06-01T00:00:00Z");

// A 24h-horizon LONG signal emitted at t0 with BTC at $100,000.
signals.record({
  type: "whale_net_flip",
  coin: "BTC",
  direction: "long",
  refPx: 100_000,
  horizonMinutes: 1440, // 24 hours
  ts: t0,
});

// The engine was down. It comes back 30 DAYS later, and by then BTC is $160,000
// (a move that has nothing to do with the 24h window the signal claimed).
const restart = t0 + 30 * DAY;
const PRICE_NOW = 160_000;

const hl = {
  metaAndAssetCtxs: async () => [
    { universe: [{ name: "BTC", szDecimals: 5, maxLeverage: 50 }] },
    [
      {
        markPx: String(PRICE_NOW),
        oraclePx: String(PRICE_NOW),
        midPx: String(PRICE_NOW),
        funding: "0.00001",
        openInterest: "1",
        dayNtlVlm: "1",
        prevDayPx: String(PRICE_NOW),
      },
    ],
  ],
  userFillsByTime: async () => [],
};

const store = {
  alerts: { listActive: () => [], updateState() {}, recordFired() {} },
  signals,
  snapshots: { record() {}, nearest: () => undefined, keys: () => [] },
  scores: { due: () => [], setOutcome() {}, markAttempt() {} },
};

const engine = new AlertEngine(hl, store, { sign: () => ({ signature: "x" }) }, {});
await engine.tick(restart);

const rec = signals.trackRecord();
const row = rec.find((r) => r.type === "whale_net_flip");

console.log("SIGNAL:");
console.log("  emitted at      2026-06-01, ref price $100,000, horizon 24h");
console.log("  therefore its forward return should be measured at 2026-06-02");
console.log("");
console.log("ENGINE:");
console.log("  next ran 30 days later, at which point BTC = $160,000");
console.log("");
console.log("PUBLISHED TRACK RECORD:");
console.log("  type:", row.type, "| scored:", row.scored, "| hitRate:", row.hitRatePct + "%");
console.log("  avgReturnPct:", row.avgReturnPct + "%  <-- presented as the 24h forward return");
console.log("");
const claimed24h = row.avgReturnPct;
console.log(
  claimed24h === 60
    ? "RESULT: REPRODUCED — a 30-day move (+60%) is published as this signal's 24h forward return."
    : `RESULT: not reproduced (got ${claimed24h})`,
);
console.log("");
console.log("No staleness guard exists: dueForScoring() returns everything past its horizon,");
console.log("and scoreDueSignals() prices it at the CURRENT tick price regardless of how late.");
