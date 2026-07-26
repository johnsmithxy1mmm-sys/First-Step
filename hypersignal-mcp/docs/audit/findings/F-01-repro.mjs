/**
 * Reproducer: hl_polymarket_divergence fabricates a large, top-ranked "edge"
 * from a Polymarket question that is not a USD price threshold at all.
 *
 * Runs the REAL tool against a local mock Gamma API + a stub Hyperliquid
 * client. No network to Hyperliquid or Polymarket.
 */
import http from "node:http";

const BASE = "/home/user/Polymarket-Mint-Bot/hypersignal-mcp/dist";

// --- mock Polymarket Gamma API -------------------------------------------
const MARKETS = [
  {
    question: "Will Bitcoin dominance rise above 60%?",
    slug: "btc-dominance-60",
    endDate: "2026-12-31T00:00:00Z",
    active: true,
    closed: false,
    outcomes: '["Yes","No"]',
    outcomePrices: '["0.50","0.50"]',
    liquidityNum: 900000,
    volumeNum: 5000000,
  },
  {
    question: "Will BTC be above $150,000 on Dec 31 2026?",
    slug: "btc-150k",
    endDate: "2026-12-31T00:00:00Z",
    active: true,
    closed: false,
    outcomes: '["Yes","No"]',
    outcomePrices: '["0.25","0.75"]',
    liquidityNum: 800000,
    volumeNum: 4000000,
  },
];

const srv = http.createServer((req, res) => {
  res.writeHead(200, { "content-type": "application/json" });
  res.end(JSON.stringify(MARKETS));
});
await new Promise((r) => srv.listen(0, "127.0.0.1", r));
const port = srv.address().port;
process.env.POLYMARKET_GAMMA_URL = `http://127.0.0.1:${port}`;

// --- stub Hyperliquid: BTC at $90,000, flat candles => modest vol ---------
const closes = [];
let px = 90000;
for (let i = 0; i < 40; i++) {
  px *= 1 + (i % 2 === 0 ? 0.01 : -0.0098); // ~mild oscillation
  closes.push(px);
}
const ctx = {
  config: {
    polymarket: { gammaUrl: process.env.POLYMARKET_GAMMA_URL },
    requestTimeoutMs: 5000,
  },
  hl: {
    metaAndAssetCtxs: async () => [
      { universe: [{ name: "BTC", szDecimals: 5, maxLeverage: 50 }] },
      [{ markPx: "90000", oraclePx: "90000", midPx: "90000", funding: "0.00001", openInterest: "1", dayNtlVlm: "1", prevDayPx: "89000" }],
    ],
    candles: async () =>
      closes.map((c, i) => ({ t: i * 86400000, o: String(c), h: String(c), l: String(c), c: String(c), v: "1", n: 1 })),
  },
};

const { polymarketDivergence } = await import(`${BASE}/tools/premium/polymarketDivergence.js`);
const out = await polymarketDivergence.run({ coin: "BTC", minEdge: 0.05, limit: 10 }, ctx);

console.log("SUMMARY THE AGENT RECEIVES:");
console.log("  " + out.summary);
console.log("");
console.log("RANKED DIVERGENCES (order = what the agent acts on first):");
out.data.divergences.forEach((d, i) => {
  console.log(`  #${i + 1}  edge=${(d.edge * 100).toFixed(1)}pp  threshold=$${d.thresholdUsd}  mode=${d.mode}`);
  console.log(`      HL implied=${d.hlImpliedProb}  Polymarket=${d.polymarketYesProb}`);
  console.log(`      "${d.question}"`);
});

const top = out.data.divergences[0];
console.log("");
console.log("ASSERTION: the #1 ranked opportunity is a non-price market misread as a $60 threshold");
console.log("  top question :", JSON.stringify(top?.question));
console.log("  parsed as    : $" + top?.thresholdUsd + " threshold on BTC (spot $90,000)");
console.log("  HL says P =", top?.hlImpliedProb, "(certainty, because BTC is trivially above $60)");
console.log("  reported edge:", (top?.edge * 100).toFixed(1) + "pp  <-- fabricated");
console.log("");
console.log(
  top && top.thresholdUsd === 60 && Math.abs(top.edge) > 0.4
    ? "RESULT: REPRODUCED — fabricated edge ranks first"
    : "RESULT: not reproduced",
);

srv.close();
