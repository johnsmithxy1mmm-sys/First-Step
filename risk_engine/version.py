"""Model version.

Every prediction written to the calibration journal carries this string.

§3.3 (anti-overfitting): changing anything that alters the predicted
distribution -- estimator, marginal family, copula, liquidation logic,
drift convention -- MUST bump this. The shadow-day counter is keyed on it,
so a bump resets the accumulated validation window for the new version and
forces it through champion/challenger against the incumbent. Bumping it is
therefore expensive on purpose.

Format: MAJOR.MINOR.PATCH-tag
  MAJOR/MINOR  distribution-affecting change (resets shadow counter)
  PATCH        performance, logging, plumbing (does not reset)
"""

#: 0.2.0 — audit fixes. MINOR, not PATCH, because two of them change the
#: predicted distribution and therefore reset the shadow window (§3.3):
#:   - baseline B now uses a Gaussian copula, so its predicted distribution
#:     genuinely differs (audit A-03);
#:   - horizons are checkpointed on one shared walk, so the 7d paths are no
#:     longer an independently drawn set (audit A-10).
#: The randomized PIT (A-02) changes how outcomes are *scored*, not what the
#: model predicts, but it invalidates every PIT value recorded under 0.1.x
#: just as thoroughly, so those observations cannot be pooled either.
#:
#: 0.2.1 — thread-parallel path blocks (OPEN-QUESTIONS D7). PATCH, not MINOR:
#: blocks now draw from independent spawned streams instead of sequentially
#: from one generator, so a given seed produces a different *sample*. The law
#: being sampled is identical, which is the only thing the shadow window
#: cares about, so the counter is not reset. Old seeds no longer reproduce
#: old numbers, which is a provenance note rather than a calibration one --
#: every journal row already stores the model version beside its seed.
#:
#: 0.3.0 — the copula's degrees of freedom are FITTED rather than hardcoded
#: (OPEN-QUESTIONS A9). MINOR, and this is the textbook case for it: the
#: copula df sets how strongly assets go extreme together, so changing it
#: changes the joint tail and therefore every P(liq) the model reports. On
#: the fixture the fitted value is 6.5 against the 4.0 it replaces.
#:
#: Deliberately timed. This resets the §3.3 counter, and the counter had not
#: started -- so the change cost nothing here and would have cost up to
#: twenty-one days at any point after. The alternative was spending the
#: window validating a constant nobody could source: `fit_copula_df` was
#: implemented, tested and called from nowhere, while A9 described the
#: two-stage IFM estimator in the present tense as though it were in use.
#:
#: What is NOT a version change: the fitted value moving as new returns
#: arrive. That is an estimate tracking data, exactly like the EWMA
#: volatilities and the marginal dfs beside it. The specification is what the
#: counter keys on, and the specification changed once, here.
#:
#: 0.4.0 — adversarial-audit fixes, three of which move the predicted
#: distribution. MINOR, and again taken before the clock has started:
#:
#:   - the correlation matrix now aligns coins on shared candle TIMESTAMPS
#:     rather than on trailing array index. One missing candle for one coin
#:     shifted it against the others for the whole window and measured a
#:     correlation of -0.022 where the truth was 0.9986, so this changes the
#:     dependence structure on any window with a gap -- and the whole point is
#:     that it was silently wrong before;
#:   - the copula df and the §2.3 tail gate are now fitted on the assets whose
#:     correlation row was actually ESTIMATED, over their common window, rather
#:     than on every asset truncated to the youngest one's history. On a
#:     40-hour young asset that moved the fitted df from 5.0 to 3.5;
#:   - the marginal df is rounded to one decimal, which `_neg_log_likelihood`
#:     always claimed and never did. Sub-0.1 changes in df are far below the
#:     estimation error on df itself, but they are changes.
#:
#: Also here and NOT distribution-affecting in law, though it changes the
#: sample a seed produces: bridge uniforms are keyed by coin instead of by the
#: pocket's position in the book, because `with_position` reorders pockets and
#: the paired pre-trade walk therefore disagreed with itself about which
#: column belonged to which pocket. Same law, different draw -- the 0.2.1
#: reasoning applies, and it rides along with the MINOR bump anyway.
#: 0.5.0 — the A11 conditional tail floor (OPEN-QUESTIONS A11, option A,
#: adopted by the product owner 2026-08-05). MINOR: it changes the copula df
#: the live bundle serves, and therefore the joint tail of every predicted
#: distribution. Two coupled changes, coherent by construction:
#:
#:   - `df* = min(df_ML, df_tail)`: where a pair's measured lower-tail
#:     dependence exceeds the fitted model at >=2 sigma (today: ETH/SOL at +2.7 sigma),
#:     the df is floored to the largest grid value whose model@q covers that
#:     pair's one-sided 95% lower confidence bound. Conditional per finding 3
#:     (sub-2-sigma pairs are noise, not demands), one-sided per finding 4 (lambda_U is
#:     reported, never constrained — it is non-monotone and the observed
#:     BTC/ETH upper tail is unreachable at any admissible skew). On the
#:     2026-08-05 live readings this yields df* = 2.5 against an ML fit of
#:     6.5.
#:   - the §2.3 gate's margin gains a null: `max(0.05, 1.645*SE)` instead of
#:     a flat 0.05, which at the live n=108 was 1.13*SE — the no-null
#:     criterion finding 3 measured at ~84% false positives on zero-signal
#:     data. The floor targets exactly the bound the gate now tests, so a
#:     floored bundle passes the gate by construction and recording mode
#:     (A10) ends when, and only when, the floor actually covers the demands.
#:
#: The skewed-t §2.3 prescribes was REFUTED for this venue's data before this
#: was adopted (all four A11 findings measured; see OPEN-QUESTIONS A11): not
#: per-output conservative, correlation-destroying below nu=4, and unable to
#: reach the observed BTC/ETH upper tail at any admissible skew. The df floor
#: is the surviving remedy; the rho-lever redesign (crash-regime dependence,
#: rho_tail ~ 0.944 vs EWMA 0.887) is recorded in A11 as the sharper future
#: model, deferred.
#:
#: Taken while the counter stood at zero — every observation recorded under
#: 0.4 carried `recorded_under_defect` and was never a gate-day, so this
#: reset discards nothing.
#: 0.6.0 — the demand threshold is the gate threshold (the 0.5.0 stuck band).
#: MINOR because it changes WHEN the floor engages and therefore what df a
#: live bundle can serve. 0.5.0 shipped the gate firing at 1.645*SE but a pair
#: becoming a floor demand only at 2.0 sigma, so a reading in [1.645, 2.0)
#: sigma held the gate lit -- recording mode, no gate-days -- while the floor
#: never attempted a remedy: a gate that can fire on a pair the remedy is not
#: allowed to see. One shared constant now. Free again: the first live firing
#: of 0.5.0 came back covered=False (the live ML fit is 3.5, not the fixture's
#: 6.5, and the live rho is 0.863), so every 0.5 row is defect-stamped and no
#: gate-day existed to lose.
#:
#: 0.7.0 — the A11 remedy chain: a conditional rho-lift, then the df floor
#: (OPEN-QUESTIONS A11, option R+H, adopted by the product owner 2026-08-05
#: evening). MINOR twice over: it changes the correlation entries the bundle
#: SERVES on demand pairs, and it changes the df a live bundle lands on
#: (df_ML when the lift covers, where 0.6.0 always floored to the grid wall).
#:
#: Why the lever changed, all measured on the live readings before adoption:
#: the df floor hit the family's grid wall short of the asymmetric pair's
#: bound (model@q ~0.679 at df 2.5 vs a 0.6913 target -- covered=False,
#: recording mode permanent while the reading held), and a per-output
#: measurement showed df 5.0 -> 2.5 moves P(liq) by <= 0.1 pp at the live
#: correlations: weak in output space, not only in reach. The rho-lever
#: covers at the ML df with room to spare (rho* = 0.907 at df 5.0, +0.044
#: over the EWMA 0.863; body cost +0.027 nats/obs) and is the lever that
#: actually moves outputs (same-sign books ~+0.5 pp, hedged books -2.4 to
#: -4.7 pp -- signs disclosed per-output in A11, every move TOWARD the
#: measured dependence). Same demand set as the floor, one-sided (lifts
#: only), PD-projected with coverage RE-VERIFIED on the projected entries,
#: floor composing on anything the 0.98 cap cannot reach.
#:
#: Deliberately timed, again while the counter is held: every 0.6 row is
#: defect-stamped (covered=False on both live firings), so no gate-day
#: existed to lose. The HL_UNIVERSE default widens to include HYPE in the
#: same reset (B6 decision, recorded there): the census named HYPE the
#: largest single cohort recovery and a live probe showed its pairs quiet at
#: the §2.3 gate, so the two changes share one §3.3 reset instead of costing
#: two.
MODEL_VERSION = "0.7.0-phase1"

# Distribution-affecting prefix; the shadow counter keys on this, not on the
# full string, so that PATCH releases keep accumulating validation days.
DISTRIBUTION_VERSION = ".".join(MODEL_VERSION.split("-")[0].split(".")[:2])
