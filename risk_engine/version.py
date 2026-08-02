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
MODEL_VERSION = "0.4.0-phase1"

# Distribution-affecting prefix; the shadow counter keys on this, not on the
# full string, so that PATCH releases keep accumulating validation days.
DISTRIBUTION_VERSION = ".".join(MODEL_VERSION.split("-")[0].split(".")[:2])
