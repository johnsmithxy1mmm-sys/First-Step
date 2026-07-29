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
MODEL_VERSION = "0.2.1-phase1"

# Distribution-affecting prefix; the shadow counter keys on this, not on the
# full string, so that PATCH releases keep accumulating validation days.
DISTRIBUTION_VERSION = ".".join(MODEL_VERSION.split("-")[0].split(".")[:2])
