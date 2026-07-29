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

MODEL_VERSION = "0.1.0-phase1"

# Distribution-affecting prefix; the shadow counter keys on this, not on the
# full string, so that PATCH releases keep accumulating validation days.
DISTRIBUTION_VERSION = ".".join(MODEL_VERSION.split("-")[0].split(".")[:2])
