-- Calibration journal (§3.4). Postgres dialect.
--
-- This is the product's only durable moat. A public, continuously updated
-- score of how well our own forecasts were calibrated is something no venue
-- or competitor publishes, and it converts the single worst reputational
-- exposure -- "you said 8% and I got liquidated" -- into evidence of
-- honesty, provided the number is recorded before the outcome is known and
-- never edited afterwards.
--
-- `risk_engine/shadow/journal.py` runs the same schema on SQLite for local
-- work and tests; `test_journal.py` asserts the two column sets agree.

CREATE TABLE IF NOT EXISTS calibration_predictions (
    id                   BIGSERIAL PRIMARY KEY,
    address              TEXT        NOT NULL,
    variant              TEXT        NOT NULL,  -- model | baseline_a | baseline_b
    predicted_at         TIMESTAMPTZ NOT NULL,
    horizon_hours        INTEGER     NOT NULL,
    resolves_at          TIMESTAMPTZ NOT NULL,
    model_version        TEXT        NOT NULL,
    -- §3.3 anti-overfitting: the shadow-day counter keys on this, so a
    -- distribution-affecting change resets the accumulated window.
    distribution_version TEXT        NOT NULL,
    seed                 BIGINT      NOT NULL,
    n_paths              INTEGER     NOT NULL,
    converged            BOOLEAN     NOT NULL,
    start_equity         DOUBLE PRECISION NOT NULL,
    p_liq                DOUBLE PRECISION NOT NULL,
    p_liq_ci_low         DOUBLE PRECISION NOT NULL,
    p_liq_ci_high        DOUBLE PRECISION NOT NULL,
    var_95               DOUBLE PRECISION NOT NULL,
    cvar_95              DOUBLE PRECISION NOT NULL,
    -- Predicted 24h equity-change distribution as a quantile function.
    -- Quantiles, not samples: 1001 numbers per row is a table that survives
    -- years, 20 000 is not, and every metric §3.3 asks for is computable
    -- from the quantile function.
    quantile_values      JSONB       NOT NULL,
    n_quantile_levels    INTEGER     NOT NULL,
    book_snapshot        JSONB       NOT NULL,
    UNIQUE (address, variant, predicted_at, distribution_version)
);

CREATE INDEX IF NOT EXISTS calibration_predictions_due_idx
    ON calibration_predictions (resolves_at);
CREATE INDEX IF NOT EXISTS calibration_predictions_version_idx
    ON calibration_predictions (distribution_version, variant);

CREATE TABLE IF NOT EXISTS calibration_outcomes (
    prediction_id        BIGINT PRIMARY KEY REFERENCES calibration_predictions(id),
    resolved_at          TIMESTAMPTZ NOT NULL,
    actual_equity        DOUBLE PRECISION NOT NULL,
    actual_equity_change DOUBLE PRECISION NOT NULL,
    -- §3.3's PIT test compares predicted against realised equity change, but
    -- equity also moves for reasons the model never claimed to predict.
    -- These two columns are what make the cohort split in
    -- shadow/metrics.py possible (OPEN-QUESTIONS B2). Scoring without them
    -- measures how actively the sampled traders traded, not calibration.
    external_flow_usd    DOUBLE PRECISION NOT NULL,
    book_changed         BOOLEAN     NOT NULL,
    liquidated           BOOLEAN     NOT NULL,
    -- RANDOMIZED probability integral transform. The predicted distribution
    -- has an atom at total loss, because §1.6 writes a liquidated pool to
    -- exactly zero equity. A plain CDF value maps every realised liquidation
    -- to one identical number, and the KS test then rejects even a perfectly
    -- calibrated model (audit A-02: p = 1e-104 on a by-construction-true
    -- forecast). pit is drawn uniformly inside [F(x-), F(x)]; pit_u is the
    -- uniform that was used, derived deterministically from the prediction
    -- id so the row stays reproducible.
    pit                  DOUBLE PRECISION NOT NULL,
    pit_u                DOUBLE PRECISION NOT NULL,
    crps                 DOUBLE PRECISION NOT NULL,
    var_95_breached      BOOLEAN     NOT NULL,
    -- The calendar day is the independent unit for clustered inference:
    -- addresses observed on the same day share one market (OPEN-QUESTIONS B1).
    observation_day      DATE        NOT NULL,
    -- How late the resolver ran against the prediction's own horizon. A 24h
    -- forecast scored against a 72h realisation is not a model error, it is
    -- an infrastructure gap, and scoring it silently corrupts the calibration
    -- record (audit A-04). Stale rows are excluded from every cohort by
    -- default. They are also counted -- but into the metrics registry of the
    -- short-lived `shadow resolve` process, which exits immediately after,
    -- so that counter reaches no scraper. What an operator actually sees is
    -- the resolver's own stdout ("N flagged stale") in the container log.
    -- Read that, not `/metrics`, when the window stops advancing.
    resolution_lag_s     DOUBLE PRECISION NOT NULL,
    stale_resolution     BOOLEAN     NOT NULL
);

CREATE INDEX IF NOT EXISTS calibration_outcomes_day_idx
    ON calibration_outcomes (observation_day);
