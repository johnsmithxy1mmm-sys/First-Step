"""Audit reproducers: each test here DEMONSTRATES a registered finding failing
on current HEAD (docs/audit/REGISTER.md). They are gated behind AUDIT_REPRO=1
so the normal suite stays green while the findings remain un-fixed — a red
repro is the finding's proof, not a CI accident.

Run them with:

    AUDIT_REPRO=1 python -m pytest polymarket_bot/tests/audit -q

Every test carries its finding ID. When a finding is fixed, its repro is
promoted into the main suite as the permanent regression test.
"""

import os

import pytest

if not os.environ.get("AUDIT_REPRO"):
    pytest.skip("audit reproducers run only with AUDIT_REPRO=1",
                allow_module_level=True)


def make_market(**kw):
    """Thin re-export of the main suite's factory (imported lazily to avoid a
    conftest name clash between the two test packages)."""
    from polymarket_bot.tests.conftest import make_market as _mk
    return _mk(**kw)


@pytest.fixture()
def cfg():
    from polymarket_bot.config import BotConfig
    cfg = BotConfig()
    # Reproducers must not spend the production 90s fill timeout per call.
    cfg.executor.fill_timeout_sec = 0.05
    cfg.executor.poll_interval_sec = 0.01
    return cfg


@pytest.fixture()
def ledger(tmp_path):
    from polymarket_bot.ledger import Ledger
    led = Ledger(tmp_path / "audit.sqlite")
    yield led
    led.close()
