"""LLM rules lawyer: structured verdict, discrepancy alerting, dedup."""

from types import SimpleNamespace
from unittest import mock

from polymarket_bot.ruleslawyer import RulesLawyer, RulesVerdict

from .conftest import make_market


def fake_client(verdict: RulesVerdict):
    client = mock.Mock()
    client.messages.parse.return_value = SimpleNamespace(parsed_output=verdict)
    return client


def make_lawyer(cfg, ledger, verdict):
    cfg.ruleslawyer.enabled = True
    return RulesLawyer(cfg, ledger, client=fake_client(verdict))


def test_flags_discrepancy_and_alerts(cfg, ledger):
    v = RulesVerdict(discrepancy=True, favored_side="No",
                     rationale="Rules require an official source that will not report in time.")
    law = make_lawyer(cfg, ledger, v)
    m = make_market(volume_24h_usd=50_000,
                    description="Resolves per official government data published by Dec 31.")
    with mock.patch("polymarket_bot.ruleslawyer.alert") as a:
        flagged = law.cycle([m])
    assert len(flagged) == 1 and a.called
    assert "RULES DISCREPANCY" in a.call_args[0][0]


def test_no_alert_when_no_discrepancy(cfg, ledger):
    v = RulesVerdict(discrepancy=False, favored_side="none", rationale="Clear.")
    law = make_lawyer(cfg, ledger, v)
    m = make_market(volume_24h_usd=50_000, description="Resolves YES if it happens.")
    with mock.patch("polymarket_bot.ruleslawyer.alert") as a:
        assert law.cycle([m]) == []
        a.assert_not_called()


def test_dedup_and_volume_and_call_cap(cfg, ledger):
    v = RulesVerdict(discrepancy=False, favored_side="none", rationale="ok")
    law = make_lawyer(cfg, ledger, v)
    thin = make_market(id="thin", volume_24h_usd=1000, description="rules")
    with mock.patch("polymarket_bot.ruleslawyer.alert"):
        assert law.cycle([thin]) == []          # below volume floor -> not analyzed
        m = make_market(id="m", volume_24h_usd=50_000, description="rules")
        law.cycle([m])
        # Second pass: already seen -> no second LLM call.
        law._client.messages.parse.reset_mock()
        law.cycle([m])
        law._client.messages.parse.assert_not_called()


def test_disabled_without_api(cfg, ledger):
    law = RulesLawyer(cfg, ledger)                # enabled False, no client
    assert law.available is False
    assert law.cycle([make_market()]) == []
