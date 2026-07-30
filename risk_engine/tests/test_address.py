"""Address normalisation and validation — the identity of an account.

Two properties are under test, and they are the two failures the engine had
no defence against:

  - the venue answers an address it does not recognise with a well-formed
    *empty* state (§5.1), never with an error, so a malformed address reads
    downstream as "this account holds no positions". Nothing in the response
    can tell us the address was wrong, which is why the check has to sit on
    the outbound side, before the request;
  - the calibration journal's rows are written once and never updated (§3.4,
    audit A-09) and its `address` column is compared byte-for-byte by both
    backends, so an account written under two spellings becomes two permanent
    identities: two counts towards §3.3's 200-address gate, the same realised
    outcome twice in a cohort, and a broken `(address, observation_day)` join
    in champion/challenger.

Nothing here touches the network. The `InfoClient` tests either replace `post`
with a recorder or replace `urlopen`, so what is asserted is the payload -- and
in one case the encoded body -- the client *would* have sent.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from risk_engine.domain.types import ADDRESS_HEX_DIGITS, normalise_address
from risk_engine.market.info import InfoClient
from risk_engine.shadow.journal import VARIANT_MODEL, CalibrationJournal
from risk_engine.shadow.providers import StaticAddressSource
from risk_engine.sim.stats import PredictiveDistribution

#: One real account, in the three spellings it turns up in. The checksummed
#: form is the EIP-55 one for this address -- it is what a block explorer
#: displays, and therefore what an operator pastes.
LOWER = "0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed"
CHECKSUMMED = "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"
UPPER_PREFIX = "0X5AAEB6053F3E94C9B9A09F33669435E7EF1BEAED"

NOW = datetime(2026, 7, 29, 12, tzinfo=timezone.utc)


class TestCanonicalForm:
    def test_a_lowercase_address_is_returned_unchanged(self):
        assert normalise_address(LOWER) == LOWER

    def test_an_eip55_checksummed_address_is_the_same_account(self):
        """§5.1's identity is the 20 bytes; the mixed case is a checksum
        written over them, not part of the account. An operator who pastes
        what the explorer showed them must not get a different account from
        one who typed it in lowercase."""
        assert normalise_address(CHECKSUMMED) == LOWER

    def test_case_is_folded_from_any_spelling_including_the_prefix(self):
        assert normalise_address(UPPER_PREFIX) == LOWER
        assert normalise_address(LOWER.upper()) == LOWER

    def test_a_broken_checksum_is_accepted_which_is_a_known_gap(self):
        """Records a gap, not a virtue. This assertion used to be justified as
        avoiding "either rejecting the perfectly legal all-lowercase spelling
        or refusing to measure a real account over a mis-transcribed capital",
        which is a false dilemma: EIP-55 is a *case* pattern, so an
        all-lowercase or all-uppercase string carries no checksum and is
        accepted unverified by every implementation. Only a mixed-case string
        is checkable, and `broken_checksum` below is mixed-case -- so
        verification would have rejected no legal spelling and would have
        caught this exact typo on the paste path the docstring cites.

        The decision to skip it stands on its real cost: EIP-55 is defined
        over keccak-256, `hashlib` has none, `hashlib.sha3_256` is NIST SHA-3
        and not keccak (different padding byte, unrelated digest), so the only
        routes are a third-party dependency in a numpy+scipy package or a
        hand-rolled Keccak permutation inside the module that defines account
        identity. What this test pins is the *consequence* of that decision, so
        that adding keccak later is a visible change to a recorded gap rather
        than a silent tightening -- and so nobody re-derives the false dilemma
        from the passing assertion.
        """
        # Mixed case, one capital wrong against the EIP-55 vector in
        # CHECKSUMMED above. A verifier would reject it; this one does not.
        broken_checksum = "0x5AaEb6053f3e94c9b9a09f33669435e7ef1beaed"
        assert any(c.isupper() for c in broken_checksum[2:])
        assert any(c.islower() for c in broken_checksum[2:])
        assert broken_checksum != CHECKSUMMED
        assert normalise_address(broken_checksum) == LOWER

    def test_a_swapped_hex_digit_is_caught_by_nothing_at_all(self):
        """The consequence worth stating plainly, and the one EIP-55 would not
        have fixed either.

        `...beaed` mistyped as `...beaec` is 40 valid hex digits, so every
        check in `normalise_address` passes, and it is a *different real
        account*. The venue answers an address it does not recognise with the
        well-formed empty state of §5.1 -- so this typo does not fail, it
        returns "no positions", which a risk tool renders as no risk. Nothing
        in this codebase can catch it: a lowercase string carries no checksum
        information, so even a keccak verifier would pass it through.
        """
        swapped = LOWER[:-1] + "c"
        assert swapped != LOWER
        assert normalise_address(swapped) == swapped
        # Same length, same shape, both canonical -- indistinguishable to
        # anything downstream, including the journal's identity column.
        assert len(swapped) == len(LOWER)
        assert normalise_address(swapped) != normalise_address(LOWER)

    def test_normalising_is_idempotent(self):
        """The check sits at more than one layer (client wrapper, `post`,
        journal write), so applying it twice must be indistinguishable from
        applying it once."""
        assert normalise_address(normalise_address(CHECKSUMMED)) == LOWER


class TestRejections:
    """Every rejection has to name the defect. "Invalid address" is not enough
    to fix a 42-character string by eye, and the person holding a wrong
    address is exactly who §5.1's silent empty state is waiting for.
    """

    def test_none_is_refused(self):
        with pytest.raises(ValueError, match="required"):
            normalise_address(None)

    def test_an_empty_string_is_refused(self):
        with pytest.raises(ValueError, match="empty"):
            normalise_address("")

    def test_a_non_string_is_refused_by_type(self):
        with pytest.raises(ValueError, match="must be a string"):
            normalise_address(12345)

    @pytest.mark.parametrize(
        "value",
        [f" {LOWER}", f"{LOWER} ", f"{LOWER}\n", f"{LOWER}\t", "   ", f"0x5aae\t{LOWER[6:]}"],
    )
    def test_whitespace_is_refused_rather_than_trimmed(self, value):
        """Refused, not stripped: an address list is hand-edited, so a stray
        newline is a mistake worth showing the operator rather than absorbing.
        A whitespace-only string is caught by the same rule."""
        with pytest.raises(ValueError, match="whitespace"):
            normalise_address(value)

    def test_a_missing_0x_prefix_is_refused(self):
        with pytest.raises(ValueError, match="must start with '0x'"):
            normalise_address(LOWER[2:])

    def test_a_short_address_is_refused_and_the_count_is_reported(self):
        with pytest.raises(ValueError, match="exactly 40 hex digits") as exc:
            normalise_address(LOWER[:-1])
        assert "got 39" in str(exc.value)

    def test_a_long_address_is_refused(self):
        with pytest.raises(ValueError, match="exactly 40 hex digits") as exc:
            normalise_address(LOWER + "a")
        assert "got 41" in str(exc.value)

    def test_a_readable_stub_is_refused(self):
        """The spelling this repo's own fixtures used everywhere. It is only
        two characters long after the prefix, and the venue would answer it
        with an empty state rather than an error."""
        with pytest.raises(ValueError, match="exactly 40 hex digits"):
            normalise_address("0xabc")

    def test_a_non_hex_character_is_refused_and_named(self):
        with pytest.raises(ValueError, match="hexadecimal") as exc:
            normalise_address("0x" + "g" * ADDRESS_HEX_DIGITS)
        assert "'g'" in str(exc.value)

    def test_a_transposed_o_for_zero_is_named_specifically(self):
        """The realistic typo: right length, one wrong glyph. The message has
        to point at the character, because 40 hex digits cannot be diffed by
        eye."""
        with pytest.raises(ValueError, match="hexadecimal") as exc:
            normalise_address(LOWER.replace("0x5", "0xo", 1))
        assert "'o'" in str(exc.value)


class TestInfoClientNormalisesBeforeTheRequest:
    """The API side of the choke point. `post` is replaced with a recorder
    (and, in the last test, `urlopen` is), so these assert on what the client
    would have put on the wire without any of them making a request.
    """

    @staticmethod
    def _recording_client() -> tuple[InfoClient, list[dict]]:
        client = InfoClient()
        sent: list[dict] = []

        def record(payload, weight=20):
            sent.append(payload)
            return {}

        client.post = record  # type: ignore[method-assign]
        return client, sent

    def test_clearinghouse_state_sends_the_canonical_address(self):
        client, sent = self._recording_client()
        client.clearinghouse_state(CHECKSUMMED)
        assert sent == [{"type": "clearinghouseState", "user": LOWER}]

    def test_user_funding_sends_the_canonical_address(self):
        client, sent = self._recording_client()
        client.user_funding(CHECKSUMMED, start_ms=1, end_ms=2)
        assert sent[0]["user"] == LOWER
        assert sent[0]["type"] == "userFunding"

    def test_open_orders_sends_the_canonical_address(self):
        """No callers today. Checked anyway, so wiring it up does not
        introduce a fresh unchecked entry point."""
        client, sent = self._recording_client()
        client.open_orders(UPPER_PREFIX)
        assert sent == [{"type": "openOrders", "user": LOWER}]

    def test_every_spelling_of_one_account_produces_one_payload(self):
        client, sent = self._recording_client()
        for spelling in (LOWER, CHECKSUMMED, UPPER_PREFIX):
            client.clearinghouse_state(spelling)
        assert {frozenset(p.items()) for p in sent} == {
            frozenset({"type": "clearinghouseState", "user": LOWER}.items())
        }

    def test_a_malformed_address_is_refused_before_any_weight_is_charged(self):
        """§5.3's budget is charged per request and the shadow sweep truncates
        when it runs out, so a request that was never going to be made must
        not cost any of it."""
        client = InfoClient()
        with pytest.raises(ValueError, match="40 hex digits"):
            client.clearinghouse_state("0xabc")
        assert client.budget.spent() == 0

    def test_the_agent_address_refusal_still_comes_first(self):
        """A well-formed agent address passes every format check there is.
        Diagnosing it as a format problem would send the caller looking at the
        wrong thing -- §5.1's failure is the account, not the spelling."""
        client = InfoClient()
        with pytest.raises(ValueError, match="real account address"):
            client.clearinghouse_state(CHECKSUMMED, is_agent_address=True)
        assert client.budget.spent() == 0

    def test_the_bytes_on_the_wire_carry_the_canonical_address(self, monkeypatch):
        """The strongest form of the claim: assert on the encoded request body
        rather than on the dict, with `urlopen` replaced so nothing leaves the
        process. This is the backstop for a wrapper that does not exist yet --
        `user` is the only address-carrying field in the Info request shapes,
        so the one check in `post` covers any future method that forgets."""
        captured: list[bytes] = []

        def no_network(req, timeout=None):
            captured.append(req.data)
            raise urllib.error.URLError("no network in tests")

        monkeypatch.setattr(urllib.request, "urlopen", no_network)
        client = InfoClient(max_retries=1)  # no retry, so no backoff sleep
        with pytest.raises(RuntimeError, match="info request"):
            client.post({"type": "clearinghouseState", "user": CHECKSUMMED})
        assert json.loads(captured[0])["user"] == LOWER

    def test_post_refuses_a_malformed_user_field_before_charging(self):
        client = InfoClient()
        with pytest.raises(ValueError, match="40 hex digits"):
            client.post({"type": "clearinghouseState", "user": "0xnope"})
        assert client.budget.spent() == 0


class TestJournalIdentity:
    """The storage side of the choke point, on an in-memory SQLite journal.

    `address` is TEXT with no COLLATE in schema.sql, and neither backend's
    default collation folds case, so the UNIQUE constraint cannot see two
    spellings of one account as a duplicate. The canonicalisation at the write
    is what makes them one account.
    """

    @staticmethod
    def _write(journal: CalibrationJournal, address: str, seed: int = 1) -> int:
        dist = PredictiveDistribution.from_samples(np.linspace(-100.0, 100.0, 1_001))
        return journal.record_prediction(
            address=address, variant=VARIANT_MODEL, predicted_at=NOW, horizon_hours=24,
            model_version="0.2.1", distribution_version="0.2", seed=seed, n_paths=100,
            converged=True, start_equity=1_000.0, p_liq=0.1, p_liq_ci=(0.05, 0.15),
            var_95=50.0, cvar_95=70.0, distribution=dist, book_snapshot={},
        )

    @staticmethod
    def _distinct_addresses(journal: CalibrationJournal) -> list[str]:
        rows = journal._query(
            "SELECT DISTINCT address FROM calibration_predictions ORDER BY address"
        )
        return [r["address"] for r in rows]

    def test_two_spellings_of_one_account_are_one_address(self):
        """The headline property. Written at different instants so the UNIQUE
        constraint is not what is being measured -- only the identity is."""
        with CalibrationJournal() as journal:
            self._write(journal, LOWER, seed=1)
            journal.record_prediction(
                address=CHECKSUMMED, variant=VARIANT_MODEL,
                predicted_at=NOW.replace(hour=13), horizon_hours=24,
                model_version="0.2.1", distribution_version="0.2", seed=2, n_paths=100,
                converged=True, start_equity=1_000.0, p_liq=0.1, p_liq_ci=(0.05, 0.15),
                var_95=50.0, cvar_95=70.0,
                distribution=PredictiveDistribution.from_samples(
                    np.linspace(-100.0, 100.0, 1_001)
                ),
                book_snapshot={},
            )
            assert self._distinct_addresses(journal) == [LOWER]
            count = journal._query(
                "SELECT COUNT(DISTINCT address) AS n FROM calibration_predictions"
            )[0]["n"]
            assert count == 1

    def test_the_stored_form_is_the_canonical_one(self):
        with CalibrationJournal() as journal:
            self._write(journal, CHECKSUMMED)
            assert self._distinct_addresses(journal) == [LOWER]
            # And on the way back out: the resolver replays this string to the
            # Info API a day later, so a non-canonical row would be re-sent.
            assert journal.due(NOW + timedelta(hours=25))[0].address == LOWER

    def test_the_second_spelling_hits_the_unique_constraint(self):
        """The constraint is on (address, variant, predicted_at,
        distribution_version) and compares bytes. Before canonicalisation both
        rows inserted cleanly and the journal held two predictions claiming to
        be about two different accounts."""
        import sqlite3

        with CalibrationJournal() as journal:
            self._write(journal, LOWER)
            with pytest.raises(sqlite3.IntegrityError):
                self._write(journal, CHECKSUMMED)

    def test_a_malformed_address_never_reaches_the_permanent_record(self):
        """Predictions are never updated (audit A-09), so anything written
        here cannot be corrected in place. The write is the last moment the
        address can be refused."""
        with CalibrationJournal() as journal:
            with pytest.raises(ValueError, match="40 hex digits"):
                self._write(journal, "0xaaa")
            assert self._distinct_addresses(journal) == []

    def test_two_spellings_count_once_towards_the_phase_4_gate(self):
        """§3.3's gate needs 200 distinct accounts, counted with
        `COUNT(DISTINCT address)`. Case-sensitive, that gate was satisfiable
        by one account spelled 200 ways."""
        with CalibrationJournal() as journal:
            for i, spelling in enumerate((LOWER, CHECKSUMMED, UPPER_PREFIX)):
                pid = journal.record_prediction(
                    address=spelling, variant=VARIANT_MODEL,
                    predicted_at=NOW + timedelta(hours=i), horizon_hours=24,
                    model_version="0.2.1", distribution_version="0.2", seed=i,
                    n_paths=100, converged=True, start_equity=1_000.0, p_liq=0.1,
                    p_liq_ci=(0.05, 0.15), var_95=50.0, cvar_95=70.0,
                    distribution=PredictiveDistribution.from_samples(
                        np.linspace(-100.0, 100.0, 1_001)
                    ),
                    book_snapshot={},
                )
                journal.record_outcome(
                    prediction_id=pid, resolved_at=NOW + timedelta(days=1, hours=i),
                    actual_equity=1_000.0, actual_equity_change=0.0,
                    external_flow_usd=0.0, book_changed=False, liquidated=False,
                    pit=0.5, pit_u=0.5, crps=1.0, var_95_breached=False,
                    observation_day=date(2026, 7, 29) + timedelta(days=i),
                    resolution_lag_s=0.0, stale_resolution=False,
                )
            progress = journal.progress("0.2")
            assert progress.resolved_observations == 3
            assert progress.distinct_days == 3
            assert progress.distinct_addresses == 1


class TestAddressSourcesAgree:
    """The two `AddressSource` implementations used to disagree about case,
    and both feed the same journal. Whichever one an operator reached for
    decided whether the account got a second identity.
    """

    def test_a_static_source_canonicalises_and_dedups(self):
        source = StaticAddressSource((CHECKSUMMED, LOWER, UPPER_PREFIX), "one account, three ways")
        assert source.addresses() == [LOWER]

    def test_a_static_source_refuses_a_malformed_entry(self):
        source = StaticAddressSource(("0xnot-an-address",), "typo")
        with pytest.raises(ValueError, match="40 hex digits"):
            source.addresses()
