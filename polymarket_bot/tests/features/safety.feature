# Executable specification of the safety layer, in the language a reviewer or a
# buyer can check without reading Python. Every scenario below is bound to real
# code in test_safety_bdd.py — these are not documentation, they run in CI.
#
# The scenarios deliberately describe EFFECTS ("no new orders are allowed"),
# never internal state ("the paused flag is true"): mutation testing showed a
# mutant that dropped the pause term from `trading_allowed` survived the entire
# suite precisely because the old drills asserted the flag, not the effect.

Feature: The kill-switch protects capital under failure
  As the operator of an autonomous trading bot
  I want trading to stop the moment the system cannot trust its own inputs
  So that a data fault or a loss streak cannot become an unbounded loss

  Background:
    Given a bot with a $300 global exposure cap and a $25 daily loss limit

  Scenario: A dead market-data stream stops new risk
    When the websocket stream has been silent past the staleness threshold
    Then no new orders are allowed
    And all resting quotes have been cancelled

  Scenario: Recovering the stream resumes trading
    Given the websocket stream has been silent past the staleness threshold
    When the stream recovers
    Then new orders are allowed again

  Scenario: A data anomaly pause is independent of the stream pause
    Given the websocket stream has been silent past the staleness threshold
    And a corrupt price feed was detected
    When the stream recovers
    Then no new orders are allowed

  Scenario: Breaching the daily loss limit halts trading terminally
    When equity has fallen by 25 dollars since the start of the day
    Then no new orders are allowed
    And the halt is not cleared by a pause resume

  Scenario: Unmeasurable accounting halts rather than being ignored
    When the ledger reports a non-finite equity
    Then no new orders are allowed

  Scenario: Exposure at the configured ceiling blocks new entries
    When open exposure reaches the global cap
    Then new entries are refused
    But reducing an existing position is still allowed
