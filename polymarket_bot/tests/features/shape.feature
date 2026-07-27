# Executable specification of the payoff-SHAPE contract.
#
# Every filter the bot had looked at price band, horizon or edge. None looked at
# the shape of the payoff, and none of them was wrong — the book they jointly
# admitted was: 20 legs bought at 0.944-0.990, $1,891 committed, worst case $846,
# maximum possible upside $37.27, expected loss at the same prices $36.34. EV zero
# by construction, and no rule anywhere that could refuse it or unwind it.
#
# These scenarios assert the shape rules by their EFFECT on money, not by the
# presence of a config key.

Feature: A position's payoff shape must be survivable
  As the operator
  I want the bot to refuse bets it cannot win back and to unwind them when held
  So that one adverse resolution cannot exceed the whole book's upside

  Scenario: A tail that needs ninety-nine wins to repay one loss is refused
    Given a YES tail priced at 1 cent
    When the fade evaluates it
    Then the tail is refused for its payoff shape
    And no position is opened

  Scenario: A tail with a survivable shape is still traded
    Given a YES tail priced at 5 cents resolving in 10 days
    When the fade evaluates it
    Then the tail is accepted

  Scenario: Capital is refused when it would earn too little per day
    Given a YES tail priced at 5 cents resolving in 80 days
    When the fade evaluates it
    Then the tail is refused for its return per day

  Scenario: The generic take-profit cannot rescue a fade position
    Given a fade leg bought at 0.976
    When the price rises as far as the venue allows
    Then the generic take-profit still does not fire

  Scenario: A materialising tail is cut instead of ridden to resolution
    Given a fade leg bought at 0.976
    When the implied tail probability triples
    Then the leg is sold
    And the realised loss is a fraction of the notional

  Scenario: A position with no upside left releases its capital
    Given a fade leg bought at 0.976
    When the price reaches 0.995
    Then the leg is sold

  Scenario: The market maker keeps room the directional book cannot take
    Given the fade already holds its full reserve-adjusted budget
    When a further directional trade is sized
    Then no capital is allocated
    And account-wide room is still available
