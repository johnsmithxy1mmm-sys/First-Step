# Executable specification of the money-recording contract. Each scenario here
# corresponds to an audit finding that a green 458-test suite did not catch,
# because none of those tests drove a RESTING order, a KILLED fill, or a
# POISONED number through a money path.

Feature: The ledger only ever records what actually happened
  As the operator
  I want recorded trades to match reality on the exchange
  So that PnL, exposure and every risk limit are computed from facts

  Scenario: An exit that does not fill leaves the position open
    Given a held position of 100 shares
    When a take-profit sell is placed and never fills
    Then the position is still open
    And no sale is recorded

  Scenario: A killed fill-or-kill order records nothing
    Given a fill-or-kill buy is accepted and then killed unfilled
    When the strategy records its result
    Then no trade is recorded

  Scenario: A trade row that cannot describe a real fill is refused
    When a trade is recorded with a non-finite size
    Then the write is rejected
    And exposure remains a finite number

  Scenario: Selling more than was ever bought is surfaced, not hidden
    Given a sale of 10 shares with no prior purchase
    When accounting drift is checked
    Then the drift is reported for that token
