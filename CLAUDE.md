# CLAUDE.md

Project context, rules and history live in `src/stock_screener/HANDOFF.md`. Read §4
(never do these) and §9 (conventions) before changing code.

The trading method (Minervini's SEPA, from his two books) and where the cockpit follows or
departs from it is in `src/stock_screener/SEPA_METHODOLOGY.md`. Read it before judging a
chart, running the weekend hunt, answering a question about the method, or changing a
trading rule. Its §0 says how a session uses it.

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY are used as described in
RFC 2119.

## Comments

- Comments MUST appear only where the code needs clarification.
- Comments MUST NOT narrate. They don't restate the code, walk through its steps, or
  tell its history: no bug, review, date or "previously". History goes in test
  docstrings and the ledger in `src/stock_screener/HANDOFF_HISTORY.md` (§11).
- Sentences SHOULD be short.
- A comment that states an obligation MUST use an RFC 2119 keyword.
- Docstrings follow the same rules. A docstring SHOULD state the contract: inputs,
  return shape, and when it returns None or raises.

## Commits

- The subject MUST be in the imperative mood: "Fix", "Add", "Raise", not "Fixed" or
  "Adds".
- The body MUST be omitted unless it records a fact the diff cannot show, such as why a
  choice was made, a measured result, a rejected alternative, or a manual deploy step.
