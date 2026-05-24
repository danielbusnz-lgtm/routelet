# Intent taxonomy

The fixed set of intents routelet classifies into. Frozen: do not add, rename, or merge
without re-labeling the dataset and re-running every eval. Maps 1:1 to Aegis's 5 routing paths.

## The 5 intents

| Intent | One-line definition | Examples |
|--------|---------------------|----------|
| `find_action` | Locate or operate a UI element on the current screen. | "where is the search bar", "click the play button" |
| `integration` | One discrete action against an app or service. | "play despacito on spotify", "check my email" |
| `chat` | Answer from general knowledge or conversation. No app action, no personal data. | "what's your name", "explain how transformers work" |
| `memory` | Store or recall a personal fact about the user. | "remember my name is Daniel", "what's my wifi password" |
| `agent` | A task needing two or more chained steps or a plan. | "open youtube, search for lofi, play the top result", "find the cheapest flight to tokyo and book it" |

## Boundary rules

Use these when a command looks like it could fit two intents.

1. **Steps decide `integration` vs `agent`.** One action is `integration`. Two or more
   chained actions, or anything that needs planning, is `agent`.
   "check my email" is `integration`; "read my latest email and reply that I'll be late" is `agent`.

2. **Source decides `memory` vs `chat`.** A question answered from a stored personal fact is
   `memory`. A question answered from world knowledge is `chat`.
   "what's my wifi password" is `memory`; "what's the capital of france" is `chat`.

3. **Storing a fact is always `memory`.** "remember X" is `memory` even when phrased as a
   command, not `integration`.

4. **Screen vs service decides `find_action` vs `integration`.** Naming a visible UI element
   ("the play button", "the search bar") is `find_action`. Naming an app or service capability
   ("on spotify", "my email") is `integration`.

## Tie-breaker

If a command still fits more than one after the rules above, pick the first match in this order:

`agent` > `memory` > `integration` > `find_action` > `chat`

`chat` is the default. If nothing else clearly applies, it is `chat`.
