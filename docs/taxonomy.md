# Intent taxonomy

The fixed set of intents routelet classifies into. Frozen: do not add, rename, or merge
without re-labeling the dataset and re-running every eval. Maps 1:1 to Peeky's classified
routing paths.

`agent` was removed from the set: Peeky enters agent mode only on an explicit spoken cue
("peeky agent ..."), recognized deterministically before classification, so the router
never decides it. Where a cue-less multi-step utterance lands is an open taxonomy
question; for now no class claims it and the old agent rows are gone from the dataset.

## The 4 intents

| Intent | One-line definition | Examples |
|--------|---------------------|----------|
| `find_action` | Locate or operate a UI element on the current screen. | "where is the search bar", "click the play button" |
| `integration` | One discrete action against an app or service. | "play despacito on spotify", "check my email" |
| `chat` | Answer from general knowledge or conversation. No app action, no personal data. | "what's your name", "explain how transformers work" |
| `memory` | Store or recall a personal fact about the user. | "remember my name is Daniel", "what's my wifi password" |

## Boundary rules

Use these when a command looks like it could fit two intents.

1. **Source decides `memory` vs `chat`.** A question answered from a stored personal fact is
   `memory`. A question answered from world knowledge is `chat`.
   "what's my wifi password" is `memory`; "what's the capital of france" is `chat`.

2. **Storing a fact is `memory`, but only with an explicit storage verb.** "remember/note/save X"
   is `memory` even when it looks like another intent: "remember that i like to play despacito on
   spotify" is `memory`, not `integration`. A bare preference with no storage verb is not `memory`:
   "i like to check my email" is `chat`.

3. **Another person's contact info is `integration`, not `memory`.** `memory` recalls facts about
   the user themselves ("what's my wifi password", "where did i park"). Asking for someone else's
   number or email is a Contacts lookup, so `integration`: "what's my sister's number",
   "what's my dentist's email". The possessive "my" before a person does not make it `memory`.

4. **Screen vs service decides `find_action` vs `integration`.** Naming a visible UI element
   ("the play button", "the search bar") is `find_action`. Naming an app or service capability
   ("on spotify", "my email") is `integration`. When both appear, an explicit UI verb (click, tap,
   scroll to) on a named element wins for `find_action`: "click on the song Sicko Mode in spotify"
   is `find_action`, not `integration`, even though Spotify is named. Playback verbs (skip, pause,
   next, volume) are service capabilities, so `integration`, unless a button is named ("click the
   skip button" is `find_action`).

5. **A question about a UI element is `chat`, not `find_action`.** `find_action` needs a command
   to locate or operate something ("where is X" to go there, "click X"). Asking *about* a visible
   element, with no such command, is `chat`: "what does this button do", "what's the green button",
   "tell me about the search bar", "explain how to find the menu". A UI word in the sentence does
   not make it `find_action`; the user wants an explanation, not a cursor move.

6. **A question about the assistant's capabilities is `chat`, whatever domain it names.**
   "can you see my screen", "you can play songs right?", "you can remember my name right?" ask
   what the assistant can do; nothing should be played, stored, or clicked. Domain vocabulary
   (play, remember, text, screen) does not change that. The line: a hedged imperative with a
   concrete object is still a command ("can you play despacito" is `integration`); a question
   about ability with no specific thing to act on is `chat`.

## Tie-breaker

If a command still fits more than one after the rules above, pick the first match in this order:

`memory` > `integration` > `find_action` > `chat`

`chat` is the default. If nothing else clearly applies, it is `chat`.

## Reject class (`none`)

A fifth label, `none`, sits outside the four routing intents. It is **not** a
command type: it marks out-of-distribution or garbled input the router should
not act on (gibberish, other languages, off-domain prose, ASR noise). It exists
because the model is otherwise overconfident, shown only valid commands, it
labels junk as one of the four at ~98% confidence, so a confidence gate can't
catch it. Training on a `none` class of generated OOD (see `Scripts/gen_ood.py`)
lets the model say "I don't know" directly. Peeky treats a `none` prediction as
"defer to Claude", never as a routing target. `none` is never hand-labeled or
emitted by the Claude teacher; it is learned only from generated OOD data.
