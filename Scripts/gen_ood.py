"""Generate an out-of-distribution (OOD) "none" class for the reject-class fix.

routelet is overconfident: shown only the five valid intents, it confidently
labels gibberish as one of them. Training it on a sixth "none" class of OOD /
garbled text teaches it to actively flag input it shouldn't act on, which is a
far more reliable signal than max-softmax confidence.

This produces varied synthetic OOD across several families (gibberish, char
salad, other languages, off-domain prose, number/symbol noise, ASR disfluency)
so the model learns "weird" broadly rather than one narrow pattern. Output is
split into a training file and a frozen eval file, with no overlap between them
and no overlap with the hand-written report probe set (report/ood_probe.txt),
so the eval stays honest.

Run: uv run python Scripts/gen_ood.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_OUT = PROJECT_ROOT / "data" / "none.jsonl"
EVAL_OUT = PROJECT_ROOT / "evals" / "ood_holdout.jsonl"
PROBE = PROJECT_ROOT / "report" / "ood_probe.txt"
EVAL_FRACTION = 0.2
SEED = 11

_CV = "bcdfghjklmnpqrstvwz"
_V = "aeiou"

FOREIGN = [
    "quelle heure est il maintenant", "ich moechte ein kaltes bier",
    "donde esta la biblioteca por favor", "wo ist der naechste bahnhof",
    "je voudrais un cafe au lait", "como se llama este lugar",
    "il fait tres froid aujourd hui", "dov e la stazione dei treni",
    "watashi wa nihongo ga hanasemasen", "guten morgen wie geht es ihnen",
    "merci beaucoup pour votre aide", "no entiendo lo que dices",
]
OFF_DOMAIN = [
    "the mitochondria is the powerhouse of the cell",
    "photosynthesis converts sunlight into glucose and oxygen",
    "the treaty of westphalia was signed in sixteen forty eight",
    "ribosomes translate messenger rna into proteins",
    "the french revolution began in seventeen eighty nine",
    "water boils at one hundred degrees celsius at sea level",
    "the mariana trench is the deepest part of the ocean",
    "shakespeare wrote romeo and juliet in the fifteen nineties",
    "the speed of light is roughly three hundred thousand kilometers per second",
    "tectonic plates drift a few centimeters every year",
    "the human heart beats about a hundred thousand times a day",
    "jupiter is the largest planet in the solar system",
    "the great wall of china stretches thousands of miles",
    "honeybees communicate the location of flowers by dancing",
    "the renaissance began in florence in the fourteenth century",
    "a group of crows is called a murder",
    "mount everest grows a little taller each year",
    "the amazon rainforest produces a fifth of the world oxygen",
]
DISFLUENCY = [
    "uh um like you know i mean", "er hmm well so anyway", "uhhh ahh ok ok ok",
    "yeah yeah no no wait", "hmm let me think uhh", "so like basically um",
    "wait what no hold on", "errr i dunno maybe sorta",
]
SYMBOLS = ["!@#$ %^&* ()_+", "<<<>>> {}{}{} []", "??? ... ;;; :::", "~~~ === +++ ---"]


def _gibberish_word(rng: random.Random) -> str:
    syl = rng.randint(1, 3)
    return "".join(rng.choice(_CV) + rng.choice(_V) for _ in range(syl)) + (
        rng.choice(_CV) if rng.random() < 0.4 else ""
    )


def _gibberish(rng: random.Random) -> str:
    return " ".join(_gibberish_word(rng) for _ in range(rng.randint(2, 6)))


def _char_salad(rng: random.Random) -> str:
    n = rng.randint(8, 24)
    return "".join(rng.choice(_CV + _V + "    ") for _ in range(n)).strip()


def _numbers(rng: random.Random) -> str:
    parts = [str(rng.randint(0, 9999)) for _ in range(rng.randint(2, 5))]
    return " ".join(parts)


def generate(rng: random.Random) -> list[str]:
    out: set[str] = set()
    # ~150 gibberish + ~80 char salad + ~60 numbers, plus the fixed pools, each
    # variant uniqued by content.
    for _ in range(150):
        out.add(_gibberish(rng))
    for _ in range(80):
        out.add(_char_salad(rng))
    for _ in range(60):
        out.add(_numbers(rng))
    out.update(FOREIGN)
    out.update(OFF_DOMAIN)
    out.update(DISFLUENCY)
    out.update(SYMBOLS)
    return sorted(out)


def main() -> None:
    rng = random.Random(SEED)
    probe = set()
    if PROBE.exists():
        probe = {
            ln.strip().lower()
            for ln in PROBE.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        }

    rows = [r for r in generate(rng) if r.strip() and r.lower() not in probe]
    rng.shuffle(rows)
    n_eval = int(len(rows) * EVAL_FRACTION)
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]

    for path, items in ((TRAIN_OUT, train_rows), (EVAL_OUT, eval_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for text in items:
                f.write(json.dumps({"text": text, "intent": "none"}) + "\n")

    print(f"generated {len(rows)} OOD rows ({len(train_rows)} train -> {TRAIN_OUT.name}, "
          f"{len(eval_rows)} eval -> {EVAL_OUT.name})")


if __name__ == "__main__":
    main()
