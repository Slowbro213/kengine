# M3 threshold check

What this is: the new fact checker and the inference checker, replayed over the claims
of the earlier HotpotQA run (`2026-11-03-hotpotqa-kengine-smoke-quick`, 6 questions).
No LLM was used. The real models ran on the CPU. Script: `eval/scripts/facts_calibration.py`.
All rows are in `calibration.json`.

## Counts (38 checks)

| Checker | verified | supported | contradicted | insufficient / unchecked |
|---|---|---|---|---|
| facts (26 checks) | 23 | 1 | 0 | 2 |
| inference-nli (12 checks) | 0 | 5 | 0 | 7 |

The sixth question (Age of Chance and Vector) had no claims in the log, so it adds no rows.
The consistency checker is not part of this replay.

## Verdicts that look wrong

None that is clearly wrong. I read every row against that question's documents.

- All 23 `verified` are stated nearly word for word in a document (for example "Plateau is a
  two-player abstract strategy board game invented by Jim Albea" for the Plateau claim).
  No wrong `verified`.
- There are no `contradicted` verdicts, so no wrong ones.
- The one fact `supported` ("The Medellin Cartel operated throughout the 1970s and 1980s in the
  specified regions of Bolivia, Colombia, ..., Europe", 60%) is cautious but fine: the passage
  lists exactly those places and decades. Only the vague words "the specified regions" kept it
  from `verified`.
- Two fact `unknown` rows are fair:
  - "Jose Gonzalo Rodriguez Gacha, a key leader, lived from 1947 to 1989, confirming his
    leadership spanned this period". It mixes a fact with a conclusion. The later, cleaner claim
    "Gacha lived from 1947 to 1989" was `verified`.
  - "Pierrepont Noyes became the head of this new firm and held the position for many years".
    The passage says "Noyes later became the head of Oneida Limited, a position he held for many
    years". The words "this new firm" are a pronoun the checker cannot resolve. A little strict,
    but the safe direction.
- The 7 inference `unknown` rows are the "Therefore the answer is ..." type. Most are real
  gaps: for "Therefore, the Appenzeller is the dog based in Switzerland" the premises only say
  Drent is Dutch and Appenzeller is Swiss Alps, so the step "Swiss Alps means based in
  Switzerland" is missing. That is what we want the model to be asked to state. Two Colt Express
  and Plateau "main difference" rows are vague comparisons and rightly not supported.
- The 5 inference `supported` rows (97 to 99%) are all sound. Small oddity: one Colt Express
  `supported` row rests on a premise that is itself an inference claim from the same run
  (the replay marks every claim verified, so it does not chain). Not a threshold problem.

## Threshold change

None. No row was clearly wrong at the defaults, so `FactsConfig` and `kengine.example.toml`
are unchanged.
