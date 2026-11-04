# Checkpoint C: the logic checker (M2) on folio

Run on 2026-11-03 and 04 from `main` at b4b2976, on the same machine, model (`Huihui-Qwen3.5-9B-abliterated.i1-Q4_K_M.gguf`) and seed as Checkpoints A and B. All results are in `eval/results/checkpoint-c/`. The scripts that produced the tables below are in `checkpoint-c/scripts/`.

## Setup

- Server:

  ```bash
  llama-server -m models/Huihui-Qwen3.5-9B-abliterated.i1-Q4_K_M.gguf \
    -ngl 99 -c 32768 -np 1 -fa on -ctk q8_0 -ctv q8_0 \
    --reasoning-format deepseek --spec-type ngram-mod --host 127.0.0.1 --port 8080
  ```

- Formal syntax: `text`, from the Task 7 probe (below).
- THINK role: `max_tokens` 16384, with a `thinking_budget` of 12288. The server closes the thinking block at 12288 tokens and leaves the rest for the `Working:` section. This was added before these runs: in the first, abandoned Checkpoint C attempt, 5 of 15 THINK calls on folio smoke were cut off with no claims. With the budget, 2 of 14 were.
- `--spec-type ngram-mod` was added at the same time. It doesn't change any output, and it didn't speed anything up: THINK ran at 63.7 tokens/s with it and 62.8 without. While thinking, 10–15% of its drafted tokens were accepted. A Qwen3.5-0.8B draft model was slower: 47 tokens/s against 66.
- **N = 30.** The smoke runs took 495.7 s/item (`quick`) and 510.4 (`standard`), so `t` = 1006 s, and floor(36000 / 1006 / 10) × 10 = 30. The main runs are the first 30 items of the seeded `small` sample.

## The formal-syntax probe (Task 7)

40 folio items. The givens and the conclusion are formalized directly, and z3 decides the label.

| Syntax | All givens formalized | z3 decided | Correct |
|---|---|---|---|
| **text** (chosen) | 23 | 30 | **22** |
| json | 25 | 27 | 16 |
| text with the "flat" prompt fix | 27 | 32 | 20 |

Text was chosen because it was correct most often. The flat fix formalized more items but got fewer right, so it was reverted.

For comparison, the baseline is right on 53% of folio items, and the text probe on 55% (22/40), without the model reasoning at all.

## Results

The baseline is Checkpoint A, on the same items.

| Run | n | Headline | Abstention | Answered acc. | Mean wall s/item | Baseline acc. (same items) | Baseline wall s/item |
|---|---|---|---|---|---|---|---|
| folio smoke quick | 10 | 0.40 | 0.50 | 0.800 | 496 | 0.50 | |
| folio smoke standard | 10 | 0.30 | 0.70 | 1.000 | 510 | 0.50 | |
| folio small quick (first 30) | 30 | 0.233 | 0.733 | **0.875** (7/8) | 513 | 0.533 | 299 |
| folio small standard (first 30) | 30 | 0.200 | 0.733 | 0.750 (6/8) | 548 | 0.533 | 299 |

Statuses:

- small quick: answered 8, blocked 22.
- small standard: answered 8, blocked 19, partial 3 ("partial" means the task's token budget ran out).

The only wrong answers, three in all, were "True" where the gold label is "Uncertain": item 1361 in `quick`, and items 1014 and 1271 in `standard`.

## With vs. without

From `eval-compare` against the baseline, on the 30 small items:

| Profile | Fixed | Broke | Caught | False alarms | Errors caught | False-alarm rate |
|---|---|---|---|---|---|---|
| quick | 1 | 0 | 12 | 10 | 0.929 (13/14) | 0.625 (10/16) |
| standard | 2 | 1 | 11 | 11 | 0.929 (13/14) | 0.688 (11/16) |

The table measures the gate against the baseline's answers. The gate actually sits on kengine's own answers, so here is what it held back. `run_finished` records each blocked answer:

| Profile | Passed: right / wrong | Blocked: right / wrong / none |
|---|---|---|
| quick | 7 / 1 | 7 / 15 / 0 |
| standard | 6 / 2 | 13 / 7 / 2 |
| both | **13 / 3 (81% right)** | **20 / 22 / 2 (48% right)** |

- kengine's own reasoning is about as good as the baseline's: 14/30 right in `quick` and 19/30 in `standard`, counting blocked answers, against the baseline's 16/30.
- The gate does separate right from wrong: answers that pass are right 81% of the time, blocked ones 48%.
- But it passes too few answers. 20 right answers were blocked, against 13 let through.

## Logic verdicts

Counted by `scripts/count_logic.py` over all folio sessions in `checkpoint-c/`:

| Run | pass / verified | fail ("doesn't follow") | fail / contradicted | unknown |
|---|---|---|---|---|
| smoke quick | 46 | 18 | 2 | 12 |
| smoke standard | 72 | 21 | 4 | 9 |
| small quick | 115 | 57 | 9 | 33 |
| small standard | 192 | 64 | 7 | 37 |
| **total** | **425** | **160** | **22** | **91** |

## What blocks

From `scripts/block_reasons.py`: the beliefs blocking each blocked item at its last gate, grouped by cause. An item can have more than one cause.

| Cause | small quick (22 blocked) | small standard (22 blocked or partial) |
|---|---|---|
| logic: a step doesn't follow from what it cites | 13 | 15 |
| a step has no formal form (the extractor left it out) | 10 | 6 |
| **a step was never checked (a bug, below)** | 7 | 0 |
| logic: a premise it cites has no formal form | 6 | 4 |
| logic: the givens don't settle it | 6 | 4 |
| logic: the claims it cites contradict each other (mistranslated givens) | 1 | 2 |
| contradicted | 2 | 1 |

Two `standard` items ended without any gate: the budget ran out first.

**A bug: steps that are never checked.** Under `quick`, the loop checks only the new claims the step's answer rests on. A claim that wasn't load-bearing when it was added is skipped, and nothing checks it later, even after a final answer comes to rest on it. It then stays unchecked and blocks every later gate. This happened in 7 of the 22 blocked `quick` items. `standard` checks every new claim, so it isn't affected. Fixed after this checkpoint, in 464c70e.

**Is "doesn't follow" a real error or a bad translation?** I checked six by hand, sampled at random from `quick`:

| Item | Blocked step | Verdict |
|---|---|---|
| 609 | "German is a Romance language", from "German is related to Spanish" | real reasoning error, caught |
| 1365 | "the squirrel case makes Rockie skittish", stated as a fact | real error: one branch of a case split asserted outright |
| 621 | "Beethoven is a composer", from "composers write music pieces" | real error: the converse of the given |
| 726 | "four-sided things are shapes", restating b2 | **translation**: b2 was formalized as `four_sided(x)`, without its "if" part |
| 565 | "coughing is uncertain" | **translation**: formalized as the bare atom `coughing` |
| 242 | "he no longer plays for Bayern" | **translation**: the given says `player(x)`, but b2 says `soccer_player(x)` |

So about half the "doesn't follow" blocks are real errors, and half come from translation. Add the steps with no formal form, and **translation is behind roughly half of all blocks**.

**Translating the givens** (from each item's `formalize` event):

- In `quick`, 13 of 30 items had at least one given the parser rejected. 157 of 170 givens got a formal form.
- The most common rejections, over both small runs:
  - a name starting with a digit, like `1984` (9 times)
  - a comparison missing a side (6)
  - an operator in the wrong place, such as `athlete(djokovic) -> ...` after a predicate name (4)
  - `&` and `|` mixed without brackets (2)
  - apostrophes in a name, like `korea's_republic` (2)
- Universal givens ("All X are Y") lost their "if" part 4 times out of 43.

**The model can't recover from a bad given.** In 726, b4 restates b2 correctly but fails, because b2's formal form is wrong. Over three more steps the model rewrites the same chain four times, and the gate blocks b4 every time. Nothing in the loop can re-translate a given.

## Audit trails

From `scripts/trail.py`, shortened.

**Fixed: 241 (quick), answered False, correct. The baseline got it wrong.** The extractor recorded the verdict item as `undetermined` (Uncertain), but the reasoning said False. The checker showed the conclusion's negation follows, the model rested the answer on the verified negation, and the gate let it through.

```
b4 [given] If a player leaves a team they no longer play for that team. | left_team(x, y) -> not plays_for(x, y)
b7 [inference] ... the negated playing state | prem b5, b6 | not plays_for(robert_lewandowski, bayern_munchen)
   verdict b7 logic pass/verified: follows from b5, b6
b8 [inference] Evaluating the conclusion ... | prem b7 | undetermined plays_for(robert_lewandowski, bayern_munchen)
   verdict b8 logic fail/contradicted: it isn't open: its negation follows from b7
GATE step 1 blocked [b8]
...
b16 [inference] This deduced negation ... | prem b15 | not plays_for(robert_lewandowski, bayern_munchen)
   verdict b16 logic pass/verified: follows from b15
GATE step 3 allowed
RESULT answered False (3 steps)
```

**Falsely blocked: 726 (quick). The baseline got it right, and the blocked answer, True, was right.** The givens lost their "if" parts, and the model could never get the restated b2 verified.

```
TASK All squares are four-sided. / All four-sided things are shapes. / Conclusion: All squares are shapes.
b1 [given] All squares are four-sided.        | four_sided(x)
b2 [given] All four-sided things are shapes.  | four_sided(x)
b4 [inference] That four-sided thing is a shape | prem b2 | four_sided(x) -> shape(x)
   verdict b4 logic fail: doesn't follow from b2: they allow a case where four_sided(_w1) is true and shape(_w1) is false
GATE step 1 blocked [b4]
GATE step 2 blocked [b4]
GATE step 3 blocked [b4, b6]
GATE step 4 blocked [b4, b6, b12, b13]
RESULT blocked, blocked answer True (4 steps)
```

**Blocked by an informal inference: 904 (quick). The blocked answer, True, was wrong; gold is Uncertain.** The given that mentions 1984 was rejected, because `1984` isn't a valid name. The steps resting on it had no formal form, so the wrong answer was blocked only because nothing could check it.

```
b5 [given] If 1984 is a streaming service, then 1984 is a hardcover book. | (none: "arguments must be names, not '1984'")
b8 [inference] Therefore 1984 is printed text. | prem b6, b7 | (none)
GATE step 1 blocked [b8]
b13 [fact] If 1984 is a streaming service, ... | prem b7 | (none)
   verdict b13 restatement unknown: 1984 isn't in the task's givens or verified calculations it cites
GATE step 3 blocked [b13, b14]
RESULT blocked, blocked answer True (3 steps)
```

## gsm8k regression

Formalization is on for every suite, so gsm8k smoke was rerun.

| Profile | Checkpoint B | Checkpoint C | Fixed | Broke | Mean wall s/item | Mean completion tokens |
|---|---|---|---|---|---|---|
| quick | 0.60 | **0.90** | 4 | 0 | 297 → 220 | 18,912 → 12,642 |
| standard | 0.70 | **0.90** | 2 | 0 | 277 → 201 | 17,755 → 12,490 |

- The baseline is right on all 10. kengine still never gives a wrong answer.
- The gains come from the M1 fixes made after Checkpoint B (counts written as calculations, unit conversions, ordinals) and from the THINK budget.
- The remaining misses:
  - **1047 (standard, partial):** algebra with an unknown, as before.
  - **976 (quick, blocked; its blocked answer, 540, was right):** this is **a new kind of false alarm caused by M2**. The extractor recorded sums as `inference` claims with equations, such as `gingerbread_total = 25` from `gingerbread_sun = 15` and b7. The formal form of a sum doesn't state the addition, so z3 found a counterexample (`gingerbread_total = 2`) and failed the step. The arithmetic checker would have verified it as a `computation`.

## Insights for the owner

1. **What M2 bought:** a trustworthy "yes". When kengine answers a folio question, it's right 81% of the time across both profiles, against 53% for the plain model. And it caught 13 of the baseline's 14 errors on these items. On logic, the checker finds real mistakes: a converse, an asserted branch of a case split, a non sequitur.
2. **What it cost:** it answers only 27% of questions, so the headline drops from 0.53 to 0.20–0.23. Each item takes 1.7× as long (513 s against 299). Right now it is a filter, not a booster: the model's reasoning under kengine is no better than without it, and the gate throws away 20 right answers to stop 22 wrong ones.
3. **Is formalization reliable enough?** Not yet. Translation causes roughly half of all blocks. Some causes are cheap to fix: a parser that rejects names like `1984`, givens that lose their "if" part, and inconsistent predicate names between a given and a step. The worst problem is structural: once a given is translated wrongly, every step that uses it fails, and the model has no way to fix it.
4. **What blocks most:** steps that "don't follow" (13–15 of 22 blocked items), split roughly evenly between real errors and translation errors. Next come steps with no formal form.
5. **The fine-tune question.** Translation is a real bottleneck, but not the only one. Real reasoning errors are behind the other half of blocks, and the checker is right to stop them. My recommendation:
   - **Not yet.** First make the cheap fixes (below) and rerun these 30 items. That takes a day and costs nothing.
   - **Then fine-tune for translation if it still dominates.** It's unusually cheap to try. FOLIO's train split (1,001 items; our eval uses the 203-item validation split) comes with expert formal-logic translations of every premise and conclusion. That's ready-made supervised data for the formalize role, and z3 can check each training example. A LoRA on a 9B model needs about 24 GB of GPU memory, which means a rented GPU for a few hours, plus the Hugging Face weights.
   - A fine-tune for *reasoning* is the bigger, later project: expert iteration on z3-verified traces.
6. **The cheap fixes ("M2.1"):**
   - **Check claims that become load-bearing later.** This fixes the bug above, which affected 7 of 22 blocked `quick` items.
   - **Re-translate a given when a restating step fails.** When a step that cites a single given, with matching text, fails against it, re-formalize the given instead of blaming the step. This targets item 726's dead end.
   - **Accept names that start with a digit,** or tell the formalizer to write `n1984`, and normalize apostrophes.
   - **Keep numeric equations away from z3.** Send equation-only "inferences" to the arithmetic checker as computations, or have the logic checker return `unknown`, not `fail`, when a step's equations need arithmetic. This fixes gsm8k-976.

   Also add one prompt rule for the three wrong answers: "True" needs the conclusion to be derived; failing to refute it means Uncertain.
7. **M3 or M5 next?** Neither fixes folio's blocks; those are M2's own. M3 (facts and retrieval) is the next real gap: hotpotqa is still 100% blocked. The M5 critic would mostly duplicate what z3 already catches on logic. My recommendation: do M2.1 first (about a day), then M3, as the plan orders. Dropping or reordering a milestone is your call.
