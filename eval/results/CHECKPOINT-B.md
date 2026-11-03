# Checkpoint B: kengine (M1) vs. baseline

Run on 2026-11-03 from `main` at 4bc3a1a (M1 plus the givens and restatement fix). Same machine, server, model and seed as Checkpoint A. Token budget per task: 32768 completion tokens for both solvers. kengine's THINK step budget: 8192 tokens.

## What was run, and what was cut

- **gsm8k smoke, `quick` and `standard`.** Full runs, 10 items each.
- **hotpotqa smoke, `quick` only, first 5 items.**
- **folio smoke, `quick` only, first 5 items.**

The small runs (100 items) and the remaining smoke runs were not done. This is a builder decision under the owner's "do as you wish" for Checkpoint B:

- **Time:** kengine took 4.6 to 10 minutes per item. Six 100-item runs would have taken about 60 GPU-hours.
- **Predictable outcome:** M1 has no checker for document facts (M3) or logic (M2). On hotpotqa and folio, every answer rests on claims M1 can't check, so the gate blocks them. The partial runs confirm this. The `standard` gate is stricter than `quick`, so it can't do better there.
- **Where the time went instead:** the GPU went to M2, where folio can be checked.

## Results

The baseline numbers are on the same items (Checkpoint A, smoke).

| Suite | Profile | n | Headline | Abstention | Answered acc. | Mean wall s/item | Mean completion tokens | Mean model calls |
|---|---|---|---|---|---|---|---|---|
| gsm8k | baseline | 10 | 1.00 | 0.00 | 1.000 | 87.5 | 5,569 | 1 |
| gsm8k | quick | 10 | 0.60 | 0.40 | **1.000** | 297.3 | 18,912 | 4.7 |
| gsm8k | standard | 10 | 0.70 | 0.30 | **1.000** | 276.8 | 17,755 | 6.7 |
| hotpotqa | baseline | 5 | 0.836 (EM 0.80) | 0.00 | 0.836 | 26.0 | 1,653 | 1 |
| hotpotqa | quick | 5 | 0.00 | 1.00 | n/a | 574.6 | 36,508 | 11.6 |
| folio | baseline | 5 | 0.80 | 0.00 | 0.800 | 333.9 | 20,434 | 1 |
| folio | quick | 5 | 0.20 | 0.80 | **1.000** | 479.0 | 30,598 | 6.0 |

Statuses:

- gsm8k quick: answered 6, blocked 4.
- gsm8k standard: answered 7, blocked 3.
- hotpotqa quick: blocked 5.
- folio quick: answered 1, blocked 2, partial 2. "Partial" means the task's token budget ran out.

**kengine never gave a wrong answer.** Every miss was an abstention. The answers it held back were mostly right:

| Run | Blocked | Blocked answer was right |
|---|---|---|
| gsm8k quick | 4 | 4 |
| gsm8k standard | 3 | 3 |
| hotpotqa quick | 5 | 4 (one more partly right) |
| folio quick | 2 | 2 |

**The checkers caught nothing.**

| Checker | Pass | Unknown | Fail |
|---|---|---|---|
| arithmetic | 75 | 7 (expressions with variables) | 0 |
| restatement | 22 | 17 | 0 |
| structure | 167 | 0 | 0 |

No claim was ever contradicted. On these items, M1's only effect was to hold answers back.

## Where the time goes

Feedback items across the four runs. A THINK step is one free-reasoning call.

| Run | THINK steps | Cut off at 8192 tokens | Extraction failed | Gate blocked |
|---|---|---|---|---|
| gsm8k quick | 27 | 7 | 3 | 12 |
| gsm8k standard | 25 | 11 | 1 | 9 |
| hotpotqa quick | 31 | 4 | 14 | 52 |
| folio quick | 21 | **12** | 3 | 19 |

A THINK step that's cut off before its `Working:` section yields no claims, so its tokens are lost. On folio, more than half were lost. The 8192 step budget is too small for this model on logic: the baseline's correct folio answers took a median of 10,667 completion tokens, and only 37% of them fit in 8192.

## Audit trails

There was no item where kengine helped, so the first trail shows the loop working as designed instead.

**Worked as designed: gsm8k-621 (standard), answered 3000, correct.** Step 1 was cut off. In step 2, three computations were each verified, and the gate allowed the answer.

```
step 1 feedback: your reasoning was cut off before the Working section; be more concise
step 2 verdict b5 arithmetic pass verified: `2 * 2` = 4
step 2 verdict b6 arithmetic pass verified: `4 * 6` = 24
step 2 verdict b7 arithmetic pass verified: `24 * 125` = 3000
step 2 gate: allowed
step 2 run_finished: answered 3000 (2 steps, 5 model calls)
```

**Hurt (a checker gap): gsm8k-829 (standard), blocked.** The blocked answer, 14000, was right, and the baseline got this item right. The chain itself was verified. But "Week 1 amount is 8,000 bars", citing "Boris produces 8,000 bars of chocolate the first week", failed restatement because the checker didn't read "first" as 1.

```
step 1 verdict b8 restatement unknown: 1 isn't in the task's givens or verified calculations it cites; ...
step 1 verdict b12 arithmetic pass verified: `50000 - 36000` = 14000
step 1 gate: blocked [b8 is unchecked]
step 2 feedback: your reasoning was cut off before the Working section; be more concise
... blocked on b8 until the budget ran out (573 s)
```

**Blocked: gsm8k-733 (quick).** The blocked answer, 5, was right. The question: "While walking down the street with his 3 younger siblings, Greg found $20…". The model wrote "Number of people sharing is 4" as a fact citing b1. Nothing in b1 states 4, and the model never wrote it as `3 + 1 = 4`.

```
step 2 verdict b5 restatement unknown: 4 isn't in the task's givens or verified calculations it cites; cite the given it comes from, or write it as a calculation
step 2 verdict b6 arithmetic pass verified: `20 / 4` = 5
step 2 gate: blocked [b5 is unchecked]
step 3 gate: blocked [b5 is unchecked]
step 4 gate: blocked [b5 is unchecked]
step 5 feedback: your reasoning was cut off before the Working section; be more concise
run_finished: blocked, blocked answer 5 (535 s)
```

## What the blocked gsm8k items needed

| Item | Blocking claim | Cause | Fixed on `feat/m2-logic` |
|---|---|---|---|
| 621 (quick), 733 (both) | "4 people" | a count the model worked out but wrote as a fact | feedback and THINK now say to write it as a calculation (0165326) |
| 861 (quick) | "12 months" | a unit conversion (1 year) | restatement accepts standard unit conversions (18c119b) |
| 829 (standard) | "Week 1 … 8,000" | an ordinal ("first") | restatement reads ordinals (fa4efc6) |
| 1047 (both) | `(S + 4) + S + S/2` | algebra with an unknown | no. The arithmetic checker needs numbers only. A symbolic checker is future work. |

The THINK step budget is raised to 16384 on the same branch (d499d4b). The per-task budget is unchanged. Checkpoint C reruns gsm8k smoke with all of this, and compares the result with the table above.

## Insights for the owner

1. **M1 makes the model more cautious, not more correct.** On gsm8k, an answered kengine result was always right, but the plain model was already right on all 10 items. The checkers caught no errors, because this model makes no arithmetic errors at this level. kengine's value has to come from logic (M2) and facts (M3), where the baseline actually fails.
2. **Most of the cost is lost thinking, not checking.** kengine used 3 to 4 times the baseline's time and tokens on gsm8k. More than a third of its THINK steps were cut off and discarded, and the same blocked claim was re-cited step after step. Both issues are addressed (the step budget, and feedback that names the fix). A smarter response to a repeated block is still open: for example, stop early, or ask the model to drop the claim.
3. **One loophole to remember.** The arithmetic checker verifies that `a * b = c` is computed correctly, not where `a` and `b` came from. A computation built on a number the model made up passes. In these runs the model always tied its computations to givens or facts. It would matter for a model that doesn't. The natural fix is to treat the numbers in an expression like a fact's numbers, cited or not.
4. **hotpotqa and folio under M1 are all abstentions**, as the runbook predicted. They wait for M3 and M2.
