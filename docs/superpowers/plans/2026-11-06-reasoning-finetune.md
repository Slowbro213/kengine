# Fine-tune plan, stage 2: teach the model to write steps the checker accepts

**Status: run on RunPod on 2026-11-06 and failed the stop rule. We keep the stage-1 model.** The results are in "What the RunPod run found" at the end.

Stage 1 (`2026-11-05-translation-finetune.md`) taught the model to translate. Translation got much better (probe-formal 134 of 203, against 84), but the full agent barely moved. It's blocked at the reasoning step: in 14 of the tuned model's 19 blocked folio runs, the answer it was trying to give was right, but the checker refused a step ("b79 doesn't follow from b72, b54") or the steps it cited didn't show the conclusion.

## The data

`kengine finetune-chains` writes worked solutions for FOLIO-train, from the same converted expert logic as stage 1, so it uses the same stories and the same held-back 10%.

- **True or False:** the steps apply one given rule at a time to facts already known, for example `dog(rex)` and `dog(x) -> mammal(x)` give `mammal(rex)`, and end in the conclusion or its negation. Only the steps the end needs are kept. When the conclusion is a rule (`square(x) -> shape(x)`), the last step cites the givens that show it.
- **Uncertain:** one step saying the givens don't settle the conclusion, resting on all of them.
- **Every solution is checked by the real logic checker**, step by step, and then the answer check, as the agent would check it. A solution that fails any check is dropped (none did).

Each puzzle gives two training pairs, the two calls the agent makes per step:

1. **THINK:** the exact prompt the agent sends (the task, and the givens with their formal forms). The answer is a short thinking part, then the `Working:` list, the final answer and `Rests on:`.
2. **Extract:** the exact prompt, with that reply as the reasoning to convert. The answer is the claims JSON.

| | Puzzles |
|---|---|
| Converted (stage 1's data) | 393 |
| **With a solution** | **348** (309 train, 39 held back) |
| False, but needs reasoning by cases | 45, dropped |

Labels: 123 True, 53 False, 172 Uncertain. Solution length: 303 have one step, 45 have 2 to 8.

## Test on the small model (home, free)

Trained Qwen3.5-0.8B on the 309 training puzzles (618 pairs), 2 rounds, 15 minutes, 2.6 GB of graphics memory at most. (To fit, `train.py` now works out the model's word scores only for the answer part, not the prompt.) Dev loss 0.031 after round 1, 0.028 after round 2. The loss is very low because the answers follow one fixed pattern.

Then each model answered the 39 held-back puzzles: the THINK call, then the extract call on its own reply. The real checker checked the steps against the givens' formal forms, as in the data. So this tests reasoning when the translation is the expert's. Results are in `eval/results/reasoning-08b/`.

| | Untrained 0.8B | Trained 0.8B |
|---|---|---|
| Accepted by the checker, right | 0 | **14** (4 True, 10 Uncertain) |
| Accepted, **wrong** | 0 | **0** |
| Refused by the checker | 0 | 23 |
| Claims not valid | 18 of 18 | 2 |

- **Untrained:** it never finished thinking within 3,000 tokens on any of the 18 puzzles it reached, so it never gave a reply in the agent's format. I stopped it at 18; the rest weren't going to change the picture.
- **Trained:** it writes the format every time, and the checker stopped every wrong answer: 15 refusals for a step that doesn't follow, 8 for "Uncertain" when the givens do settle it. It's weakest on True and False: it gave True for 9 Uncertain puzzles, and Uncertain for 7 puzzles that were True or False.
- **What it shows:** the data teaches the format and the citing, and the checker catches the mistakes. It doesn't show how the 9B does. The 0.8B is too small to judge reasoning by, and the givens here are the expert's translation, not the model's. Checks 1 and 2 below are the real test.

## What I'd run on a pod

The same `finetune/run.sh` as stage 1, training one new adapter on the base 9B with both datasets together (translation and reasoning), so translation isn't forgotten. About 1 hour on an A40 (about $0.60/hour), so **about $1–2**. Merge, convert and quantize at home, then upload only the GGUF if the checks run on the pod.

**Checks:**

1. **folio, 30 items, full agent:** more right than the tuned stage-1 model (5), and no wrong answers.
2. **The held-back stories (93 puzzles):** the same, on puzzles no eval uses.
3. **gsm8k smoke:** no drop (8 of 10 on the pod).
4. **probe-formal:** translation stays near 134 of 203.

**Stop rule:** if check 1 isn't at least 3 points better, or any check lets a wrong answer through, we keep the stage-1 model.

## Risks

- **More Uncertain, and wrong Uncertain answers get through.** Half the data says Uncertain. The gate accepts Uncertain when z3, working from the translation, finds the conclusion open. But a translation that loses a link also looks open. On the 30 eval items that happens 5 times, and on the held-back stories 26 times in 93. Today those runs are mostly blocked. After training, many could pass as wrong answers. This is the biggest risk. Ways to limit it:
  - drop Uncertain from the data, or keep only a few of those puzzles, so the model learns to prove True/False and isn't pushed toward Uncertain
  - or only accept Uncertain when the model's own reasoning also found no proof after trying, which the checker can't confirm
  - I'd start with the first.
- **Most solutions are one step.** 303 of 348 have one step that cites the right givens and lets z3 do the rest. That's what the checker accepts and it fixes the main block (wrong or missing citations), but it doesn't teach longer step-by-step proofs.
- **The thinking part is short and made up.** The agent runs THINK with thinking on, where the model now thinks at length. Training on short thinking may make it think less on hard puzzles. The alternative is to train only the visible reply and leave the thinking alone, which I can test at home first.
- **Small data, one style.** As in stage 1, the model may learn FOLIO's style more than reasoning. Checks 2 and 3 are there to catch that.

## What the RunPod run found

The owner approved the plan with the first fix for the main risk: only 25% of the Uncertain training puzzles are kept (205 train puzzles, 39 held back; 104 Uncertain dropped). One adapter was trained on the base 9B with both datasets: 860 training pairs, 3 rounds, 51 minutes on an A40, 10 GB of graphics memory at most. The best dev loss was after round 1 (0.073), so that checkpoint was kept. The model was merged, converted and quantized to Q4_K_M on the pod, and the checks ran there. Results are in `eval/results/finetune-runpod-stage2/`.

| Check | Stage 1 (kept) | Stage 2 | Goal |
|---|---|---|---|
| 1. folio, 30 items | 5 right, 0 wrong | **14 right, 3 wrong** | ≥ 8 right, 0 wrong |
| 2. held-back stories, 93 | not run (see below) | **24 right, 18 wrong**, 51 blocked | 0 wrong |
| 3. gsm8k smoke, 10 | 8 right | **4 right**, 1 wrong, 5 blocked | no drop |
| 4. probe-formal | 28 of the same 44 items | **31 of 44** (stopped early) | near stage 1 |

**Stop rule: failed.** Check 1 is 9 points better, but checks 1, 2 and 3 all let wrong answers through, and math dropped by half. We keep the stage-1 model.

**What went wrong:**

- **Wrong "Uncertain" answers got through, as the risk section feared.** 18 of the 22 wrong answers are "Uncertain" when the answer key has a definite answer (3 of 3 in check 1, 14 of 18 in check 2, and the one in math). The model's translation drops a link, z3 then finds the conclusion open, and the gate accepts Uncertain. Keeping only a quarter of the Uncertain data wasn't enough.
- **The Uncertain habit spread to math.** On 5 of the 10 gsm8k problems the model answered "Uncertain". The checker blocked 4 of them, but one got through, and the other blocks cost right answers.
- **True and False got better.** In check 1 all 8 True/False answers were right (stage 1: 5), and in check 2, 12 of 16. So the reasoning data does help with proofs. What fails is Uncertain.
- **Translation held up.** The probe was too slow to finish on the pod (it runs one item at a time), so it was stopped at 44 of 203. On those 44 items stage 2 got 31 right and stage 1 got 28.

**Cost:** the pod ran about 1 h 45 min at $0.59/hour, **about $1.05**. It is deleted.

**Notes for next time:**

- Check 2 on the stage-1 model was started for comparison, but at about 15 minutes per item while sharing the GPU it would have taken hours, so it was stopped after 5 items. The stage-1 shortcut test (`probe-formal --split train-dev`) already covers those stories for translation.
- probe-formal runs one item at a time, so on a pod it should run alone or be split into shards like the evals.
- A setup script failed silently: `uv run kengine --help | head -3` under `set -o pipefail` exits with an error when `head` closes the pipe, so the "setup done" marker never printed and the pod sat idle about 10 minutes. Don't pipe into `head` in a pipefail script.

**What could come next (not approved):**

1. Train on True/False reasoning only (no Uncertain at all), and let Uncertain come only from the checker. The True/False results above suggest this could keep the gains without the wrong Uncertains.
2. Make the gate stricter: accept Uncertain only when a second, independent translation of the givens also finds the conclusion open.
3. Add some gsm8k-style data so the math habit isn't lost.
