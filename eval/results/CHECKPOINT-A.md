# Checkpoint A: baseline on the real model

Run on 2026-11-02/03, branch `feat/foundation`. Solver: `baseline` (one call, thinking on, no checks), seed 0.

## Setup

- GPU: NVIDIA GeForce RTX 5060, 8 GB, driver 570.195.03.
- llama.cpp 0.8.0 (build 11960, commit 8642541), built against CUDA 12.8 (the newest the 570 driver accepts). About 67 tokens/s generation.
- Model: `Huihui-Qwen3.5-9B-abliterated.i1-Q4_K_M.gguf`.
- Server command:

  ```
  llama-server -m models/Huihui-Qwen3.5-9B-abliterated.i1-Q4_K_M.gguf \
    -ngl 99 -c 32768 -np 1 -fa on -ctk q8_0 -ctv q8_0 \
    --reasoning-format deepseek --host 127.0.0.1 --port 8080
  ```

- Token budget per task: 32768 completion tokens (`eval.token_budget_per_task`). Think sampling: temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5.
- FOLIO was loaded through the `tasksource` fallback (item ids `folio-tasksource-N`), not the gated upstream dataset.
- No item ended in status `error` in any run.

## Results

| Suite | Size | n | Headline | Abstention | Answered acc. | Mean wall s/item | Mean completion tokens |
|---|---|---|---|---|---|---|---|
| gsm8k | smoke | 10 | 1.00 | 0.00 | 1.000 | 87.5 | 5,569 |
| gsm8k | small | 100 | **0.93** | 0.03 | 0.959 | 91.8 | 5,780 |
| folio | smoke | 10 | 0.50 | 0.20 | 0.625 | 334.3 | 20,398 |
| folio | small | 100 | **0.46** | 0.16 | 0.548 | 320.7 | 19,614 |
| hotpotqa | smoke | 10 | 0.898 (EM 0.80) | 0.00 | 0.898 | 32.7 | 2,082 |
| hotpotqa | small | 100 | **0.822** (EM 0.65) | 0.00 | 0.822 | 37.3 | 2,361 |

Abstention here is always `partial`: the reply hit the token budget with no `Final answer:` line.

**HotpotQA scoring (owner decision, 2026-11-03): partial credit.** The headline is mean token F1. Exact match (EM) is kept as `exact_accuracy` in `metrics.json` and shown in brackets above. These numbers were first computed by EM, as the plan specified, and re-scored with `kengine eval-rescore`. The failure samples and taxonomies are unchanged: they were drawn from the EM misses. As a rough guide, and not a metric the code computes, a normalized containment match (one string contains the other) gives 0.86 on small.

## Error taxonomy (hand-labelled failure samples)

| Suite / size | Items labelled | Labels |
|---|---|---|
| gsm8k small | 7 | other 3, truncated 3, misread_question 1 |
| folio smoke | 5 | invalid_inference 3, truncated 2, fabricated_fact 1 |
| folio small | 25 | invalid_inference 16, truncated 8, other 1 |
| hotpotqa smoke | 2 | other 2 |
| hotpotqa small | 25 | other 23, format_failure 1, misread_question 1 |

Top 3 overall: **invalid_inference** (19), **other** (29, all of them scoring or dataset problems, not model errors; see below), and **truncated** (13).

Notes per suite, from the `note` fields:

- **gsm8k.** Zero arithmetic errors. The 3 `other` are dataset problems: one inconsistent question (255), one wrong gold (454), one debatable gold (649). Two truncations (286, 300) reached the right answer and then looped on "I'll output it now" until the budget ran out. The third (1067) degenerated into whitespace.
- **folio.** The model's own logic is the weak point, as the plan expected. Most failures are converse errors, taking consistency for entailment, or hedging to Uncertain when a counterexample proves False. A third of failures are runaway thinking that uses up 32k tokens, two of them as whitespace degeneration. Item 1244's premises contradict each other.
- **hotpotqa.** The model is usually right and exact match is the problem: 19 of 25 sampled misses are answers that are correct but phrased differently ('Ricky Rubio' vs 'Ricard Rubio i Vives', 'The Ansonia' vs 'The Ansonia Hotel'). Three more have bad gold answers (Mozart/Spontini is wrong in the gold, one gold is a sentence, one is truncated), and one is defensible (Bruce Springsteen Band vs E Street Band). Only 2 of 25 are real model errors: a yes/no question answered with names, and one misread 'later than which'. 24 of all 35 misses have F1 ≥ 0.5.

## Insights for the owner (review point 1)

1. **Scoring rule for hotpotqa.** Exact match understated the model by about 17 points. Decided: F1 partial credit is the headline from now on (see above).
2. **Truncation and runaway thinking are the cheapest wins.** They cause 13 of 37 labelled failures on gsm8k and folio, including cases where the model already had the right answer. kengine's per-step THINK budget and loop detection target exactly this.
3. **folio is where checking should pay off**, but only once there's a logic checker (M2). M1 has only the structure and arithmetic checkers.
4. **The M1 gate blocks more than the runbook says.** The runbook warns that folio and hotpotqa will abstain heavily under `standard`. The live `kengine run` pen example shows gsm8k word problems will too:
   - The computation `1.25 * 12 = 15` was **verified**.
   - But it rests on "Each pen costs $1.25" and "There are 12 pens". The extractor may only emit `fact` for these, since `given` is reserved for the task, and M1 never turns the task into `given` claims.
   - Facts stay `unchecked` in M1, and the gate checks the full premise closure. So the answer `15` was blocked at every step, and the run ended `blocked` after 7 steps, 25 calls and 503 s.
   - Without a fix, kengine `standard` on gsm8k in Checkpoint B would mostly measure this gap, not the checks. The owner left the fix to me. It's described in `docs/real-machine-runbook.md` (M1 gate notes) and was in place before Checkpoint B ran.

## Live agent checks (Checkpoint B steps 2–3, run early)

- `KENGINE_TEST_LLM_URL=http://127.0.0.1:8080 uv run pytest -m llm -v`: 1 passed (10 s).
- `kengine run "A shop sells pens at \$1.25. How much do 12 pens cost?" --profile standard` printed:

  ```
  status: blocked
  blocked answer (unverified): 15
  ```

  Exit code 3. `events.jsonl` starts with `session_started` and ends with `run_finished`; `beliefs.sqlite` has the beliefs, changes and verdicts tables; the computation belief is `verified`. See insight 4 for why it was still blocked.
