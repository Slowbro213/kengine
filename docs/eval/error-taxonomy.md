# Baseline error taxonomy

Use this rubric to label wrong answers from a baseline run (spec §6, step zero). The counts decide which checkers matter most, so label what actually went wrong, not what a checker could catch.

## How to label

1. Open `failures_sample.jsonl` in a results directory. Each line has the question, the gold answer, the model's answer, the status and a `transcript_excerpt` (the end of the reasoning, `---`, then the end of the visible reply).
2. For each line, find the **first error that made the answer wrong** and give it that label.
3. Add a second label only if there's another, **independent** error that would also have made the answer wrong on its own. Errors that only follow from the first one don't get labels.
4. Write the labels to `taxonomy.jsonl` in the same directory, one JSON object per line:

   ```json
   {"item_id": "gsm8k-1042", "labels": ["arithmetic_error"], "note": "12*15 written as 170"}
   ```

   `labels` needs at least one entry. `note` is optional but helps the owner review: say where the error is.
5. Run `kengine eval-summarize <results dir>`. It validates every line (errors name the line number), prints the counts and writes `taxonomy_summary.json`.

If the excerpt is too short to see the error, label from what you can see and say so in `note`. Don't guess at hidden reasoning.

## Labels

| Label | Definition | Example |
|---|---|---|
| `fabricated_fact` | States a fact that isn't in the question or documents and is false, or isn't supported by them where the task requires it. | HotpotQA: "The director was born in 1952" when no document says so and the real year is 1949. |
| `invalid_inference` | A reasoning step that doesn't follow from what came before, though the inputs were right. | FOLIO: from "All A are B" and "X is B" concludes "X is A". |
| `arithmetic_error` | A calculation done wrong. The setup was right; the number isn't. | "12 × 15 = 170". |
| `inconsistency` | Contradicts something it said earlier, or uses two different values for the same quantity. | Says Tom has 8 apples, later uses 6 for the same count. |
| `forgotten_constraint` | Drops a condition from the question partway through. | "How many are left after she gives away a third?" — never subtracts the third. |
| `misread_question` | Answers a different question from the one asked, or misreads a quantity or entity from the start. | Asked for the total cost, gives the cost per item. |
| `format_failure` | The reasoning reached the right answer but the final answer line is missing, malformed or unparseable. | Ends with "so it's forty-two" and no `Final answer:` line; or gives "True" when the gold label set was asked as "Uncertain" spelled out differently. |
| `truncated` | Ran out of tokens before answering (status `partial`, or the reasoning stops mid-thought). | Reasoning loops re-checking the same step until the budget runs out. |
| `other` | Anything else. Explain in `note`. If several items end up `other` for the same reason, tell the owner: it may need its own label. | The gold answer looks wrong. |

## Tie-breaks

- **Right reasoning, wrong format** is `format_failure`, not the reasoning labels.
- **Wrong because of a wrong fact used in a correct inference** is `fabricated_fact`.
- **Thinking ran away and then guessed** is `truncated` if there's no answer, otherwise label the error in the guess.
- **`error` status** (backend failure) isn't a model error. Don't label it; note how many there were in the checkpoint summary.
