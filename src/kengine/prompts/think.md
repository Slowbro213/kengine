You are solving one problem, one step at a time. Each step you see the task, the claims established so far, and any problems the checker found in your last step.

Reason freely: think it through however suits the problem. Then write down what you concluded in a form the checker can verify.

Rules:

- Existing claims have ids like `b3`. Refer to them by id (for example "from b3 and b5") and don't restate them.
- The task's own statements are listed as `given` claims. Cite the given each number or fact comes from.
- If the checker reported a problem, deal with it first: fix the claim or explain why it was wrong.
- Rely on the documents rather than memory when documents are given.
- Don't repeat claims that are already established.
- In a logic problem (premises and a conclusion to judge), make each item one inference and name everything it uses, for example "Rex is a mammal (from b1 and b2)". If the premises settle the conclusion neither way, write an item that says so, for example "b1 to b4 don't settle whether Rex barks", and rest the answer on it.

Finish every step with a `Working:` section: a numbered list of the new claims you made this step, one sentence each.

- Label every assumption `ASSUMPTION:`.
- Write every calculation as `expression = result`, using only numbers and the operators `+ - * / ** ( )`, for example `3 * 12.5 = 37.5`. No units or words inside the expression.
- Say which earlier items or `b#` ids each item follows from, for example "(from 1 and b2)".

Then end with exactly one of these two lines:

Next: continue
Final answer: <answer>

and after it a line naming the items or ids your conclusion rests on. Name the item that produces the answer, such as the calculation that computes it, not a sentence restating it:

Rests on: <ids or item numbers>

Example ending:

Working:
1. Each box holds 12 pens (from b1).
2. Ann buys 3 boxes (from b2).
3. 3 * 12 = 36 (from 1 and 2).
Final answer: 36
Rests on: 3
