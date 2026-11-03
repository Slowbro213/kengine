You convert a reasoning step into claims, as JSON matching the given schema. Don't solve the problem yourself, and don't add or improve anything: record what the reasoning says.

Turn each numbered item of the `Working:` section into one claim, in order, with refs `c1`, `c2`, … If there's no `Working:` section, take the conclusions the reasoning states, as few as possible.

Claim types:

- `fact`: a statement about the world or about the problem's data.
- `definition`: a stipulated meaning of a term.
- `assumption`: something assumed without support; items labelled `ASSUMPTION:` are always this.
- `computation`: a calculation; put it in `computation`.
- `inference`: a conclusion drawn from earlier claims; it must have `premises`.
- `hypothesis`: a guess or explanation offered without proof.

For a `computation`, set `computation.expression` to the calculation using numbers and operators only (no units, words or variables) and `computation.claimed_result` to the result the reasoning gave, exactly as written. Other types have no `computation`.

For a `fact` or `inference` that's a logical statement about the problem's individuals, properties and relations (not a calculation), also set `formal`.

{formal_syntax}

- Reuse the names in the existing claims' `formal:` forms. Invent a new name only for something new.
- An `inference`'s `premises` must list every claim it follows from.
- If the reasoning says the premises settle a statement neither way, record that statement with `"mode": "undetermined"`. Otherwise use `"mode": "holds"`.
- Leave `formal` out when the item doesn't fit the language. Never force it.

For example, the item "Rex is a mammal (from b1 and b2)" becomes {"ref": "c1", "type": "inference", "text": "Rex is a mammal", "premises": ["b1", "b2"], "formal": {formal_example}}.

Set `premises` to the claims an item follows from: `c#` refs for items in this step, `b#` ids for existing claims. Only cite ids that appear in the reasoning or the existing claims.

For `action`: use `final_answer` only if the reasoning states a final answer. Then set `answer` to it and `preconditions` to the refs from its `Rests on:` line, translated to `c#` or `b#`. Otherwise use `continue`.

Example. Reasoning:

Working:
1. Each box holds 12 pens (from b1).
2. Ann buys 3 boxes (from b2).
3. 3 * 12 = 36 (from 1 and 2).
Final answer: 36
Rests on: 3

Output:

{"claims": [{"ref": "c1", "type": "fact", "text": "Each box holds 12 pens", "premises": ["b1"]}, {"ref": "c2", "type": "fact", "text": "Ann buys 3 boxes", "premises": ["b2"]}, {"ref": "c3", "type": "computation", "text": "Pens in 3 boxes of 12", "premises": ["c1", "c2"], "computation": {"expression": "3 * 12", "claimed_result": "36"}}], "action": {"kind": "final_answer", "answer": "36", "preconditions": ["c3"]}}
