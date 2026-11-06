"""Generate THINK then extract replies for the dev chain pairs, with or without an adapter.

generate.py BASE DEV_JSONL OUT_JSONL [ADAPTER]
"""

import json
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

base, dev, out = sys.argv[1:4]
adapter = sys.argv[4] if len(sys.argv) > 4 else None
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
if adapter:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, adapter)
model.eval()


def gen(messages, thinking, max_new):
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        o = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
    return tok.decode(o[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)


pairs = [json.loads(line) for line in open(dev)]
by_item = {}
for p in pairs:
    by_item.setdefault(p["item_id"], {})[p["role"]] = p
with open(out, "w") as f:
    for item_id, d in by_item.items():
        think = gen(d["think"]["messages"], True, 3000)
        reply = think.split("</think>", 1)[-1].strip()
        gold_reply = d["think"]["completion"].split("</think>\n\n", 1)[1]
        msgs = [dict(m) for m in d["extract"]["messages"]]
        assert msgs[1]["content"].endswith(gold_reply)
        msgs[1]["content"] = msgs[1]["content"][: -len(gold_reply)] + reply
        ext = gen(msgs, False, 1500)
        f.write(
            json.dumps(
                {"item_id": item_id, "label": d["think"]["label"], "think": think, "extract": ext}
            )
            + "\n"
        )
        f.flush()
        print(item_id, flush=True)
