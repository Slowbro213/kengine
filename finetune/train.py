"""QLoRA fine-tune of kengine's FORMALIZE call.

The plan is docs/superpowers/plans/2026-11-05-translation-finetune.md. This trains a LoRA
adapter on the pairs `kengine finetune-data` writes. Each pair is the prompt the agent
sends, rendered with the model's chat template as llama-server renders it, plus the answer.
Thinking is off unless the pair says `"thinking": true` (the THINK pairs from `kengine
finetune-chains`, whose answer is the thinking, `</think>`, then the reply). Only the
answer's tokens count toward the loss.

The adapter with the lowest dev loss is kept. Training stops early at --max-hours.

    python train.py --base BASE_DIR --data finetune/data --out finetune/out
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# The language model's projections: full attention, linear attention and MLP. Not the
# vision tower, and not linear attention's small gate projections (in_proj_a, in_proj_b).
TARGETS = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj"
    r"|gate_proj|up_proj|down_proj)"
)
END = "<|im_end|>\n"


def encode(tok: Any, pair: dict[str, Any], max_len: int) -> dict[str, list[int]]:
    prompt = tok.apply_chat_template(
        pair["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=pair.get("thinking", False),
    )
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(pair["completion"] + END, add_special_tokens=False)["input_ids"]
    if len(p) + len(c) > max_len:
        raise ValueError(f"{pair['item_id']} is {len(p) + len(c)} tokens, over {max_len}")
    return {"input_ids": p + c, "labels": [-100] * len(p) + c}


def load(path: Path, tok: Any, max_len: int) -> list[dict[str, list[int]]]:
    pairs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [encode(tok, p, max_len) for p in pairs]


class Collate:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [self.pad_id] * (n - len(b["input_ids"])) for b in batch]
        labels = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        mask = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        # Logits only from the token before the first answer token on, since nothing before
        # counts toward the loss. Over a 250k vocabulary, the full sequence's logits don't
        # fit in 8 GB.
        start = min(next(i for i, t in enumerate(b["labels"]) if t != -100) for b in batch)
        keep = n - start + 1
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels)[:, -keep:],
            "attention_mask": torch.tensor(mask),
            "logits_to_keep": keep,
        }


class StopAfter(TrainerCallback):
    """Ends training, saving a checkpoint first, once `hours` have passed."""

    def __init__(self, hours: float) -> None:
        self.deadline = time.monotonic() + hours * 3600

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if time.monotonic() > self.deadline:
            control.should_training_stop = True
            control.should_evaluate = True
            control.should_save = True
        return control


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="HF model directory")
    ap.add_argument("--data", type=Path, required=True, help="train.jsonl and dev.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=-1, help="for a quick test")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=8, help="pairs per optimizer step")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-hours", type=float, default=2.0)
    ap.add_argument("--bf16-base", action="store_true", help="LoRA on a bf16 base, not 4-bit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    train = load(args.data / "train.jsonl", tok, args.max_len)
    dev = load(args.data / "dev.jsonl", tok, args.max_len)
    print(f"train {len(train)} pairs, dev {len(dev)}", flush=True)

    quant = None
    if not args.bf16_base:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["visual", "lm_head"],
        )
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, quantization_config=quant, device_map={"": 0}
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        target_modules=TARGETS,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(train) / args.accum)
    every = args.max_steps if args.max_steps > 0 else steps_per_epoch
    targs = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(0.05 * steps_per_epoch * args.epochs)),
        weight_decay=0.0,
        logging_steps=max(1, min(5, every)),
        eval_strategy="steps",
        eval_steps=every,
        save_strategy="steps",
        save_steps=every,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=Collate(tok.pad_token_id),
        callbacks=[StopAfter(args.max_hours)],
    )
    t0 = time.monotonic()
    trainer.train()
    trainer.save_model(str(args.out / "adapter"))
    tok.save_pretrained(str(args.out / "adapter"))
    summary = {
        "train_pairs": len(train),
        "dev_pairs": len(dev),
        "best_eval_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "steps": trainer.state.global_step,
        "hours": round((time.monotonic() - t0) / 3600, 3),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "args": {k: str(v) for k, v in vars(args).items()},
        "log": trainer.state.log_history,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "log"}), flush=True)


if __name__ == "__main__":
    main()
