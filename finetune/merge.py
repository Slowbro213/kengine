"""Merge a LoRA adapter into the base model's weights, one safetensors shard at a time.

Each targeted weight becomes `W + scale * B @ A`. Loading one shard at a time keeps memory
to about the largest shard, so the 9B model merges on a machine with little RAM. The other
files (config, tokenizer, chat template) are copied unchanged, so llama.cpp converts the
result exactly as it converts the base.

    python merge.py --base BASE_DIR --adapter finetune/out/adapter --out MERGED_DIR
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

PREFIX = "base_model.model."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads((args.adapter / "adapter_config.json").read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    lora = load_file(str(args.adapter / "adapter_model.safetensors"))
    # base weight name -> (A, B)
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key, a in lora.items():
        if ".lora_A." not in key:
            continue
        name = key.removeprefix(PREFIX).replace(".lora_A.weight", ".weight")
        pairs[name] = (a, lora[key.replace(".lora_A.", ".lora_B.")])
    if len(pairs) * 2 != len(lora):
        raise SystemExit(f"adapter has {len(lora)} tensors, expected {len(pairs) * 2}")

    args.out.mkdir(parents=True, exist_ok=True)
    shards = sorted(args.base.glob("*.safetensors"))
    merged = 0
    for shard in shards:
        with safe_open(str(shard), "pt") as f:
            metadata = f.metadata()
        tensors = load_file(str(shard))
        for name, w in tensors.items():
            if name in pairs:
                a, b = pairs[name]
                delta = scale * (b.float() @ a.float())
                tensors[name] = (w.float() + delta).to(w.dtype)
                merged += 1
        save_file(tensors, str(args.out / shard.name), metadata=metadata or {"format": "pt"})
        print(f"{shard.name}: {merged} merged so far", flush=True)
    if merged != len(pairs):
        missing = sorted(set(pairs) - {n for s in shards for n in load_file(str(s))})[:5]
        raise SystemExit(f"merged {merged} of {len(pairs)}; not in the base: {missing}")
    for f in args.base.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, args.out / f.name)
    print(f"merged {merged} weights (scale {scale}) into {args.out}")


if __name__ == "__main__":
    main()
