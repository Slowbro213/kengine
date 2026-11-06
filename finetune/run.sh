#!/usr/bin/env bash
# On the rented GPU: install, fetch the base model, train. Only the adapter (tens of MB)
# comes back; merging, conversion to GGUF and quantizing happen on the home machine.
#
#   git clone https://github.com/Slowbro213/kengine && cd kengine && bash finetune/run.sh
set -euo pipefail
cd "$(dirname "$0")"

BASE=${BASE:-huihui-ai/Huihui-Qwen3.5-9B-abliterated}
HOURS=${HOURS:-2}
DATA=${DATA:-data data-chains}
MAXLEN=${MAXLEN:-2560}

python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
.venv/bin/hf download "$BASE" --local-dir base
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/python train.py --base base --data $DATA --out out --max-hours "$HOURS" \
  --max-len "$MAXLEN" 2>&1 | tee out.log
tar czf adapter.tgz out/adapter out/summary.json out.log
echo "done: $(pwd)/adapter.tgz"
