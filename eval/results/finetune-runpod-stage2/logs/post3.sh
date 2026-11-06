set -euxo pipefail
F=/root/kengine/finetune; L=/root/llama.cpp; W=/root/work; PY=$F/.venv/bin/python; UV=/root/.local/bin/uv
until ! ps -eo args | grep -q "^bash /root/prep.sh"; do sleep 10; done
grep -q PREP-OK /root/prep.log
cd /root/kengine && git pull -q && $UV sync -q --extra eval && git log --oneline -1
$PY $F/merge.py --base $F/base --adapter $F/out/adapter --out $W/ft2-hf
PYTHONPATH=$L/gguf-py $PY $L/convert_hf_to_gguf.py $W/ft2-hf --outtype bf16 --outfile $W/ft2-bf16.gguf > $W/convert-ft2.log 2>&1
rm -rf $W/ft2-hf
$L/build/bin/llama-quantize $W/ft2-bf16.gguf $W/ft2-Q4_K_M.gguf Q4_K_M > $W/quant-ft2.log 2>&1
rm $W/ft2-bf16.gguf
ls -la $W/*.gguf
port=18081
for n in ft1 ft2; do
  np=$([ $n = ft1 ] && echo 12 || echo 21)
  setsid nohup $L/build/bin/llama-server -m $W/$n-Q4_K_M.gguf -ngl 99 -c $((np * 32768)) -np $np -fa on -ctk q8_0 -ctv q8_0 --reasoning-format deepseek --host 127.0.0.1 --port $port --spec-type ngram-mod > $W/server-$n.log 2>&1 < /dev/null &
  mkdir -p $W/evals-$n && printf '[backend]\nbase_url = "http://127.0.0.1:%s"\n' $port > $W/evals-$n/kengine.toml
  port=$((port+1))
done
for p in 18081 18082; do timeout 300 bash -c "until curl -sf http://127.0.0.1:$p/health >/dev/null; do sleep 2; done"; done
nvidia-smi --query-gpu=memory.used --format=csv
ev() { (cd $W/evals-$1 && $UV run --project /root/kengine kengine "${@:2}" --out .); }
pids=""
# ft1: only the held-back stories (its other checks ran in stage 1), 12 at a time.
for k in $(seq 12); do ev ft1 eval folio --solver kengine --profile quick --split train-dev --shard $k/12 > $W/evals-ft1/dev-$k.log 2>&1 & pids="$pids $!"; done
# ft2: every check at once, 21 at a time.
for k in $(seq 6); do ev ft2 eval folio --solver kengine --size small --profile quick --limit 30 --shard $k/6 > $W/evals-ft2/folio-$k.log 2>&1 & pids="$pids $!"; done
for k in 1 2; do ev ft2 eval gsm8k --solver kengine --size smoke --profile quick --shard $k/2 > $W/evals-ft2/gsm8k-$k.log 2>&1 & pids="$pids $!"; done
ev ft2 probe-formal --size full > $W/evals-ft2/probe.log 2>&1 & pids="$pids $!"
for k in $(seq 12); do ev ft2 eval folio --solver kengine --profile quick --split train-dev --shard $k/12 > $W/evals-ft2/dev-$k.log 2>&1 & pids="$pids $!"; done
fail=0; for p in $pids; do wait $p || fail=1; done
for n in ft1 ft2; do
  for d in $W/evals-$n/2026-*; do [ -d "$d" ] && ls $d | grep -q '^shard-' && (cd $W/evals-$n && $UV run --project /root/kengine kengine eval-merge $d | tail -1 | sed "s#^#$n $(basename $d): #"); done
done
tail -1 $W/evals-ft2/probe.log
echo "EVALS-DONE fail=$fail"
