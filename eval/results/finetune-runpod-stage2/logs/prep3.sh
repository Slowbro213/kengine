set -euxo pipefail
export PATH=/usr/local/cuda/bin:$PATH
cd /root/llama.cpp
rm -rf build
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /root/cmake.log 2>&1
# 3 cores while training runs, then 8.
setsid cmake --build build -j 3 --target llama-server llama-quantize > /root/build.log 2>&1 &
b=$!
while kill -0 $b 2>/dev/null && ps -eo args | grep -q "^bash /root/x.sh"; do sleep 10; done
if kill -0 $b 2>/dev/null; then kill -- -$b || true; sleep 5; fi
cmake --build build -j 8 --target llama-server llama-quantize >> /root/build.log 2>&1
ls -la build/bin/llama-server build/bin/llama-quantize
[ -x /root/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
cd /root/kengine && /root/.local/bin/uv sync -q --extra eval && /root/.local/bin/uv run kengine --help | head -3
echo PREP-OK
