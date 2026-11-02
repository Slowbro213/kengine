# Real-machine runbook

What the GPU machine runs and commits (spec §3). Everything here was run on the reference machine: NixOS, RTX 5060 (8 GB), NVIDIA driver 570 (CUDA 12.8).

## Setup

1. Clone and install:

   ```bash
   git clone <repo> && cd kengine
   nix-shell            # NixOS only: uv, ruff, Python 3.12, libstdc++ for the numpy/datasets wheels
   uv sync --extra eval
   ```

   Off NixOS, skip `nix-shell` and install uv and ruff your usual way.

2. Build llama.cpp with CUDA. On NixOS, from nixpkgs (needs a llama.cpp new enough for the `qwen35` architecture; nixpkgs `llama-cpp` 0.8.0, build 11960, works):

   ```bash
   NIXPKGS_ALLOW_UNFREE=1 nix build --impure -o ~/.local/share/kengine/llama-cuda --expr '
     let pkgs = import (builtins.getFlake "nixpkgs") {
       system = "x86_64-linux";
       config = { allowUnfree = true; cudaSupport = true; cudaCapabilities = [ "12.0" ]; };
     };
     in pkgs.llama-cpp.override { cudaSupport = true; cudaPackages = pkgs.cudaPackages_12_8; }'
   ```

   - `cudaCapabilities` is the GPU's compute capability (`nvidia-smi --query-gpu=compute_cap --format=csv`); 12.0 is Blackwell consumer cards.
   - `cudaPackages` must not be newer than the driver supports (the `CUDA Version` in `nvidia-smi`). nixpkgs' default is CUDA 13, which driver 570 can't run: llama-server then logs `CUDA driver version is insufficient for CUDA runtime version` and silently runs on the CPU.
   - The `-o` link is a GC root, so `nix-collect-garbage` won't delete the build.

3. Get the model. The reference GGUF is the huihui-ai abliterated Qwen3.5-9B, imatrix Q4_K_M quant by mradermacher (5.63 GB):

   ```bash
   mkdir -p ~/.local/share/kengine/models && cd ~/.local/share/kengine/models
   curl -L -C - -O https://huggingface.co/mradermacher/Huihui-Qwen3.5-9B-abliterated-i1-GGUF/resolve/main/Huihui-Qwen3.5-9B-abliterated.i1-Q4_K_M.gguf
   ```

   For an unmodified comparison model, use the same quant from `bartowski/Qwen_Qwen3.5-9B-GGUF` or `unsloth/Qwen3.5-9B-GGUF`.

4. Start the server:

   ```bash
   ~/.local/share/kengine/llama-cuda/bin/llama-server \
     -m ~/.local/share/kengine/models/Huihui-Qwen3.5-9B-abliterated.i1-Q4_K_M.gguf \
     -ngl 99 -c 32768 -np 1 -fa on -ctk q8_0 -ctv q8_0 \
     --reasoning-format deepseek --host 127.0.0.1 --port 8080
   ```

   - On the reference machine this uses 6.4 of 8 GB VRAM and generates about 67 tokens/s.
   - `-np 1` gives the one slot the whole 32k context; the eval runner sends one request at a time.
   - `--reasoning-format deepseek` returns the thinking in `reasoning_content`, which the backend reads. `--jinja` is the default in this version.
   - Check that the server really is on the GPU: `nvidia-smi` should show about 6 GB in use, and the log must not say `failed to initialize CUDA`.
   - Flags change between llama.cpp versions. Check them against `llama-server --help`, and record the exact command in the checkpoint summary.
   - If llama.cpp refuses the Qwen3.5 architecture, use standard Qwen3-8B (spec §3), change only the config, and record it.

5. Write `kengine.toml` in the repo root only if the defaults (`kengine.example.toml`) need changing. It's gitignored.

## Checkpoint A (owner review point 1)

Run these inside `nix-shell` on NixOS.

1. `uv run kengine doctor --json eval/results/doctor-<date>.json`. It must report `ok`.
2. For each suite in gsm8k, folio, hotpotqa: `uv run kengine eval <suite> --solver baseline --size smoke`, then `--size small`. Runs resume: rerunning the same command skips items already in `items.jsonl`.
3. For each `failures_sample.jsonl`, write `taxonomy.jsonl` using `docs/eval/error-taxonomy.md`, then run `uv run kengine eval-summarize <dir>`.
4. Commit everything under `eval/results/` except `sessions/` (gitignored), plus `eval/results/CHECKPOINT-A.md`: headline accuracy per suite, top 3 error labels, the server command and the GGUF file name. Push only with the owner's go-ahead.
