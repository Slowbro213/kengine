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

## Checkpoint B (kengine vs. baseline)

Run these inside `nix-shell` on NixOS, with the server from Setup step 4 running.

1. `git pull && uv sync --extra eval && uv run kengine doctor`.
2. Smoke-test the agent against the server: `KENGINE_TEST_LLM_URL=http://127.0.0.1:8080 uv run pytest -m llm -v`.
3. `uv run kengine run "A shop sells pens at \$1.25. How much do 12 pens cost?" --profile standard`. Record the output. Check that `beliefs.sqlite` and `events.jsonl` in the printed session directory look sensible: the computation claim should be `verified`, and the log should start with `session_started` and end with `run_finished`.
4. For each suite in gsm8k, folio, hotpotqa: `uv run kengine eval <suite> --solver kengine --size small --profile quick`, then `--profile standard`. Sessions go under each results directory's `sessions/`.
5. Write `eval/results/CHECKPOINT-B.md` with:
   - a table of headline accuracy, abstention rate and answered accuracy per suite and profile, next to the Checkpoint A baseline
   - mean wall-clock time per item
   - 3 example audit trails (one where kengine helped, one where it hurt, one blocked), as excerpts from `events.jsonl`
6. Commit everything under `eval/results/` except `sessions/`. Push only with the owner's go-ahead.

M1 has the structure and arithmetic checkers, plus the restatement checker described below. Under the `standard` gate, any answer that rests on an unchecked fact or inference is blocked. Expect a high abstention rate on folio and hotpotqa until the logic checker (M2) and evidence checks (M3) arrive. That's by design: the checkpoint should report it, not work around it.

**Task givens and the restatement checker** (added after Checkpoint A; owner left the design to the builder). In Checkpoint A's live `kengine run`, a correct, verified `1.25 * 12 = 15` was blocked. It rested on "each pen costs $1.25", which the model may only label `fact`, and nothing in M1 checks facts. The fix has three parts:

- The task's statements are seeded as `given` claims (source `task`) before the first step: gsm8k question sentences, folio premises (never the conclusion), and the sentences of a `kengine run` task. hotpotqa documents are evidence, not givens, so none are seeded there.
- The `restatement` checker (evidence kind) marks a numeric `fact` as `supported` when every number it states appears in what it cites, which must be givens or verified computations. An uncited fact can instead match a single given that has all its numbers and shares a word with it. Facts without numbers are untouched. Standard unit conversions (12 months in a year, 60 minutes in an hour, 100 cents in a dollar and so on) count as stated when the fact names one unit and the fact or what it cites names the other; this was added after the first Checkpoint B smoke runs, where "1 year is 12 months" blocked a correct answer.
- The prompt examples now end with the answer resting on the calculation, not on a restating inference, which nothing in M1 can check.

`supported` passes the `standard` gate but not `strict`. The check is lexical: a fact that cites the right given but uses a number from another of the givens it cites can pass. It's recorded in the verdict's evidence for audit.

## Checkpoint C (logic checker, owner review point 2)

Run these inside `nix-shell` on NixOS, with the server from Setup step 4 running. M2 adds the formal fragment, the z3 `logic` checker and formalization of the task's givens (`[logic]` in the config).

1. **Size the runs.** `uv run kengine eval folio --solver kengine --size smoke --profile quick`, then `--profile standard`. Let `t` be the two runs' `mean_wall_s` added together. Then `N = min(100, floor(36000 / t / 10) * 10)`, and at least 10: a 10-hour budget for step 2.
2. **Main runs.** `uv run kengine eval folio --solver kengine --size small --profile quick --limit N`, then `--profile standard`. Leave out `--limit` when N is 100. `--limit` takes the first N items of the seeded sample and adds `-firstN` to the results directory name.
3. **Arithmetic regression.** Formalization is on for every suite, so rerun `uv run kengine eval gsm8k --solver kengine --size smoke` under `quick` and `standard`. Compare each run with its Checkpoint B counterpart: `uv run kengine eval-compare <checkpoint B dir> <new dir>`.
4. **Compare with the baseline.** For each folio run, `uv run kengine eval-compare <baseline folio small dir> <run dir>`. The baseline dir is the Checkpoint A one (`ls eval/results`). The comparison covers only the items both runs have and is written to `<run dir>/compare-<base>.json`.
5. **Write `eval/results/CHECKPOINT-C.md`** with:
   - setup: commit, server command, the chosen formal syntax (from the Task 7 probe), N and why
   - the probe summaries for both syntaxes and the decision
   - a results table per run (n, headline, abstention, answered accuracy, mean wall s/item, and the baseline's accuracy on the same items)
   - with vs. without: fixed, broke, caught, false alarms, errors caught and false-alarm rate per profile
   - a count of `logic` verdicts by outcome and status over all folio sessions, with the script that counted them
   - 3 audit-trail excerpts from `events.jsonl`: one item kengine fixed, one it broke or falsely blocked, and one blocked by an informal inference
   - the gsm8k regression against Checkpoint B
   - insights for the owner: what M2 bought and cost on folio, whether formalization is reliable enough to rely on, what blocks most, and whether M3 or M5 should come next
6. Commit everything under `eval/results/` except `sessions/` and push.
