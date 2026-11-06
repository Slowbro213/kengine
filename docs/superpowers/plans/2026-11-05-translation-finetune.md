# Fine-tune plan: teach the model to translate into logic

**Status: approved by the owner on 2026-11-05, and run on RunPod on 2026-11-06.** Translation improved a lot (probe-formal: 134 right of 203, against 84), but the full agent barely moved. See "What the RunPod run found".

The spec lists fine-tuning as a v1 non-goal. The owner asked to consider it once Checkpoint C showed where the errors come from, so this plan would change that.

## Why translation first

The logic checker is only as good as the translation behind it.

- **The checker is reliable.** In M2.1 every answer it let through was right: 10 of 10 on folio and 16 of 16 on gsm8k.
- **Bad translations break it in both directions.**
  - They block right answers, such as 289's conclusion.
  - The Uncertain check would wrongly pass bad answers. Letting the checker answer Uncertain itself would be right only 16 times in 27, and every miss traced back to a translation error. Asking the model to spot those translations didn't work either: it flagged all 27.
- **The model's own reasoning isn't better than the bare model** (13–15 right out of 30, against 16). So a reasoning fine-tune is worth doing too. But it would teach the model to say Uncertain more often, and that's only safe once the translations are good. Reasoning is stage 2, with its own approval.

"Translation" means the FORMALIZE call. It turns the premises and the conclusion into our formal language before the first step.

## The data

FOLIO's training split has 1001 examples over 340 stories. Each premise and conclusion comes with an expert's logic formula. **No story is shared with the 203-item validation split that every eval uses**, so the eval stays clean.

A converter turns the expert formulas into our language. A first version is written and tested on the training split. It does the standard steps: push negations inward, name the "some" individual, then split into rules.

- **393 of 1001 examples convert cleanly, and z3 gets the gold label from them** (123 True, 98 False, 172 Uncertain). Only these are kept. The z3 check filters out the expert data's own mistakes. The converter also drops two other kinds of expert mistake:
  - a statement that says something of every individual (nearly always a lost "if" part)
  - "some … if …", which says almost nothing
- The rest are dropped (`finetune/data/report.json` has the full list):

  | Reason | Examples |
  |---|---|
  | uses "=" | 148 |
  | the converted forms give the wrong label | 153 |
  | the conclusion says "some", which our language can't state | 87 |
  | says something of everything | 69 |
  | "some" with an "if" part | 53 |
  | "some" inside "every" | 43 |
  | malformed | 36 |
  | other | 19 |

Each kept example becomes one training pair:

- **Input:** exactly the FORMALIZE prompt the agent sends (the same system prompt, with the premises and conclusion numbered).
- **Output:** the JSON answer in our format, using the expert's names in snake case.

10% of the stories are held back to pick the best checkpoint: 350 pairs train and 43 are held back.

**Guarding non-folio tasks.** FORMALIZE also runs on gsm8k, where it should mostly translate nothing. 100 gsm8k-train examples, paired with what the model answers today, are mixed in so the tuned model doesn't start translating arithmetic. The model translated 195 of their 356 sentences; the pairs keep those as it wrote them, and leave the rest empty.

The data is built by `kengine finetune-data` and committed in `finetune/data/`. The gsm8k half comes from the model's answers, so a rerun wouldn't give exactly the same pairs.

## How it's trained

- **Method:** a LoRA adapter, which trains a small add-on (about 100 MB) instead of the whole model. Base: `huihui-ai/Huihui-Qwen3.5-9B-abliterated`, the HF weights (19 GB) behind the GGUF we run. QLoRA, rank 16, 2–3 passes over the data.
- **Hardware:** a rented GPU with 24 GB or more (RTX 4090, A5000 or L4), for 1–2 hours. The local RTX 5060 has 8 GB, which isn't enough for a 9B model.
- **Afterwards:** merge the adapter and convert to Q4_K_M GGUF with the same llama.cpp. That can be done locally on the CPU, so the rented GPU is only used for training.

## What runs locally first (free)

1. The converter, in the repo with tests, and the dataset builder.
2. The training script, run end to end on the small Qwen3.5-0.8B. That checks the model family loads and trains, and that the output works with llama-server.
3. Merge and GGUF export, tested on the small model.
4. **A fair baseline.** The model we run now is an "i1" quantization made by someone else. The tuned model will be quantized by us, so we also quantize the untuned weights the same way, and compare the two.

## What the local test found (2026-11-05)

All four local steps ran on the home machine, for free.

- **Training works on this model family.** The libraries are pinned in `finetune/requirements.txt` (torch 2.11, transformers 5.22, peft 0.21, flash-linear-attention 0.6). Qwen3.5-0.8B trained for one pass in 2.4 minutes, using 5.3 GB of GPU memory at peak. The held-out loss fell from 0.44 to 0.23.
- **The merge works.** `finetune/merge.py` matches peft's own merge to within one bf16 rounding step. It loads one shard at a time, at most 5.4 GB for the 9B, so the merge fits in this machine's RAM. It also keeps the model's multi-token-prediction weights, which peft's merge drops.
- **The export works.** The merged model converts with llama.cpp's `convert_hf_to_gguf.py` (at the same commit as our llama-server), then loads and answers in llama-server.
- **Our own quantization is a fair baseline.** Converted and quantized to Q4_K_M by us, the untuned 9B scores the same as the i1 file we run.

probe-formal on all 203 validation items (how often z3, given the model's translation alone, gets the gold label):

| Model | Right | Right, True or False | Wrong, True or False |
|---|---|---|---|
| 0.8B, untuned | 15 | 4 | 2 |
| 0.8B, one pass of training | **104** | 47 | 8 |
| 9B, the i1 file we run | 86 | 35 | 10 |
| 9B, our own Q4_K_M (the baseline) | 84 | 29 | 10 |

The tiny model, after two minutes of training, beats the untuned 9B. The gain isn't just more Uncertain answers: its definite True and False answers went from 4 right to 47. This is the first sign that the plan works, before anything is paid for.

Two caveats:
- The tiny model gains partly by learning FOLIO's naming style, which the validation split shares. Checks 2 and 3 below test whether the full agent gains too.
- Its most common miss is still Uncertain where the gold label is True or False: 63 times, the same count as the untuned 9B. That means a missing link between the statements. Most of the gain comes from fewer translations that fail outright: 28 items got no answer, against 46 for the 9B.

The probe results are in `eval/results/finetune-local/`.

**On the rented GPU** only training runs (`finetune/run.sh`). The adapter comes back home, where it is merged, converted and quantized the same way as the baseline.

## What the RunPod run found (2026-11-06)

In the end everything ran on the pod: training, merge, GGUF export and all three checks. That was one RunPod A40 (48 GB, secure cloud, $0.59/h) for about 3.5 hours, $2.04 in total. Training took 0.49 h, and the best checkpoint came after the first pass. The trained model is kept at home as `~/.local/share/kengine/models/ours-9B-ft1-Q4_K_M.gguf`, and the adapter is in `~/.local/share/kengine/ft/run1/`.

**Check 1 passes easily.** probe-formal on all 203 items, with both models quantized the same way on the pod:

| Model | Right | Right, True or False | Wrong, True or False | No answer |
|---|---|---|---|---|
| 9B, untuned (pod) | 85 | – | – | – |
| 9B, untuned (home baseline) | 84 | 29 | 10 | 46 |
| **9B, tuned** | **134** | **70** | **3** | **10** |

The stop rule asked for 94. The most common miss is still Uncertain where the gold label is True or False (56 times, against 63).

**Checks 2 and 3: the agent as a whole hasn't gained yet.**

| Run | folio, 30 items | gsm8k smoke, 10 items |
|---|---|---|
| i1 model at home (m2.1-fixes) | 4 right, **1 wrong**, 25 blocked or partial | 10 right |
| **Tuned (pod)** | **5 right, 0 wrong**, 25 blocked or partial | 8 right, 0 wrong, 2 blocked |
| Untuned (pod), stopped partway | 3 right of 15 done, 0 wrong | 6 right of 9 done, 0 wrong |

On the 15 folio items the untuned pod run finished, the tuned model got 2 right, the untuned 3, and the home run 4. On the 9 shared gsm8k items, the tuned model got 8 and the untuned pod run 6. So the pod setup costs a little on its own: 8 parallel streams per model at about 20 tokens/s each, and a 30-minute reply timeout. **The tuned model is no worse than the untuned one end to end, but not clearly better.** No wrong answer got through in any pod run.

**Why the better translation doesn't show up yet.** The agent is blocked at the reasoning step, not the translation.

- In 14 of the tuned model's 19 blocked folio runs, the answer it was trying to give was right. The gate refused it because a step didn't follow ("b79 doesn't follow from b72, b54") or because the cited steps didn't show the conclusion.
- On the same 30 items, z3 working from the tuned translation alone gets 24 right: 8 True or False and 16 Uncertain. It gets 5 Uncertain wrong and gives no definite wrong answer.

**Next (each needs the owner's approval):**

1. A free change at home: let a definite True or False from the translation alone count as an answer. Only accept it when a second check agrees, either the model's own answer or a second translation. Test it first on items outside the eval set: the 43 held-back stories plus a fresh FOLIO slice. Keep it only if it gives no wrong answers there.
2. Stage 2: train the reasoning step. The data would be checker-valid step chains, generated from FOLIO-train's expert logic. Build and test it on the 0.8B at home first, then use a short pod (about $2).

**Lessons for the next pod:**

- The CPU quota is about 7.6 cores, not the 96 `nproc` reports.
- RunPod's nginx holds port 8081.
- Base THINK calls can exceed a 600 s timeout when 16 streams share one GPU.
- Results, logs and the probe outputs are in `eval/results/finetune-runpod/`.

## What the owner does

1. Create a RunPod account and add about $10 of credit. I never create accounts or spend money.
2. Start a pod with a 24 GB GPU (RTX 4090 or A5000) and a PyTorch template, with at least 50 GB of disk.
3. Either give me SSH access to the pod, or run this in its terminal:

   ```bash
   git clone https://github.com/Slowbro213/kengine && cd kengine && bash finetune/run.sh
   ```

4. When it prints `done`, download `finetune/adapter.tgz` (tens of MB) and terminate the pod. A stopped pod still costs a little for its disk; a terminated one costs nothing.

**Expected cost:** about $1–3. The budget is $10, and the run stops on its own after a set time.

## How we'll know it worked

Each check compares the tuned model against the same-quantized untuned one, with the same seed.

1. **probe-formal on all 203 validation items:** how often z3, given the model's translation alone, gets the gold label. The baseline is 84 of 203 (41%).
2. **folio small, quick, 30 items, full kengine:** no wrong answer passes, as in M2.1, and the headline goes up.
3. **gsm8k smoke:** no drop.

**Stop rule:** if probe-formal doesn't gain at least 5 points (to 94 of 203), we keep the current model and don't start stage 2.

## Risks

- **393 examples is small.** The model may learn FOLIO's style and not much else. Check 3 and the gsm8k mix are there to catch that.
- **FOLIO's expert logic is imperfect.** The z3 filter removes examples that give the wrong label, but not wrong formulas that happen to give the right label.
- **The 9B hasn't been trained anywhere yet.** The 0.8B run checks the code and the libraries, but not the 9B's memory use on 24 GB. If it runs out of memory, `--max-len 1536` or rank 8 makes it fit. The pod is billed by the minute, so a failed start costs cents.
