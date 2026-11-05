# Fine-tune plan: teach the model to translate into logic

**Status: for the owner to review.** Nothing here is trained or paid for until the owner approves it.

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

- **439 of 1001 examples convert, and z3 gets the gold label from them** (130 True, 110 False, 199 Uncertain). Only these are kept. The z3 check filters out the expert data's own mistakes, such as `Residentof` in one premise and `ResidentOf` in the next.
- The rest are dropped:

  | Reason | Examples |
  |---|---|
  | uses "=" | 148 |
  | the conclusion says "some", which our language can't state | 97 |
  | "some" inside "every" | 68 |
  | the converted forms give the wrong label | about 190 |
  | malformed | about 55 |

Each kept example becomes one training pair:

- **Input:** exactly the FORMALIZE prompt the agent sends (the same system prompt, with the premises and conclusion numbered).
- **Output:** the JSON answer in our format, using the expert's names in snake case.

10% of the stories are held back to pick the best checkpoint.

**Guarding non-folio tasks.** FORMALIZE also runs on gsm8k, where it should mostly translate nothing. About 100 gsm8k-train examples, paired with what the model answers today, are mixed in so the tuned model doesn't start translating arithmetic.

## How it's trained

- **Method:** a LoRA adapter, which trains a small add-on (about 100 MB) instead of the whole model. Base: `huihui-ai/Huihui-Qwen3.5-9B-abliterated`, the HF weights (19 GB) behind the GGUF we run. QLoRA, rank 16, 2–3 passes over the data.
- **Hardware:** a rented GPU with 24 GB or more (RTX 4090, A5000 or L4), for 1–2 hours. The local RTX 5060 has 8 GB, which isn't enough for a 9B model.
- **Afterwards:** merge the adapter and convert to Q4_K_M GGUF with the same llama.cpp. That can be done locally on the CPU, so the rented GPU is only used for training.

## What runs locally first (free)

1. The converter, in the repo with tests, and the dataset builder.
2. The training script, run end to end on the small Qwen3.5-0.8B. That checks the model family loads and trains, and that the output works with llama-server.
3. Merge and GGUF export, tested on the small model.
4. **A fair baseline.** The model we run now is an "i1" quantization made by someone else. The tuned model will be quantized by us, so we also quantize the untuned weights the same way, and compare the two.

## What the owner does

1. Pick a provider: RunPod or Vast.ai, both pay-as-you-go.
2. Create the account and add about $10 of credit. I never create accounts or spend money.
3. Either give me SSH access to the machine, or run one command that I'll send.

**Expected cost:** about $1–3. The budget is $10, and the run stops on its own after a set time.

## How we'll know it worked

Each check compares the tuned model against the same-quantized untuned one, with the same seed.

1. **probe-formal on all 203 validation items:** how often z3, given the model's translation alone, gets the gold label. Today it's 20 of 40 on the first 40.
2. **folio small, quick, 30 items, full kengine:** no wrong answer passes, as in M2.1, and the headline goes up.
3. **gsm8k smoke:** no drop.

**Stop rule:** if probe-formal doesn't gain at least 5 points, we keep the current model and don't start stage 2.

## Risks

- **439 examples is small.** The model may learn FOLIO's style and not much else. Check 3 and the gsm8k mix are there to catch that.
- **FOLIO's expert logic is imperfect.** The z3 filter removes examples that give the wrong label, but not wrong formulas that happen to give the right label.
- **The architecture may need a recent library.** Qwen3.5 may need a newer `transformers` or `peft`. The local run on the small model finds this out before anything is paid for.
- **Our own quantization may not match the i1 one.** That's why the baseline is quantized the same way.
