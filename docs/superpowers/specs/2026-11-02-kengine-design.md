# kengine: Design

**Status:** Draft for review
**Date:** 2026-11-02

---

## 1. What this is and why

kengine is an agent harness built around **small local LLMs**. It treats the model's reasoning as a set of *claims to check*, not conclusions to act on.

Small models make mistakes in recognizable ways:

- They make up facts.
- They take invalid logical steps.
- They get arithmetic wrong.
- They forget constraints and contradict themselves over a long task.
- They act on assumptions they never stated or checked.

kengine wraps the model in a loop that does three things:

1. It pulls the model's reasoning out into explicit, typed claims.
2. It checks each claim with the cheapest reliable tool available: deterministic code, a solver, retrieval plus an entailment model, or a narrowly prompted critic.
3. It tracks what the agent believes and why. Actions and answers that depend on unverified assumptions are blocked or flagged.

The owner wants **two outcomes equally**:

- **Better answers.** A small model (Qwen3.5-9B) inside kengine should make fewer factual, logical and arithmetic errors than the same model on its own.
- **Auditability.** Every answer comes with an inspectable trail: what was assumed, what was verified and how, what was inferred from what, and what could not be confirmed.

**Core design principle.** Deterministic code owns the logic, the state and the bookkeeping. The model gets small, narrow, schema-constrained jobs. The harness never trusts the model to write solver code, to keep track of its own beliefs, or to do arithmetic.

---

## 2. Goals, non-goals, anti-goals

### Goals
- **G1.** A general-purpose agent loop, not tied to one domain, built from scratch (not a wrapper around another agent).
- **G2.** Every substantive step goes through claim extraction, checking and a belief-store update.
- **G3.** Facts are checked against evidence. Logic is checked by a solver. Arithmetic is computed, never trusted.
- **G4.** A belief store with dependency tracking. When a premise is disproved, everything derived from it is invalidated.
- **G5.** An action gate. Tool calls and final answers declare the claims they rely on, and the gate checks them against policy.
- **G6.** A complete audit trail for each session, rendered as a human-readable report.
- **G7.** Extensions that work like Claude Code's: **skills, MCP tools, subagents, hooks**. kengine uses the Claude Code formats wherever that's cheap, so existing skill folders, `.mcp.json` files and hook scripts carry over.
- **G8.** Runs well on the target hardware (section 3).
- **G9.** Its benefit is measurable: the same model with and without the harness, on fixed benchmarks.
- **G10.** The time budget is flexible per task (*decided by owner*: "depending on the task I can wait for 5 minutes or 5 days"). Effort is a parameter of each run (5.2), and long runs can be resumed.

**Intended uses** (*decided by owner*): research, coding, and general operations work such as managing a k3s cluster.

### Non-goals (v1)
- Running inside Claude Code, or pointing Claude Code at the local model. Claude Code's built-in instructions don't fit the context budget.
- Matching Claude Code feature for feature. Only skills, MCP, subagents and hooks are in scope.
- Lean 4 or other full theorem proving. The formal fragment (section 5.3) is deliberately small.
- Training or fine-tuning models.
- Multiple users or a hosted service. kengine is a local tool for one user.
- A GUI. A command-line interface (CLI) plus generated reports is enough.

### Anti-goals (*my call*: failure even if it technically works)
- **Verification theater.** Showing a claim as "verified" when no checker actually established it. When a checker can't decide, the result is `unchecked` or `unknown`, never a pass. This is the most important rule in the project.
- **Too many false alarms.** If the harness constantly rejects reasoning that was fine, it will get switched off. The false-alarm rate is a first-class metric.
- **Worse than the bare model.** If kengine scores below the bare model on the benchmarks, it has failed. Format overhead and over-correction are the usual causes.
- **Silent trust in outside content.** Tool results, subagent answers and skill content are all claims with a recorded source. None of them bypasses the checks.
- **Lock-in to one model or backend.** Model and backend sit behind an interface and are set in configuration.

---

## 3. Constraints

### Hardware and model (decided by owner)
- **GPU:** RTX 5060, 8 GB VRAM. **CPU:** Ryzen 7 7800X3D. **RAM:** 32 GB DDR5. **OS:** Linux.
- **Model:** Qwen3.5-9B, abliterated variant, run locally.
- **Model facts** (from the Qwen/Qwen3.5-9B model card):
  - Native context is 262,144 tokens.
  - It's a hybrid model: 24 Gated DeltaNet (linear-attention) layers and 8 full-attention layers, with 4 KV heads of dimension 256.
  - It thinks by default, emitting `<think>` blocks. Thinking can be turned off per request.
  - The model card lists recommended sampling settings per mode.

### What the hardware means (*my analysis; the builder should measure*)
- **Weights.** Weights at Q4_K_M should be about 5.5–6 GB. That leaves roughly 1.5–2 GB of VRAM for the KV cache and compute buffers, less if the GPU also drives a display.
- **KV cache.** Only the 8 full-attention layers store a per-token cache: 2 × 8 × 4 × 256 × 2 bytes ≈ **32 KiB per token at fp16**, or about 16 KiB at q8. The DeltaNet layers keep a fixed-size state.
- **Practical context.** That works out to roughly **32k tokens at fp16 or about 64k at q8**. Small models reason worse in long contexts anyway, so the design targets **per-step prompts of about 12–16k tokens**, configurable. The rest of the window is left for thinking and output.
- **One model at a time.** Only one LLM fits in VRAM. All model roles (thinker, extractor, critic, subagents) share it with different prompts and settings. Requests run mostly one after another.
- **CPU-side models.** The entailment and embedding models run on the CPU, so they never compete for VRAM.
- **Abliteration.** Removing refusals may cost some accuracy. The benchmark setup should make it easy to compare against the standard Qwen3.5-9B.

### Software (*my call*, delegated by owner: "anything you want" with explicit type safety)
- **Python 3.12+**, **pyright in strict mode** (enforced in CI), and **pydantic v2** for every data structure that crosses a boundary.
- **uv** for dependency and environment management. Setup on the agent machine is `git clone` + `uv sync`.
- The repo is `kengine`. It's developed here, pushed to GitHub and pulled onto the agent machine.

### Development vs. the real machine (*decided by owner*)
- **The development machine has no GPU.** The builder builds and tests everything against the fake backend (5.1). For end-to-end smoke tests, use a tiny Qwen3.5 model (e.g. 0.8B, or the smallest available) on the CPU through llama.cpp; that's *my call*.
- **A separate agent runs real-model work** on the RTX 5060 machine, working from a clone of this repo. The builder doesn't run the real model. The plan must therefore:
  - give every real-model step a self-contained, scripted entry point (`kengine doctor`, `kengine eval ...`) that the other agent can run with no other context
  - write results as files under `eval/results/<date>-<suite>-<config>/`: metrics JSON, error taxonomy, a sample of audit reports. The files come back through git.
  - include a short `docs/real-machine-runbook.md` explaining what to run at each checkpoint and what to commit back
- **The model file is configured, not fixed.** The exact abliterated GGUF is a configuration value, and `kengine doctor` reports what's loaded. If llama.cpp doesn't support the Qwen3.5 architecture, the fallback is standard Qwen3-8B, with the same pipeline and configuration change only (*my call*).

---

## 4. Architecture overview

```
          ┌──────────── compact context assembled from belief store ───────────┐
          ▼                                                                    │
   [THINK] free text (thinking mode)                                           │
          │                                                                    │
   [EXTRACT] schema-constrained claims (non-thinking mode, n samples, vote)    │
          │                                                                    │
          ├── structure check (deterministic)                                  │
          ├── arithmetic check (computed)                                      │
          ├── logic check (Z3 over the formal fragment)                        │
          ├── consistency check (solver + entailment model against beliefs)    │
          ├── fact check (retrieval + entailment model)                        │
          └── critic (narrow model prompts: soft inferences, omissions)        │
          │                                                                    │
   [UPDATE] belief store (truth maintenance) ─── short feedback ───────────────┘
          │
   [GATE] declared preconditions vs policy (built-in "before tool call" hook)
          │ pass                     │ fail → feedback, back to loop
          ▼
   [ACT] tool call / subagent / final answer → results become observation claims
```

### Components (suggested package layout; the builder may adjust)
```
src/kengine/
  backend/      model backend interface + llama.cpp server client
  loop/         the agent loop and step orchestration
  claims/       claim schema (IR), formal fragment, compilers
  beliefs/      belief store / truth maintenance system, persistence
  checkers/     Checker interface and each checker
  gate/         action gate + policy
  context/      per-step context assembly under a token budget
  retrieval/    local corpus index, optional web search
  ext/
    skills/     skill discovery, load on demand, kengine extensions
    mcp/        MCP client, deferred tool definitions, tool search
    subagents/  subagent definitions and runner
    hooks/      hook config, event bus, command-hook runner
  audit/        event log, report rendering
  eval/         benchmark harness, baselines, ablations
  cli/          command-line entry points
```

---

## 5. Component design

### 5.1 Model backend
- **Default backend: the `llama-server` HTTP server from llama.cpp.** It supports GGUF quantizations, constraining output to a JSON schema or a GBNF grammar, and running several requests at once.
- **Check before building:** the builder must confirm that the llama.cpp build in use supports the Qwen3.5 hybrid architecture.
- **Backend interface.** It covers chat completion, structured completion against a JSON schema or grammar, token counting and a thinking on/off switch. Ollama or vLLM can be added later behind the same interface.
- **Settings per role.** Sampling settings are configured per role, with defaults from the model card:
  - THINK uses thinking mode.
  - EXTRACT and the critics use non-thinking mode at low temperature.
- **Bounded thinking.** Thinking output must have a token limit. The builder picks the mechanism.
- **Fake backend for tests.** A scripted backend is required, so the loop, checkers and gate can be tested without a GPU.

### 5.2 The agent loop
Each step goes through these stages:

1. **Assemble context** (5.8).
2. **THINK.** The model reasons in free text. Strict output formats are known to hurt reasoning (Tam et al. 2024, "Let Me Speak Freely?"), so this stage has none.
3. **EXTRACT.** A separate, schema-constrained call turns the THINK text into claims plus a proposed next action.
   - It runs *n* times (set by the effort profile, below).
   - Claims are matched across samples. Claims and links that appear in fewer than *k* samples are dropped or flagged.
4. **CHECK.** Every applicable checker runs on each new claim (5.5).
5. **UPDATE.** The belief store is updated and statuses propagate (5.4).
6. **FEEDBACK.** If checks fail, the harness either fixes the problem itself, such as dropping an unsupported claim that nothing depends on, or prepares short, specific feedback for the next THINK.
7. **GATE.** The proposed action is checked (5.6).
8. **ACT.** The tool call, subagent or final answer is executed. Results become observation claims.

**Effort profiles.** These replace a single light/full switch (*my call*, serving G10). Each run picks a profile, and every setting in it can be overridden in configuration or on the command line.

| Profile | Extraction samples *n* | What gets checked | Critic | Typical use |
|---|---|---|---|---|
| `quick` | 1 | Only claims that support an action or the final answer | Off for inferences, on for hypotheses that support an action | Minutes |
| `standard` | 3 | Everything; fact checks on load-bearing facts only | On | Interactive work |
| `thorough` | 3 | Everything; every fact checked | On, including omission checks for every hypothesis | Hours |
| `exhaustive` | 5 | Everything, plus re-verification of `supported` claims with more retrieval | Repeated passes | Days, unattended |

- **Hard limits.** Every run also has optional limits on wall-clock time, tokens and steps.
- **Long runs.** These run unattended. Sessions persist after every step (5.4), so a run that's interrupted or stopped can be resumed with `kengine resume <session>`. Progress is visible through `kengine status <session>`.

**Stopping conditions:** a final answer passes the gate; or the step or token budget runs out; or the same check keeps failing repeatedly (loop detection). In the last two cases the agent stops with an honest partial result and a report.

### 5.3 Claim schema (the intermediate representation)
A claim is a pydantic model. Its fields:

- `id`
- `type`
- `text`: the claim in natural language
- `premises`: ids of the claims it rests on
- `formal`: optional formal content
- `computation`: optional
- `confidence`: the model's own estimate, for information only
- `source`: model step, tool, skill, subagent or user

**Claim types:**

| Type | Meaning | Main checkers |
|---|---|---|
| `observation` | Result returned by a tool or subagent | Source recorded; consistency |
| `given` | Supplied by the user, task or a skill's kengine files | Trusted within its scope; consistency |
| `fact` | Claim about the world | Fact check, consistency |
| `definition` | Stipulated meaning | Consistency |
| `assumption` | Explicitly assumed, not verified | Flagged; the gate decides |
| `computation` | Arithmetic or numeric result | Arithmetic checker (recomputed) |
| `inference` | Deductive step from premises | Logic checker (if formal), else critic |
| `hypothesis` | Abductive or inductive leap | Critic (alternatives); never shown as verified |
| `precondition` | What an action relies on | Gate |

**Formal fragment for v1** (*my call*). It's kept flat so a small model can fill it in reliably under constrained decoding:

- A **literal** is a predicate atom `pred(arg, ...)`, a negated atom, or a comparison between linear arithmetic terms.
- A **fact** is a ground literal (no variables).
- A **rule** is a conjunction of literals implying one literal. Its variables are implicitly universally quantified.
- **Checking:** the harness substitutes every constant that appears in the problem into the rules (grounding over a finite set). The result is a quantifier-free problem for Z3, which always terminates. Validity means premises ∧ ¬conclusion is unsatisfiable. If it's satisfiable, Z3's model is a counterexample.
- **Escape hatch.** Anything that doesn't fit has no `formal` field. It goes to the critic and can never become `verified`. At most it becomes `critic_accepted` (5.4).

How the fragment is serialized (nested JSON vs. a GBNF-constrained s-expression string) is left to the builder. The rule: pick whichever Qwen3.5-9B fills in more reliably on a small test set.

### 5.4 Belief store (truth maintenance system)
This follows the classic justification-based truth maintenance systems (Doyle 1979; de Kleer's ATMS 1986).

- **Contents.** Nodes are claims. Justifications record which premises support a claim and which checker verdicts back it.
- **Statuses** (*my call*; the important property is that they're honest):
  - `unchecked`: no checker has run.
  - `verified`: a deterministic checker (solver or arithmetic) passed, or the fact check found supporting evidence above the threshold.
  - `supported`: some evidence was found but the threshold wasn't met.
  - `critic_accepted`: only the model critic approved it. This is weaker than `verified`.
  - `assumed`: explicitly an assumption.
  - `contradicted`: a checker found counter-evidence or an inconsistency.
  - `retracted`: withdrawn by the agent or the harness.
  - `invalidated`: a premise it depends on became `contradicted` or `retracted`.
- **Propagation.** Retraction or contradiction spreads to every claim that depends on the claim. The audit log records every status change with its reason.
- **Persistence.** One SQLite database per session, plus an append-only JSONL event log. Sessions can be resumed and audited later.

### 5.5 Checkers
There's a common `Checker` interface: `(claim, belief_store_view) -> Verdict`. A verdict records the status, the evidence, a short explanation, the checker id and an optional counterexample.

1. **Structure** (deterministic): premises that don't exist, cyclic justifications, an inference with no premises, a conclusion that doesn't connect to anything.
2. **Arithmetic** (deterministic): the model writes the *expression*. The harness evaluates it with a safe evaluator (e.g. sympy, with no `eval`) and compares it to the result the model claimed.
3. **Logic** (Z3): validity of inferences with formal content, and satisfiability of the formal beliefs held together. Counterexamples are rendered back into plain language for feedback.
4. **Consistency** (entailment model): new claims are compared with existing verified or given beliefs. Embeddings pick the candidates to compare; an entailment cross-encoder scores contradiction. This catches self-contradiction across turns that has no formal content.
5. **Fact check** (retrieval + entailment model):
   - Retrieval searches a local corpus and optionally the web.
   - The verdict is `supported`, `contradicted`, `insufficient` or `conflicting`, with citations.
   - `insufficient` maps to `unchecked`. It's never a pass.
6. **Critic** (model-based, narrow prompts, run through the subagent mechanism):
   - (a) "Does B follow from A?" for inferences with no formal content.
   - (b) An omission or alternatives prompt for hypotheses: "What else could explain this?"
   - The critic can only reach `critic_accepted`, never `verified`.
7. **Execution evidence** (deterministic, *my call*; serves the coding and ops uses). Some claims can be checked by running something: "the tests pass", "this type-checks", "this manifest is valid", "this change only touches deployment X". These are checked by running the command and recording the output as an `observation`:
   - test runs and type checkers
   - `kubectl apply --dry-run=server` and `kubectl diff`
   - `helm template`
   
   A passing run is `verified` evidence for that claim. The gate can require such an observation as a precondition for changes (5.6).

**Defaults for CPU-side models** (*my call*; the builder may swap them): a DeBERTa-v3 cross-encoder trained on entailment (MNLI/FEVER/ANLI-class), and a small sentence-embedding model. Both run on the CPU with ONNX Runtime or PyTorch.

**Retrieval** (*my call*):
- **What's in the corpus.** There's no large general index (such as a Wikipedia dump) in v1. In normal use, evidence comes from:
  - (a) the files of the current project or working directory, indexed on demand and incrementally
  - (b) documents the user adds to a session or project
  - (c) tool observations already in the belief store
  - (d) **web search, which is in scope for v1** (the owner's original idea included an agent with internet access checking claims)
- **Index.** SQLite, with FTS5 full-text search for keyword matching plus a vector index, combined.
- **Web search.** Pluggable, with a self-hosted SearXNG instance as the default provider. Fetched pages are indexed into the session corpus with URL and fetch time recorded.
- **Benchmarks.** These use only the context paragraphs each task supplies (e.g. the HotpotQA distractor setting), so results are reproducible.
- **Trust.** Web content is untrusted evidence. It's never treated as instructions.

### 5.6 Action gate
- Every proposed action (tool call, subagent call, final answer) declares `precondition` claims that point at the beliefs it relies on.
- **The gate is a built-in "before tool call" hook** on the same event bus as user hooks (5.9).
- **Policy** is configured per tool, or per tool pattern, and for the final answer:
  - `strict`: every precondition must be `verified` or `given`.
  - `standard`: `supported` and `critic_accepted` are allowed. `assumed` is allowed but recorded as "acted on assumption X".
  - `permissive`: allow and record everything.
  - In every mode, a `contradicted` or `invalidated` precondition blocks the action.
- **Permissions.** Tools also carry allow/deny/ask rules, separate from the epistemic policy. Tools that can destroy or change things default to `ask`, which requires the user's confirmation.
- **Shell commands.** Permission rules can match on the command itself (e.g. allow `kubectl get *`, ask for `kubectl apply *`, deny `kubectl delete namespace *`).
- **Default for changing commands** (*my call*): commands that change things, such as `kubectl apply/delete/patch/scale` and `helm install/upgrade/uninstall`, require a verified dry-run or diff observation (5.5 #7) as a precondition, in addition to the permission rule.
- **Unattended runs.** In an `exhaustive` run with no one watching, an `ask` becomes a pause: the session waits for approval through `kengine resume`. It's never treated as approved.

### 5.7 Feedback
- Feedback is short and specific, with at most a few items per step. For example: *"c5 contradicts c2 (verified; source: docs/config.md). Revise c5 or explain."*
- Counterexamples are shown in plain language.
- The harness fixes things itself where safe, such as dropping an unsupported claim nothing depends on. It never silently changes a claim's content.

### 5.8 Context assembly
- The context for each step is rebuilt from the belief store; it's not a growing transcript. It contains:
  - the goal and constraints
  - the `given` and `verified` facts relevant to the current step
  - open assumptions
  - recent feedback
  - names and descriptions of loaded skills
  - the deferred tool index
  - the last few actions and observations
- **Relevance ranking** uses embeddings plus the dependency graph.
- **Token budget.** There's a configurable token budget with a fixed share for each section. The full transcript is kept in the audit log, not in the prompt.

### 5.9 Extensions (Claude Code-compatible where cheap)
**Rule** (*agreed with owner*): kengine accepts Claude Code formats; kengine-only features go in separate files so nothing breaks in either direction. **The builder should check the formats against Claude Code's current documentation and the published Agent Skills spec**, not against this document.

- **Skills**
  - **Discovery paths:** `.kengine/skills/`, `~/.kengine/skills/`, and read-only `.claude/skills/` and `~/.claude/skills/`.
  - **Format:** `SKILL.md` with `name`/`description` frontmatter, plus supporting files.
  - **Load on demand:** only the name and description sit in context. The full body loads when the agent picks the skill.
  - **kengine extension:** an optional `kengine/` subfolder in a skill can carry facts and rules in the formal fragment. They become `given` claims scoped to the active skill, so the logic checker knows that domain. Claude Code ignores the subfolder.

- **MCP tools**
  - A client built on the official Python `mcp` SDK, supporting stdio and streamable-HTTP servers.
  - Configuration in a `.mcp.json`-compatible file.
  - **Deferred definitions:** only a compact tool index sits in context. Full schemas load on demand through a built-in tool-search tool.
  - Tool results become `observation` claims with their source recorded: server, tool, arguments and time.

- **Subagents**
  - Definitions are markdown with frontmatter, compatible with `.claude/agents/*.md` (name, description, tools, system prompt), discovered under `.kengine/agents/` and read-only `.claude/agents/`.
  - A subagent is the same backend model with a fresh, small context and one narrow task. It runs its own kengine loop, with checks.
  - Its answer comes back as `observation` claims for the parent to verify.
  - The built-in critic and fact-check helpers use the same mechanism.
  - Subagents run one at a time by default. Parallel runs are configurable within the KV budget.

- **Hooks**
  - Configuration follows Claude Code's settings hook schema: event, then matcher, then command hooks.
  - The command receives event JSON on stdin and returns a decision through its exit code and optional JSON on stdout, following Claude Code's block/allow conventions.
  - **Events Claude Code also has:** `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Stop`, `SubagentStop`.
  - **kengine-only events:** `ClaimExtracted`, `CheckCompleted`, `BeliefChanged`, `ContradictionFound`, `GateDecision`.
  - Hooks can add evidence or block actions. They can't mark a claim `verified` unless they're configured as a trusted checker.

### 5.9a Built-in tools (*my call*, serving research, coding and ops)
- kengine ships a minimal built-in toolset, **using Claude Code's tool names and close-enough argument shapes**. That way existing Claude Code skills, hooks matchers and subagent `tools:` lists work unchanged:
  - `Read`, `Write`, `Edit`, `Glob`, `Grep`
  - `Bash`, with timeout and output capture
  - `WebFetch`, `WebSearch` (provider as in 5.5)
- Everything else comes through MCP.
- All built-ins go through the gate and permissions, and their outputs become `observation` claims.
- Exact parity with Claude Code's argument schemas is not required. The builder matches them where it's cheap and documents any differences.

### 5.10 Audit trail
- **Event log.** Every event (prompt, model call, extraction, verdict, status change, gate decision, action, hook outcome) is appended to the session's JSONL log.
- **Report** (`kengine audit <session>`, as Markdown and a self-contained HTML file). It shows:
  - the final answer
  - the claims the answer rests on, as a tree with their statuses and evidence
  - assumptions that were acted on
  - counterexamples found
  - contradictions and retractions
  - what remained unchecked
- **Export** of the belief graph to JSON.

### 5.11 Interface
CLI commands:
- `kengine run "<task>"`
- `kengine chat`
- `kengine audit <session>`
- `kengine eval <suite> [--size smoke|small|full] [--profile ...]`
- `kengine resume <session>`, `kengine status <session>`
- `kengine doctor`: checks the backend, model, VRAM and context settings, CPU-side models and configuration.

All commands take `--profile` (5.2) and optional time, token and step limits.

Configuration is a TOML file, project-level and user-level, validated by pydantic.

---

## 6. Evaluation
- **Step zero: baseline.** Run bare Qwen3.5-9B on the suites below before building any checkers. Sort its errors by type: made-up fact, invalid step, arithmetic, inconsistency, forgotten constraint. The results decide which checkers matter most.
- **Suites:**
  - FOLIO and ProofWriter/PrOntoQA for logic
  - MuSiQue or HotpotQA (distractor setting, supplied paragraphs only) for multi-hop facts
  - a GSM8K/MATH subset for arithmetic
  - **a custom tool-using set in three groups that mirror the intended uses** (*my call*). In every task, acting on a wrong assumption has visible, scorable consequences:
    - **research**: questions where a plausible wrong premise is tempting, with a fixed local document set
    - **coding**: small repos with failing tests or bugs, scored by hidden tests
    - **ops**: a throwaway local k3s/k3d cluster in a deliberately broken state, scored by the final cluster state and by whether destructive commands were attempted without a dry run
- **Sizes.** Every suite has `smoke` (minutes), `small` (about an hour) and `full` sizes with fixed, seeded subsets. Short checkpoints and multi-day runs use the same suites.
- **Scoring** (*my call*):
  - **Headline accuracy** counts gate-blocked answers, partial results and refusals to answer as **wrong**. This is the conservative reading of "worse than the bare model".
  - **Abstention rate** and **accuracy on answered tasks** are reported separately.
  - During evals, the final-answer gate uses the `standard` policy.
- **Baseline definition** (*my call*): the same model, quantization and backend, with thinking on (model-card sampling settings), the same total token budget per task, and no kengine checks. For tool tasks it gets a simple ReAct-style tool loop with the same built-in tools and permissions.
- **Metrics:**
  - task accuracy
  - errors caught (out of the baseline errors)
  - false-alarm rate (correct reasoning that was rejected)
  - model calls, tokens and wall-clock time per task
  - completeness of the audit trail
- **Comparisons:**
  - bare model vs. each effort profile
  - ablations with each checker turned off
  - abliterated vs. standard Qwen3.5-9B

---

## 7. Suggested milestone order (*my call*; the builder may re-sequence)
- **M0** Backend client, fake backend, eval harness, baseline run and error taxonomy.
- **M1** Claim schema, belief store with propagation, loop skeleton, structure and arithmetic checkers, audit event log.
- **M2** Formal fragment, Z3 logic checker, counterexample rendering.
- **M3** Retrieval, entailment model, fact and consistency checkers.
- **M4** Action gate and policy, built-in tools (5.9a), execution-evidence checker, MCP client with deferred tools.
- **M5** Skills (with the kengine extension), subagents, critic built on subagents, hooks.
- **M6** Audit report rendering, effort profiles, resume and status, a full evaluation pass.

**Owner checkpoints** (*my call*). Each milestone ends with a runbook step for the real-machine agent (section 3). There are two **owner review points**:
- after **M0**, on the baseline error taxonomy
- after **M2**, on the first measured with-vs-without comparison

**Scope rule.** The v1 scope is fixed. Baseline results change *order and defaults*, not scope. Dropping or deferring a checker requires the owner's sign-off at a review point.

---

## 8. Decision record

**Decided by the owner**
- Goals are both better answers and auditability, equally.
- General-purpose agents, with a loop built from scratch.
- For small local models that are prone to reasoning errors and unproved assumptions.
- Qwen3.5-9B abliterated on an RTX 5060 8 GB, 7800X3D and 32 GB.
- Extensions: skills, MCP tools, subagents, hooks.
- Any language with explicit type safety; the choice was delegated.
- Repo `kengine`, pushed to GitHub and pulled onto the agent machine.
- Intended uses: research, coding, general operations (e.g. managing a k3s cluster).
- The builder doesn't run the real model. Another agent runs it on the GPU machine from this repo.
- A flexible time budget per task, "5 minutes or 5 days".
- Web search is part of fact checking (from the owner's original idea: RAG and/or an internet-connected checking agent).

**Agreed during discussion** (proposed by Claude, approved by the owner)
- Fact check / logic check / computed arithmetic as separate checks.
- A belief store with dependency tracking.
- An action gate.
- Context rebuilt from the belief store.
- A baseline measurement first.
- Claude Code-compatible formats, with kengine extras in separate files.
- Skills may carry rules for the checker.
- Subagent results are verified, not trusted.
- The critic is built on subagents.
- Not running inside Claude Code.

**Claude's calls**
- Python, pyright strict, pydantic v2, uv.
- llama.cpp server as the default backend.
- The flat formal fragment with grounding, checked by Z3.
- Status vocabulary and gate policy levels.
- Entailment model and embeddings on the CPU.
- SQLite for storage and retrieval.
- TOML configuration.
- CLI-only interface.
- Anti-goals.
- Milestone order.
- No Lean in v1.
- Effort profiles `quick` / `standard` / `thorough` / `exhaustive`; resumable sessions.
- A built-in toolset with Claude Code tool names.
- The execution-evidence checker; dry run required before cluster changes.
- What's in the corpus (project files, user documents, observations, web); no large general index.
- Eval scoring (abstention counts as wrong in the headline), the baseline definition, the custom research/coding/ops task set, suite sizes.
- Fixed v1 scope with owner review points after M0 and M2.
- A tiny CPU model for development smoke tests; Qwen3-8B as the fallback if the architecture isn't supported.

## 9. Left to the builder
- Serialization of the formal fragment (nested JSON vs. grammar-constrained string). Measure which Qwen fills in more reliably.
- The mechanism for limiting thinking tokens.
- Specific entailment and embedding model checkpoints, and the runtime for them (ONNX vs. PyTorch).
- The algorithm for matching claims across extraction samples, and the voting threshold *k*.
- Default token budget shares for each context section.
- Default verification thresholds for fact checking.
- Internal module boundaries and naming.

## 10. Known risks
- **Extraction is the weakest link.** Agreement across samples reduces errors but doesn't remove them. The schema will need tuning against this specific model.
- **Latency.** A step can take 5–15 model calls run one after another. The `quick` profile and checking only load-bearing claims are the mitigations.
- **Formal coverage.** Much real reasoning won't fit the formal fragment. Most claims may end up checked by the fact checker or the critic, so the honest status vocabulary matters.
- **The critic shares the model's blind spots.** A later option is a second, smaller critic model on the CPU.
- **Hook and MCP security.** Hooks and MCP servers run arbitrary code. The permission rules and the `ask` default for tools that change things are the v1 safeguard; sandboxing comes later.
- **Shell access on real infrastructure.** A small model with `Bash` on a live k3s cluster is the highest-stakes use. Command-level permission rules, required dry runs and "pause, never auto-approve" when unattended are the v1 safeguards. Ops evals run only against throwaway clusters.
- **Eval cost.** `full` suites at the `thorough`/`exhaustive` profiles with every ablation may take days on one GPU. Suite sizes and the review points let the owner choose how much to run.

## 11. Prior art to borrow from
- **Logic-LM** (Pan et al. 2023), **LINC** (Olausson et al. 2023), **SatLM**, Faithful CoT: LLM-to-solver pipelines.
- **Chain-of-Verification**, **SAFE**, **FActScore**: breaking text into claims and checking them.
- **AWS Bedrock Automated Reasoning checks**: checking answers against policies with SMT.
- **Truth maintenance systems** (Doyle 1979; de Kleer 1986): the belief store.
- **"Let Me Speak Freely?"** (Tam et al. 2024): the reason for the two-pass think/extract design.
- **Agent Skills spec**, **MCP**, **Claude Code hooks and subagents**: extension formats.
