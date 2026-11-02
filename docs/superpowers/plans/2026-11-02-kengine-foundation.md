# kengine Foundation (M0 + M1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working kengine core: a model backend, an evaluation harness with a bare-model baseline, and an agent loop that extracts claims, votes across samples, keeps them in a truth-maintenance belief store, checks them for structure and arithmetic, and gates the final answer. It must run end to end against the real model on the GPU machine.

**Architecture:**
- Deterministic Python owns all state: the belief store in SQLite and a JSONL event log for each session.
- The model is reached only through a `ModelBackend` protocol. There are two implementations: one for the `llama-server` HTTP server and one scripted fake.
- Each agent step goes THINK (free text) → EXTRACT (n schema-constrained samples) → VOTE → INSERT → CHECK → GATE → FEEDBACK.
- The eval harness runs a `Solver`, either the bare-model baseline or kengine, over fixed seeded benchmark subsets. It writes results to `eval/results/`, which the real-machine agent commits back.

**Tech Stack:** Python 3.12, uv, pyright (strict), ruff, pytest, pydantic v2, httpx, typer, sqlite3 (stdlib), `datasets` (eval extra only).

**Spec:** `docs/superpowers/specs/2026-11-02-kengine-design.md`. Read it first. Section numbers below (§) refer to it.

**Scope:**
- This plan covers milestones M0 and M1 (§7).
- It deliberately leaves out these spec parts, each covered by a later plan:
  - the formal fragment and the Z3 logic checker (M2)
  - retrieval, the fact checker and the consistency checker (M3)
  - tools, MCP and the execution-evidence checker (M4)
  - skills, subagents, the critic and hooks (M5)
  - the audit report, `resume` and `status`, and the `thorough`/`exhaustive` critic behaviour (M6)
- The final-answer gate is pulled forward from M4 because eval scoring needs it. In this plan it's called directly, not through hooks.

## Global Constraints

- Python `>=3.12`.
- `pyright` in strict mode reports **0 errors** on `src/` and `tests/`. `ruff check` is clean. CI enforces both (§3 Software).
- Every data structure that crosses a module, process or model boundary is a pydantic v2 model (§3).
- Dependencies are managed with `uv`. Setup is `git clone` + `uv sync` (with `--extra eval` for benchmarks).
- **No status of `verified` without a deterministic or evidence checker. Anything undecidable stays `unchecked`, or gets the `unknown` outcome. It never passes** (§2 anti-goal #1). This is enforced in code (Task 9).
- All model access goes through `ModelBackend`. Unit tests never need network access, a GPU or a real model.
- Never call `eval`/`exec` on text produced by the model (§5.5 #2).
- Headline accuracy counts gate-blocked, partial, refused and error results as **wrong**. The abstention rate and accuracy on answered items are reported separately (§6).
- Results go to `eval/results/<YYYY-MM-DD>-<suite>-<solver>-<size>-<profile>/` (§3).
- The token budget counts **completion (generated) tokens only**, for both the baseline and kengine. That's *a call this plan makes*: prompt tokens get re-sent every step, so counting them would make the comparison meaningless.
- Model roles and sampling defaults come from the Qwen3.5-9B model card (§3, §5.1). THINK uses thinking mode; EXTRACT uses non-thinking mode at low temperature.

## Review Focus

These are the input classes most likely to bite a real user that no feature test would naturally hit. Each has a pinning test in the task named.

1. **Truncated or malformed extraction JSON** despite the schema constraint (`finish_reason == "length"`, missing fields). That sample is skipped. If every sample fails, the step produces feedback and the loop continues. It never crashes. *Tests in Task 3 and Task 15.*
2. **THINK hits the token limit with empty content** (thinking ran away). The step records a failure, gives "be more concise" feedback and moves on, with no extraction attempt. *Test in Task 15.*
3. **The backend fails partway through a run or an eval** (connection refused, timeout, HTTP 500). The agent retries once, then stops with status `error` and the session on disk intact. The eval runner records the item as `error` and carries on. *Tests in Task 6 and Task 15.*
4. **Pathological arithmetic** (`9**9**9`, a 100k-character expression, `1/0`). The verdict is `unknown` within 1 s, with no hang and no crash. *Test in Task 11.*
5. **Real-world number formats** in answers and claims: `1,234`, `$30.00`, `−5` (Unicode minus), `15%`, `**42**`, `42.`. These are normalized consistently for scoring and the arithmetic check. *Tests in Task 4 and Task 11.*

---

## File Structure

```
pyproject.toml                         uv project, pyright/ruff/pytest config, `kengine` script
.github/workflows/ci.yml               ruff + pyright + pytest
.gitignore
kengine.example.toml                   documented example config
docs/real-machine-runbook.md           what the GPU-machine agent runs and commits (§3)
docs/eval/error-taxonomy.md            labelling rubric for baseline failures (§6 step zero)
src/kengine/
  __init__.py
  config.py                            KengineConfig + TOML loading
  numbers.py                           shared number parsing/normalization
  backend/
    __init__.py                        make_backend()
    types.py                           Message, SamplingParams, RoleConfig, Completion, errors, ModelBackend
    fake.py                            ScriptedBackend
    llamacpp.py                        LlamaCppBackend
  eval/
    __init__.py
    types.py                           EvalItem, SolveResult, ItemScore, Solver, RunMetrics
    answers.py                         extract_final_answer, normalize_text
    scoring.py                         score_numeric, score_em_f1, score_label
    suites.py                          SuiteSpec registry + load_items
    adapters/gsm8k.py, hotpotqa.py, folio.py
    runner.py                          run_eval, results_dir_name
    baseline.py                        BaselineSolver
    kengine_solver.py                  KengineSolver
    taxonomy.py                        ErrorLabel, summarize_taxonomy
  claims/
    schema.py                          ClaimType, Source, Computation, Claim
    extraction.py                      ExtractedClaim, ProposedAction, Extraction
    voting.py                          vote, VoteResult
  beliefs/
    status.py                          Status
    records.py                         BeliefRecord (kept apart to avoid an import cycle)
    store.py                           BeliefStore, StatusChange
  checkers/
    base.py                            CheckerKind, Outcome, Verdict, Checker, BeliefView
    structure.py                       StructureChecker
    arith_eval.py                      safe_eval
    arithmetic.py                      ArithmeticChecker
  audit/
    events.py                          Event union, EventLog
    session.py                         Session
  loop/
    task.py                            Task
    profiles.py                        Profile, PROFILES
    gate.py                            GatePolicy, GateDecision, evaluate_gate
    context.py                         assemble_context, StepContext
    feedback.py                        build_feedback
    agent.py                           Agent, RunResult, Limits
  prompts/think.md, extract.md, baseline.md
  cli/
    __init__.py
    main.py                            typer app: doctor, eval, eval-summarize, run
    doctor.py                          run_doctor, DoctorReport
tests/                                 mirrors src/ (tests/backend/test_fake.py, ...)
tests/integration/test_llm_smoke.py    opt-in, real llama-server
```

---

## Task 1: Project scaffold, toolchain and config

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.github/workflows/ci.yml`, `kengine.example.toml`, `src/kengine/__init__.py`, `src/kengine/config.py`, `src/kengine/backend/__init__.py` (empty for now), `src/kengine/backend/types.py` (only `SamplingParams` and `RoleConfig` in this task)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `SamplingParams(temperature: float, top_p: float, top_k: int, min_p: float, presence_penalty: float, repetition_penalty: float = 1.0)`
  - `RoleConfig(sampling: SamplingParams, thinking: bool, max_tokens: int)`
  - `BackendConfig(kind: Literal["llamacpp"] = "llamacpp", base_url: str = "http://127.0.0.1:8080", model: str = "qwen3.5-9b", timeout_s: float = 600.0)`
  - `RolesConfig(think: RoleConfig, extract: RoleConfig, critic: RoleConfig, baseline: RoleConfig)`
  - `ContextConfig(step_prompt_tokens: int = 14000)`
  - `EvalConfig(token_budget_per_task: int = 32768)`
  - `KengineConfig(backend, roles, context, eval, data_dir: Path, default_profile: str = "standard")`
  - `load_config(paths: Sequence[Path]) -> KengineConfig`
  - `default_config_paths() -> list[Path]`

- [ ] **Step 1: Create `pyproject.toml`**

  - `[project]`: name `kengine`, `requires-python = ">=3.12"`, dependencies `pydantic>=2.7`, `httpx>=0.27`, `typer>=0.12`.
  - `[project.optional-dependencies] eval = ["datasets>=2.19"]`.
  - `[dependency-groups] dev = ["pytest>=8", "pyright>=1.1.380", "ruff>=0.6"]`.
  - `[project.scripts] kengine = "kengine.cli.main:app"`.
  - Build backend `hatchling` with `src/` layout.
  - `[tool.pyright] typeCheckingMode = "strict"`, `include = ["src", "tests"]`, `pythonVersion = "3.12"`.
  - `[tool.ruff] line-length = 100`, `lint.select = ["E","F","I","UP","B"]`.
  - `[tool.pytest.ini_options] markers = ["llm: needs a real llama-server (KENGINE_TEST_LLM_URL)"]`, `addopts = "-m 'not llm'"`.

  `.gitignore` covers `.venv/`, `__pycache__/`, `*.sqlite`, `eval/results/*/sessions/`.

  `ci.yml` runs on push and PR, ubuntu-latest:
  - `astral-sh/setup-uv`
  - `uv sync`
  - `uv run ruff check`
  - `uv run pyright`
  - `uv run pytest`

- [ ] **Step 2: Write the failing tests** (`tests/test_config.py`)

```python
def test_defaults_match_model_card():
    cfg = load_config([])
    assert cfg.backend.base_url == "http://127.0.0.1:8080"
    t = cfg.roles.think
    assert (t.thinking, t.sampling.temperature, t.sampling.top_p, t.sampling.top_k,
            t.sampling.min_p, t.sampling.presence_penalty) == (True, 1.0, 0.95, 20, 0.0, 1.5)
    e = cfg.roles.extract
    assert e.thinking is False and e.sampling.temperature == 0.3 and e.sampling.presence_penalty == 0.0
    assert cfg.roles.baseline == cfg.roles.think.model_copy(update={"max_tokens": cfg.eval.token_budget_per_task})
    assert cfg.context.step_prompt_tokens == 14000
    assert cfg.eval.token_budget_per_task == 32768
    assert cfg.default_profile == "standard"

def test_later_files_override_earlier(tmp_path):
    a = tmp_path / "a.toml"; a.write_text('[backend]\nbase_url = "http://a:1"\nmodel = "m-a"\n')
    b = tmp_path / "b.toml"; b.write_text('[backend]\nmodel = "m-b"\n[roles.think]\nmax_tokens = 999\n')
    cfg = load_config([a, b])
    assert cfg.backend.base_url == "http://a:1"      # deep merge keeps a's key
    assert cfg.backend.model == "m-b"
    assert cfg.roles.think.max_tokens == 999
    assert cfg.roles.think.sampling.top_k == 20       # untouched nested default survives

def test_missing_files_are_skipped(tmp_path):
    assert load_config([tmp_path / "nope.toml"]) == load_config([])

def test_invalid_value_rejected(tmp_path):
    p = tmp_path / "c.toml"; p.write_text('[context]\nstep_prompt_tokens = "lots"\n')
    with pytest.raises(pydantic.ValidationError):
        load_config([p])
```

- [ ] **Step 3: Run them and confirm they fail**

Run: `uv sync && uv run pytest tests/test_config.py -v`
Expected: FAIL (`ImportError: kengine.config`).

- [ ] **Step 4: Implement `config.py` and the two types in `backend/types.py`**

  - Defaults:
    - `think`: `max_tokens=8192`.
    - `extract`: temperature 0.3, top_p 0.8, top_k 20, min_p 0.0, presence 0.0, `max_tokens=4096`.
    - `critic`: same as `extract` but temperature 0.2.
    - `baseline`: a copy of `think` with `max_tokens = eval.token_budget_per_task`, computed in a model validator unless the user set it explicitly.
    - `data_dir`: `~/.local/share/kengine`.
  - `load_config` merges TOML dicts recursively (deep merge) in the given order, using `tomllib`, then calls `KengineConfig.model_validate`.
  - `default_config_paths()` returns `[~/.config/kengine/config.toml, Path.cwd() / "kengine.toml"]`.
  - `kengine.example.toml` lists every key with a one-line comment.

- [ ] **Step 5: Run tests, pyright and ruff, and confirm they pass**

Run: `uv run pytest tests/test_config.py -v && uv run pyright && uv run ruff check`
Expected: 4 passed, `0 errors`, `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: project scaffold, toolchain, config loading"
```

---

## Task 2: Backend types and scripted fake backend

**Files:**
- Modify: `src/kengine/backend/types.py`
- Create: `src/kengine/backend/fake.py`
- Test: `tests/backend/test_fake.py`

**Interfaces:**
- Consumes: `RoleConfig` (Task 1).
- Produces (in `backend/types.py`):
  - `Message(role: Literal["system","user","assistant"], content: str)`
  - `Usage(prompt_tokens: int, completion_tokens: int)`
  - `Completion(content: str, reasoning: str | None, usage: Usage, finish_reason: Literal["stop","length","other"])`
  - `BackendHealth(ok: bool, model: str | None, n_ctx: int | None, detail: str)`
  - `class BackendError(Exception)`
  - `class StructuredOutputError(BackendError)` with attribute `raw: str`
  - `class ScriptExhausted(BackendError)`
  - `class ModelBackend(Protocol)`:
    - `complete(self, messages: Sequence[Message], role: RoleConfig) -> Completion`
    - `complete_structured(self, messages: Sequence[Message], role: RoleConfig, schema: type[T]) -> tuple[T, Completion]`, with `T` bound to `BaseModel`
    - `count_tokens(self, text: str) -> int`
    - `health(self) -> BackendHealth`
- Produces (in `backend/fake.py`):
  - `ScriptedReply(content: str = "", reasoning: str | None = None, finish_reason: ... = "stop", error: str | None = None)`. If `error` is set, the call raises `BackendError(error)`.
  - `RecordedCall(messages: list[Message], role: RoleConfig, schema_name: str | None)`
  - `ScriptedBackend(script: Sequence[ScriptedReply], health: BackendHealth | None = None)`, with attribute `calls: list[RecordedCall]`

- [ ] **Step 1: Write the failing tests**

```python
def test_replies_consumed_in_order_and_calls_recorded(): ...
    # two replies; complete() twice returns them in order; calls[1].schema_name is None
def test_structured_valid_json_parses():
    # reply content '{"ok": true}' with schema Probe(ok: bool) -> (Probe(ok=True), completion)
    # calls[0].schema_name == "Probe"
def test_structured_invalid_json_raises_with_raw():
    # content '{"ok": ' -> StructuredOutputError; exc.raw == '{"ok": '
def test_structured_truncated_raises_even_if_parseable():
    # content '{"ok": true}', finish_reason "length" -> StructuredOutputError
def test_error_reply_raises_backend_error():
    # ScriptedReply(error="connection refused") -> BackendError("connection refused")
def test_exhausted_script_raises():
    # empty script -> ScriptExhausted
def test_count_tokens_is_ceil_chars_over_4():
    # count_tokens("abcde") == 2; count_tokens("") == 0
def test_usage_counts_tokens_of_reply():
    # usage.completion_tokens == count_tokens(content + (reasoning or ""))
    # prompt_tokens == sum of count_tokens over message contents
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/backend/test_fake.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement the types and `ScriptedBackend`**

The fake follows the same structured-output semantics the real backend must have: truncation, or JSON that doesn't validate, raises `StructuredOutputError`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/backend -v && uv run pyright`
Expected: all pass, `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: backend protocol and scripted fake backend"
```

---

## Task 3: llama.cpp server backend

**Files:**
- Create: `src/kengine/backend/llamacpp.py`
- Modify: `src/kengine/backend/__init__.py`
- Test: `tests/backend/test_llamacpp.py`, using `httpx.MockTransport`, with no network

**Interfaces:**
- Consumes: Task 2 types, `BackendConfig` (Task 1).
- Produces:
  - `LlamaCppBackend(config: BackendConfig, client: httpx.Client | None = None)`, which implements `ModelBackend`
  - `make_backend(config: BackendConfig) -> ModelBackend`

- [ ] **Step 1: Write the failing tests**

Each test installs a `MockTransport` handler that captures the request and returns canned JSON.

```python
def test_complete_sends_sampling_and_thinking_flag():
    # POST {base}/v1/chat/completions; body has model, messages, temperature, top_p, top_k,
    # min_p, presence_penalty, repeat_penalty (llama-server name), max_tokens,
    # chat_template_kwargs == {"enable_thinking": True}
def test_reasoning_content_returned_separately():
    # response message {"content": "Final answer: 5", "reasoning_content": "2+3..."}
    # -> Completion.content == "Final answer: 5", reasoning == "2+3..."
def test_inline_think_tags_are_split_out():
    # content "<think>abc</think>\n\nhello" -> content == "hello", reasoning == "abc"
def test_structured_sends_json_schema_response_format():
    # body["response_format"] == {"type": "json_schema",
    #   "json_schema": {"name": "Probe", "schema": Probe.model_json_schema()}}
    # and chat_template_kwargs.enable_thinking is False when role.thinking is False
def test_structured_length_finish_raises():          # Review Focus #1
    # finish_reason "length" -> StructuredOutputError
def test_structured_schema_violation_raises():       # Review Focus #1
    # content '{"wrong": 1}' -> StructuredOutputError with .raw == '{"wrong": 1}'
def test_http_500_and_connect_error_raise_backend_error():   # Review Focus #3
def test_finish_reason_mapping():
    # "stop"->"stop", "length"->"length", anything else->"other"
def test_count_tokens_uses_tokenize_endpoint():
    # POST /tokenize {"content": "hi"} -> {"tokens": [1, 2, 3]} -> 3
def test_health_reads_props():
    # GET /health 200; GET /props {"default_generation_settings": {"n_ctx": 32768},
    #   "model_path": "/m/qwen.gguf"} -> BackendHealth(ok=True, n_ctx=32768, model="qwen.gguf")
    # /props with top-level "n_ctx" is also accepted
def test_health_unreachable_is_not_ok():
    # connect error -> BackendHealth(ok=False, detail contains the error), no exception
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/backend/test_llamacpp.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `LlamaCppBackend`**

  - Use `httpx.Client(base_url=..., timeout=config.timeout_s)`.
  - Map every `httpx.HTTPError` and non-2xx response to `BackendError`, with the status and the start of the body.
  - Usage comes from `usage.prompt_tokens` and `usage.completion_tokens`.
  - `make_backend` returns `LlamaCppBackend(config)`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/backend -v && uv run pyright`
Expected: all pass, `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: llama-server backend with structured output and thinking toggle"
```

---

## Task 4: Number normalization, answer extraction and scoring

**Files:**
- Create: `src/kengine/numbers.py`, `src/kengine/eval/__init__.py`, `src/kengine/eval/types.py`, `src/kengine/eval/answers.py`, `src/kengine/eval/scoring.py`
- Test: `tests/test_numbers.py`, `tests/eval/test_answers.py`, `tests/eval/test_scoring.py`

**Interfaces:**
- Produces (in `numbers.py`):
  - `parse_number(text: str) -> Fraction | None`
  - `decimal_places(text: str) -> int | None`. This is the number of digits after the decimal point in the first numeric token, used by Task 11.
- Produces (in `eval/types.py`):
  - `SuiteName = Literal["gsm8k","hotpotqa","folio"]`
  - `Size = Literal["smoke","small","full"]`
  - `SIZE_COUNTS: dict[str, int] = {"smoke": 10, "small": 100}`
  - `EvalItem(id: str, suite: SuiteName, question: str, documents: list[str] = [], gold: str, answer_format: str)`
  - `SolveStatus = Literal["answered","abstained","partial","blocked","error"]`
  - `SolveResult(answer: str | None, status: SolveStatus, model_calls: int, prompt_tokens: int, completion_tokens: int, wall_s: float, session_dir: str | None = None, transcript_excerpt: str | None = None, error: str | None = None)`
  - `ItemScore(correct: bool, detail: str = "")`
  - `class Solver(Protocol)`: `name: str`, `solve(self, item: EvalItem) -> SolveResult`
- Produces (in `eval/answers.py`):
  - `extract_final_answer(text: str) -> str | None`
  - `normalize_text(s: str) -> str`
- Produces (in `eval/scoring.py`):
  - `score_numeric(gold: str, answer: str | None) -> ItemScore`
  - `score_em_f1(gold: str, answer: str | None) -> ItemScore`. Here `correct` is exact match, and `detail` is `"f1=0.667"`.
  - `score_label(gold: str, answer: str | None) -> ItemScore`

- [ ] **Step 1: Write the failing tests** (Review Focus #5)

```python
@pytest.mark.parametrize("text,expected", [
    ("1,234", Fraction(1234)), ("$30.00", Fraction(30)), ("−5", Fraction(-5)),
    ("15%", Fraction(15)), ("**42**", Fraction(42)), ("42.", Fraction(42)),
    ("18 dollars", Fraction(18)), ("3.25", Fraction(13, 4)), ("1 234", Fraction(1234)),
    ("abc", None), ("", None)])
def test_parse_number(text, expected): assert parse_number(text) == expected

def test_decimal_places(): assert (decimal_places("3.30"), decimal_places("12"), decimal_places("x")) == (2, 0, None)

def test_extract_final_answer_takes_last_marker():
    assert extract_final_answer("Final answer: 3\n...\nfinal answer: **7**.") == "7"
    assert extract_final_answer("The answer is 7") is None
    assert extract_final_answer("Final Answer：  Paris ") == "Paris"   # full-width colon

def test_score_numeric():
    assert score_numeric("1234", "$1,234.00").correct
    assert not score_numeric("1234", "1235").correct
    assert not score_numeric("5", None).correct

def test_score_em_f1():
    s = score_em_f1("the Eiffel Tower", "Eiffel tower.")
    assert s.correct and s.detail == "f1=1.000"
    p = score_em_f1("Barack Obama", "Obama")
    assert not p.correct and p.detail == "f1=0.667"

def test_score_label_accepts_unknown_as_uncertain():
    assert score_label("Uncertain", "unknown").correct
    assert score_label("True", "true").correct
    assert not score_label("False", "True").correct
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_numbers.py tests/eval -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement**

  - **`parse_number`:**
    - First strip markdown `*`/`_`, then currency symbols `$€£`.
    - Map `−` to `-`.
    - Remove a thousands separator (`,` or a single space or thin space) only when it sits between a digit and exactly 3 digits.
    - Take the first match of `-?\d+(\.\d+)?`, then `Fraction(str)`. A trailing `%` is dropped (the number is kept as written).
  - **`extract_final_answer`:**
    - Use the last match of `(?im)^\s*final answer\s*[:：]\s*(.+?)\s*$`.
    - Strip surrounding `*`/`_` and one trailing `.`.
  - **`normalize_text`:** SQuAD-style. Lowercase, remove punctuation, remove the articles `a|an|the`, collapse whitespace.
  - **`score_em_f1`:** exact match plus token-overlap F1 over the normalized tokens.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_numbers.py tests/eval -v && uv run pyright`
Expected: all pass, `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: number normalization, answer extraction, scorers"
```

---

## Task 5: Benchmark suite adapters and loading

**Files:**
- Create: `src/kengine/eval/adapters/__init__.py`, `gsm8k.py`, `hotpotqa.py`, `folio.py`, `src/kengine/eval/suites.py`
- Test: `tests/eval/test_adapters.py`, `tests/eval/test_suites.py`

**Interfaces:**
- Consumes: `EvalItem`, the scorers (Task 4).
- Produces:
  - In each adapter: `parse_row(row: Mapping[str, object], index: int) -> EvalItem`
  - `SuiteSpec(name: SuiteName, hf_id: str, hf_config: str | None, split: str, parse_row: Callable[[Mapping[str, object], int], EvalItem], score: Callable[[str, str | None], ItemScore])`
  - `SUITES: dict[SuiteName, SuiteSpec]`
  - `select_indices(total: int, size: Size, seed: int) -> list[int]`
  - `load_items(suite: SuiteName, size: Size, seed: int = 0) -> list[EvalItem]`

Pinned dataset sources (§6):

| Suite | `hf_id` | config | split | score |
|---|---|---|---|---|
| gsm8k | `openai/gsm8k` | `main` | `test` | `score_numeric` |
| hotpotqa | `hotpotqa/hotpot_qa` | `distractor` | `validation` | `score_em_f1` |
| folio | `yale-nlp/FOLIO` | None | `validation` | `score_label` |

FOLIO may be gated. If it fails to load, `load_items` tries the fallback `tasksource/folio` (same field names) and records which one it used in the item ids' prefix.

- [ ] **Step 1: Write the failing tests**, using inline fixture rows that match the HF field names

```python
def test_gsm8k_row():
    item = gsm8k.parse_row({"question": "Q?", "answer": "steps...\n#### 1,234"}, 7)
    assert (item.id, item.gold, item.suite) == ("gsm8k-7", "1234", "gsm8k")
    assert item.answer_format == "Give the final answer as a number."

def test_hotpotqa_row_builds_documents():
    row = {"id": "5a8b", "question": "Who?", "answer": "Ann",
           "context": {"title": ["T1", "T2"], "sentences": [["a.", "b."], ["c."]]}}
    item = hotpotqa.parse_row(row, 0)
    assert item.id == "hotpotqa-5a8b" and item.documents == ["T1: a. b.", "T2: c."]
    assert item.answer_format == "Give the final answer as a short phrase."

def test_folio_row():
    row = {"example_id": 3, "premises": "All A are B.", "conclusion": "X is B.", "label": "Uncertain"}
    item = folio.parse_row(row, 0)
    assert item.id == "folio-3" and item.gold == "Uncertain"
    assert "Premises:\nAll A are B." in item.question and "Conclusion: X is B." in item.question
    assert item.answer_format == "Answer with exactly one of: True, False, Uncertain."

def test_select_indices_seeded_sorted_and_sized():
    a = select_indices(1000, "smoke", 0)
    assert len(a) == 10 and a == sorted(a) and a == select_indices(1000, "smoke", 0)
    assert a != select_indices(1000, "smoke", 1)
    assert select_indices(50, "small", 0) == list(range(50))    # fewer rows than size → all
    assert select_indices(30, "full", 0) == list(range(30))
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/eval/test_adapters.py tests/eval/test_suites.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement the adapters, `SUITES`, `select_indices` and `load_items`**

  - `select_indices` uses `sorted(random.Random(seed).sample(range(total), n))`.
  - `load_items` imports `datasets` inside the function. If it's missing, it raises `RuntimeError("install with: uv sync --extra eval")`.
  - Rows are cast to `Mapping[str, object]`, with a narrow `# pyright: ignore[reportMissingTypeStubs]` on the import only.
  - `load_items` isn't unit-tested, because it needs the network. Task 7's runbook covers it.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/eval -v && uv run pyright`
Expected: all pass, `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: gsm8k/hotpotqa/folio adapters and seeded subset selection"
```

---

## Task 6: Eval runner (resumable) and baseline solver

**Files:**
- Create: `src/kengine/eval/runner.py`, `src/kengine/eval/baseline.py`, `src/kengine/prompts/baseline.md`, `src/kengine/prompts/__init__.py` (with `load_prompt(name: str) -> str` via `importlib.resources`)
- Test: `tests/eval/test_runner.py`, `tests/eval/test_baseline.py`

**Interfaces:**
- Consumes: Tasks 2, 4 and 5.
- Produces:
  - `RunMetrics(suite: str, solver: str, n: int, headline_accuracy: float, abstention_rate: float, answered_accuracy: float | None, status_counts: dict[str, int], mean_model_calls: float, mean_prompt_tokens: float, mean_completion_tokens: float, mean_wall_s: float)`, defined in `eval/types.py`
  - `results_dir_name(date: datetime.date, suite: str, solver: str, size: str, profile: str) -> str`, giving `"2026-11-02-gsm8k-baseline-smoke-none"`
  - `run_eval(spec: SuiteSpec, items: Sequence[EvalItem], solver: Solver, out_dir: Path, run_config: Mapping[str, object], failure_sample_size: int = 25, seed: int = 0) -> RunMetrics`
  - `BaselineSolver(backend: ModelBackend, role: RoleConfig, token_budget: int)`, with `name = "baseline"`
  - `load_prompt(name: str) -> str`

**What `run_eval` writes to `out_dir`:**
- `config.json`: the `run_config` it was given
- `items.jsonl`: one line per item, `{item_id, status, answer, gold, correct, detail, model_calls, prompt_tokens, completion_tokens, wall_s, session_dir, error}`. It's appended and flushed after each item.
- `failures_sample.jsonl`: up to `failure_sample_size` incorrect items, chosen with a seeded sample, with `question`, `gold`, `answer`, `status`, `transcript_excerpt`
- `metrics.json`

**Resuming:** if `items.jsonl` already exists, items whose ids are present are skipped, and metrics are computed over every line.

**Errors:** if `solver.solve` raises `BackendError`, the item is recorded with `status="error"` and the run carries on.

- [ ] **Step 1: Write the failing tests**

```python
class StubSolver:  # returns canned SolveResults keyed by item id; may raise BackendError
    ...

def test_metrics_count_non_answered_as_wrong(tmp_path):
    # 4 items: answered-correct, answered-wrong, blocked, abstained
    m = run_eval(spec, items, StubSolver(...), tmp_path, {})
    assert m.headline_accuracy == 0.25
    assert m.abstention_rate == 0.5
    assert m.answered_accuracy == 0.5
    assert m.status_counts == {"answered": 2, "blocked": 1, "abstained": 1}

def test_backend_error_recorded_and_run_continues(tmp_path):   # Review Focus #3
    # solver raises BackendError on item 2 of 3 -> 3 lines in items.jsonl, item 2 status "error"

def test_resume_skips_completed_items(tmp_path):
    # pre-write items.jsonl with item 1; run over items 1..3; solver.solve called only for 2 and 3;
    # metrics.n == 3

def test_failures_sample_capped_and_seeded(tmp_path):
    # 40 wrong items, failure_sample_size=25 -> 25 lines; same ids on a second run with same seed

def test_results_dir_name():
    assert results_dir_name(date(2026, 11, 2), "gsm8k", "baseline", "smoke", "none") \
        == "2026-11-02-gsm8k-baseline-smoke-none"

def test_baseline_single_call_with_budget_and_answer_extraction():
    backend = ScriptedBackend([ScriptedReply(content="... Final answer: 42", reasoning="r")])
    r = BaselineSolver(backend, think_role, token_budget=1000).solve(item)
    assert r.status == "answered" and r.answer == "42" and r.model_calls == 1
    assert backend.calls[0].role.max_tokens == 1000 and backend.calls[0].role.thinking
    user = backend.calls[0].messages[-1].content
    assert item.question in user and item.answer_format in user

def test_baseline_truncated_without_answer_is_partial():
    # finish_reason "length", no Final answer line -> status "partial", answer None
def test_baseline_no_answer_line_is_abstained():
    # finish_reason "stop", no marker -> "abstained"
def test_baseline_documents_are_numbered_in_prompt():
    # documents ["A", "B"] -> user message contains "[1] A" and "[2] B"
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/eval/test_runner.py tests/eval/test_baseline.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement the runner, the baseline and `baseline.md`**

  **`baseline.md` (system prompt):** a careful assistant. It thinks step by step and ends with exactly one line `Final answer: <answer>`.

  **User message layout:**
  - `## Documents`, with `[i] text` lines, only if there are documents
  - `## Question`
  - the `answer_format` line

  **`transcript_excerpt`:** the last 2000 characters of the reasoning, then `\n---\n`, then the last 2000 characters of the content.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/eval -v && uv run pyright`
Expected: all pass, `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: resumable eval runner and bare-model baseline solver"
```

---

## Task 7: CLI (`doctor`, `eval`, `eval-summarize`), error taxonomy and runbook checkpoint A

**Files:**
- Create: `src/kengine/cli/__init__.py`, `src/kengine/cli/main.py`, `src/kengine/cli/doctor.py`, `src/kengine/eval/taxonomy.py`, `docs/eval/error-taxonomy.md`, `docs/real-machine-runbook.md`
- Test: `tests/cli/test_doctor.py`, `tests/cli/test_cli.py` (typer `CliRunner`), `tests/eval/test_taxonomy.py`

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces:
  - `DoctorReport(config_paths: list[str], health: BackendHealth, structured_ok: bool, structured_detail: str, thinking_separated: bool | None, gpu: str | None, ok: bool)`
  - `run_doctor(config: KengineConfig, config_paths: Sequence[Path], backend: ModelBackend, nvidia_smi: Callable[[], str | None]) -> DoctorReport`
  - `ErrorLabel` (StrEnum): `fabricated_fact`, `invalid_inference`, `arithmetic_error`, `inconsistency`, `forgotten_constraint`, `misread_question`, `format_failure`, `truncated`, `other`
  - `TaxonomyRow(item_id: str, labels: list[ErrorLabel], note: str = "")`
  - `summarize_taxonomy(path: Path) -> dict[ErrorLabel, int]`
  - A typer `app` with these commands:
    - `kengine doctor`
    - `kengine eval SUITE --solver [baseline|kengine] --size [smoke|small|full] --profile NAME --seed INT --out PATH`
    - `kengine eval-summarize RESULTS_DIR`
  - Module-level `backend_factory: Callable[[BackendConfig], ModelBackend] = make_backend`, so tests can monkeypatch it
  - `--solver kengine` exits with code 2 and the message "available after Task 16" until Task 16 wires it up

- [ ] **Step 1: Write the failing tests**

```python
def test_doctor_ok_with_healthy_fake():
    # ScriptedBackend(health=ok, script=[structured '{"ok": true}', thinking reply with reasoning="r"])
    rep = run_doctor(cfg, [], backend, nvidia_smi=lambda: "8151MiB / 8188MiB")
    assert rep.ok and rep.structured_ok and rep.thinking_separated is True and rep.gpu.startswith("8151")

def test_doctor_not_ok_when_unreachable():
    # health ok=False -> rep.ok is False, structured probe not attempted (backend.calls == [])

def test_doctor_not_ok_when_structured_fails():
    # structured reply '{"ok": ' -> rep.ok False, structured_detail mentions StructuredOutputError

def test_cli_doctor_exit_codes(monkeypatch):     # 0 when ok, 1 when not ok

def test_cli_eval_baseline_writes_results(monkeypatch, tmp_path):
    # monkeypatch backend_factory -> ScriptedBackend answering "Final answer: 1234";
    # monkeypatch suites.load_items -> two gsm8k items
    # run: kengine eval gsm8k --solver baseline --size smoke --out <tmp>
    # -> exit 0; exactly one dir under tmp named "*-gsm8k-baseline-smoke-none" with metrics.json

def test_summarize_taxonomy(tmp_path):
    # taxonomy.jsonl rows with labels [arithmetic_error], [arithmetic_error, truncated] ->
    # {ARITHMETIC_ERROR: 2, TRUNCATED: 1}; unknown label -> ValidationError naming the line number
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/cli tests/eval/test_taxonomy.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement the CLI and doctor**

  **`run_doctor` probes:**
  - health
  - `complete_structured` with `Probe(ok: bool)` using the `extract` role
  - `complete` with the `think` role on "What is 2+3? Answer briefly." It reports `thinking_separated = completion.reasoning is not None`.
  - `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader` if it's on PATH

  **Output:**
  - `doctor` prints a readable report and also writes it as JSON with `--json PATH`.
  - `eval-summarize` prints a count table, sorted by count, and writes `taxonomy_summary.json` next to the input.

- [ ] **Step 4: Write `docs/eval/error-taxonomy.md`**

  The rubric gives:
  - one definition and one example per `ErrorLabel`
  - the rule "label the *first* error that made the answer wrong; add more labels only if independent"
  - the `taxonomy.jsonl` line format

- [ ] **Step 5: Write `docs/real-machine-runbook.md`, with Setup and Checkpoint A**

  **Setup:**
  - `git clone`, then `uv sync --extra eval`
  - Build llama.cpp with CUDA.
  - Start the server with:
    `llama-server -m <GGUF> -ngl 99 -c 32768 --jinja --reasoning-format deepseek -fa on --cache-type-k q8_0 --cache-type-v q8_0 --port 8080`
    Note: flags change between llama.cpp versions. Check them against `llama-server --help`, and record the exact command used in the results.
  - Write `kengine.toml` if the defaults need changing.
  - If llama.cpp refuses the Qwen3.5 architecture, use standard Qwen3-8B (§3), change only the config, and record it.

  **Checkpoint A** (owner review point 1):
  1. `kengine doctor --json eval/results/doctor-<date>.json`
  2. For each suite in gsm8k, folio, hotpotqa: run `kengine eval <suite> --solver baseline --size smoke`, then `--size small`.
  3. For each `failures_sample.jsonl`, write `taxonomy.jsonl` using the rubric, then run `kengine eval-summarize <dir>`.
  4. Commit everything under `eval/results/` except `sessions/`, plus a `eval/results/CHECKPOINT-A.md` summary: headline accuracy per suite, top 3 error labels, the server command, the GGUF file name. Push.

- [ ] **Step 6: Run the tests and checks and confirm they pass**

Run: `uv run pytest -v && uv run pyright && uv run ruff check && uv run kengine --help`
Expected: all pass, `0 errors`, and help lists `doctor`, `eval`, `eval-summarize`.

- [ ] **Step 7: Commit, and tell the owner that Checkpoint A can run**

```bash
git add -A && git commit -m "feat: doctor/eval CLI, error taxonomy, real-machine runbook checkpoint A"
```

---

## Task 8: Claim schema, extraction schema and statuses

**Files:**
- Create: `src/kengine/claims/__init__.py`, `claims/schema.py`, `claims/extraction.py`, `src/kengine/beliefs/__init__.py`, `beliefs/status.py`
- Test: `tests/claims/test_schema.py`

**Interfaces:**
- Produces:
  - `Status` (StrEnum, §5.4): `unchecked`, `verified`, `supported`, `critic_accepted`, `assumed`, `contradicted`, `retracted`, `invalidated`
  - `TERMINAL: frozenset[Status] = {contradicted, retracted, invalidated}`
  - `ClaimType` (StrEnum, §5.3): `observation`, `given`, `fact`, `definition`, `assumption`, `computation`, `inference`, `hypothesis`, `precondition`
  - `SourceKind` (StrEnum): `model`, `tool`, `skill`, `subagent`, `user`, `task`
  - `Source(kind: SourceKind, ref: str = "")`
  - `Computation(expression: str, claimed_result: str)`
  - `Claim(id: str, type: ClaimType, text: str, premises: tuple[str, ...] = (), computation: Computation | None = None, confidence: float | None = None, source: Source, step: int)`, frozen
  - `ExtractableType = Literal["fact","definition","assumption","computation","inference","hypothesis"]`
  - `ExtractedClaim(ref: str, type: ExtractableType, text: str, premises: list[str] = [], computation: Computation | None = None)`
  - `ProposedAction(kind: Literal["final_answer","continue"], answer: str | None = None, preconditions: list[str] = [])`
  - `Extraction(claims: list[ExtractedClaim], action: ProposedAction)`

- [ ] **Step 1: Write the failing tests**

```python
def test_computation_claim_requires_computation():
    with pytest.raises(ValidationError): Claim(id="b1", type=ClaimType.COMPUTATION, text="x", source=SRC, step=1)
def test_non_computation_claim_rejects_computation():
    with pytest.raises(ValidationError):
        Claim(id="b1", type=ClaimType.FACT, text="x", computation=Computation(expression="1", claimed_result="1"), source=SRC, step=1)
def test_confidence_bounds():   # 1.2 -> ValidationError; 0.5 ok
def test_claim_id_pattern():    # "b12" ok, "c1" -> ValidationError
def test_extracted_ref_pattern_and_limits():
    # ref "c1" ok, "x" rejected; text "" rejected; text of 401 chars rejected; 13 claims rejected
    # premises may contain "c2" or "b7"; "zz" rejected
def test_extraction_json_schema_is_self_contained():
    s = Extraction.model_json_schema()
    assert s["properties"]["claims"]["maxItems"] == 12
    assert "computation" in s["$defs"]["ExtractedClaim"]["properties"]
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/claims -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement the models**

  Pinned constraints:
  - `Claim.id` matches `^b[0-9]+$`
  - `ExtractedClaim.ref` matches `^c[0-9]+$`
  - each premise matches `^[bc][0-9]+$`
  - `text` is 1–400 characters
  - `claims` has at most 12 items (`Field(max_length=12)`)
  - `confidence` is in `[0, 1]`
  - `computation` is present if and only if `type == computation`, for both `Claim` and `ExtractedClaim`

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/claims -v && uv run pyright`
Expected: all pass, `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: claim, extraction and status models"
```

---

## Task 9: Checker contract, belief store and session event log

**Files:**
- Create: `src/kengine/checkers/__init__.py`, `checkers/base.py`, `beliefs/records.py`, `beliefs/store.py`, `src/kengine/audit/__init__.py`, `audit/events.py`, `audit/session.py`
- Test: `tests/beliefs/test_store.py`, `tests/audit/test_session.py`

**Interfaces:**
- Consumes: Task 8.
- Produces (in `checkers/base.py`):
  - `CheckerKind` (StrEnum): `deterministic`, `evidence`, `critic`
  - `Outcome` (StrEnum): `pass`, `fail`, `unknown`
  - `Verdict(checker_id: str, checker_kind: CheckerKind, outcome: Outcome, status: Status | None, explanation: str, evidence: list[str] = [], counterexample: str | None = None)`
  - `class BeliefView(Protocol)`: `get(self, id: str) -> BeliefRecord`, `exists(self, id: str) -> bool`, `all(self) -> list[BeliefRecord]`, `premise_closure(self, ids: Sequence[str]) -> list[str]`
  - `class Checker(Protocol)`: `id: str`, `kind: CheckerKind`, `applies_to(self, claim: Claim) -> bool`, `check(self, claim: Claim, view: BeliefView) -> Verdict`
- Produces (in `beliefs/records.py`, which imports only `claims` and `beliefs.status`, so that `checkers/base.py` can import it while `beliefs/store.py` imports `Verdict`):
  - `BeliefRecord(claim: Claim, status: Status, status_reason: str)`
- Produces (in `beliefs/store.py`):
  - `StatusChange(belief_id: str, old: Status, new: Status, reason: str, verdict: Verdict | None, step: int)`
  - `class UnknownPremiseError(ValueError)`
  - `class VerificationGuardError(ValueError)`
  - `BeliefStore`, which implements `BeliefView`:
    - `open(cls, path: Path | Literal[":memory:"]) -> BeliefStore`
    - `add(self, *, type: ClaimType, text: str, premises: Sequence[str], source: Source, step: int, computation: Computation | None = None, confidence: float | None = None) -> Claim`
    - `apply_verdict(self, id: str, verdict: Verdict, step: int) -> list[StatusChange]`
    - `retract(self, id: str, reason: str, step: int) -> list[StatusChange]`
    - `dependents(self, id: str) -> list[str]`
    - `history(self, id: str) -> list[StatusChange]`
    - `close(self) -> None`
- Produces (in `audit/events.py`): an `Event` discriminated union on `kind`. Every variant has `ts: datetime` (UTC) and `step: int`. The variants:
  - `session_started(task: Task-as-dict)`
  - `model_call(role: str, messages: list[Message], completion: Completion, wall_s: float)`
  - `think(text: str, reasoning: str | None)`
  - `extraction(sample_index: int, ok: bool, raw: str | None, error: str | None)`
  - `vote(kept: list[str], dropped: list[str], n_valid: int)`
  - `belief_added(claim: Claim)`
  - `verdict(belief_id: str, verdict: Verdict)`
  - `status_changed(change: StatusChange)`
  - `gate(decision: dict)`
  - `feedback(items: list[str])`
  - `run_finished(result: dict)`
  - `EventLog(path: Path)` with `append(self, event: Event) -> None` (one JSON line, flushed) and `read(self) -> list[Event]`
- Produces (in `audit/session.py`):
  - `Session(dir: Path, store: BeliefStore, log: EventLog)`
  - `Session.create(root: Path) -> Session`, which makes the directory `<root>/<YYYYmmdd-HHMMSS>-<6 hex>/` containing `beliefs.sqlite` and `events.jsonl`
  - `Session.open(dir: Path) -> Session`

**Store rules** (§5.4, and the Global Constraints):
- **Ids:** `b1`, `b2`, … in insertion order.
- **Premises:** every premise must already exist, otherwise `UnknownPremiseError`. So the graph has no cycles by construction.
- **Initial status:** `assumed` for type `assumption`; `unchecked` for everything else.
- **`apply_verdict`:**
  - `verdict.status is None` → no status change, but the verdict is recorded.
  - Setting `verified` or `supported` with `checker_kind == critic` raises `VerificationGuardError`.
  - Setting `verified` is allowed only for kinds `deterministic` and `evidence`.
  - If the belief is already in a `TERMINAL` status, the verdict is recorded and the status is left unchanged. There's no un-retracting in v1.
- **Propagation:** when the new status is `contradicted` or `retracted`, every transitive dependent that isn't already terminal becomes `invalidated`, with reason `"depends on {id} ({new})"`.
- **`premise_closure(ids)`:** the ids together with all their transitive premises, deduplicated, in insertion order.
- **Persistence:** tables `beliefs(id, seq, claim_json, status, reason)` and `changes(seq, belief_id, old, new, reason, verdict_json, step)`. Reopening the file restores everything.

- [ ] **Step 1: Write the failing tests**

```python
def test_ids_sequential_and_initial_status():
    s = BeliefStore.open(":memory:")
    a = s.add(type=ClaimType.FACT, text="x", premises=[], source=SRC, step=1)
    b = s.add(type=ClaimType.ASSUMPTION, text="y", premises=[], source=SRC, step=1)
    assert (a.id, b.id) == ("b1", "b2")
    assert s.get("b1").status == Status.UNCHECKED and s.get("b2").status == Status.ASSUMED

def test_unknown_premise_rejected():  # premises=["b9"] -> UnknownPremiseError

def test_contradiction_propagates_transitively():
    # b1 fact; b2 inference(b1); b3 inference(b2); b4 fact (independent)
    changes = s.apply_verdict("b1", Verdict(..., status=Status.CONTRADICTED, ...), step=2)
    assert [c.belief_id for c in changes] == ["b1", "b2", "b3"]
    assert s.get("b3").status == Status.INVALIDATED and s.get("b3").status_reason == "depends on b1 (contradicted)"
    assert s.get("b4").status == Status.UNCHECKED

def test_critic_cannot_verify():
    with pytest.raises(VerificationGuardError):
        s.apply_verdict("b1", Verdict(checker_kind=CheckerKind.CRITIC, status=Status.VERIFIED, ...), 1)

def test_terminal_status_is_sticky():
    # retract b1, then a deterministic VERIFIED verdict -> status stays RETRACTED, returns [], history has 1 change

def test_none_status_verdict_records_without_change():

def test_premise_closure_order():
    # b3(b2), b2(b1) -> premise_closure(["b3"]) == ["b1", "b2", "b3"]

def test_reopen_restores_state(tmp_path):
    # add + contradict, close, BeliefStore.open(same path) -> same statuses, same next id ("b4")

def test_session_create_and_event_roundtrip(tmp_path):
    sess = Session.create(tmp_path)
    assert (sess.dir / "beliefs.sqlite").exists()
    sess.log.append(FeedbackEvent(step=1, items=["x"]))
    assert Session.open(sess.dir).log.read()[0].items == ["x"]
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/beliefs tests/audit -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `checkers/base.py`, `beliefs/store.py`, `audit/events.py` and `audit/session.py`**

  - `sqlite3` with a single connection. Commit after every mutating call, because sessions must survive crashes (§5.4).
  - `Task` doesn't exist until Task 13, so `session_started` stores `task: dict[str, object]`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/beliefs tests/audit -v && uv run pyright`
Expected: all pass, `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: TMS belief store with verification guard, session event log"
```

---

## Task 10: Structure checker

**Files:**
- Create: `src/kengine/checkers/structure.py`
- Test: `tests/checkers/test_structure.py`

**Interfaces:**
- Consumes: Task 9.
- Produces: `StructureChecker()`, with `id = "structure"` and `kind = deterministic`. It applies to every claim.

**Rules** (§5.5 #1). Cycles and missing premises can't happen, because the store prevents them (Task 9).
1. An `inference` with no premises → `fail`, status `None`, explanation `"inference has no premises"`.
2. Any premise in a `TERMINAL` status → `fail`, status `invalidated`, explanation `"rests on {pid}, which is {status}"`, naming the first such premise.
3. Otherwise → `pass`, status `None`. Structure alone never verifies anything.

- [ ] **Step 1: Write the failing tests**

```python
def test_inference_without_premises_fails_without_status():
    v = StructureChecker().check(inference_claim_no_premises, store)
    assert (v.outcome, v.status, v.explanation) == (Outcome.FAIL, None, "inference has no premises")
def test_premise_in_terminal_status_invalidates():
    # b1 retracted; b2 fact with premise b1 -> FAIL, INVALIDATED, "rests on b1, which is retracted"
def test_well_formed_claim_passes_without_verifying():
    # -> Outcome.PASS, status None
```

- [ ] **Step 2: Run them and confirm they fail.** Run: `uv run pytest tests/checkers/test_structure.py -v`. Expected: FAIL (`ImportError`).
- [ ] **Step 3: Implement `StructureChecker`.**
- [ ] **Step 4: Run the tests and confirm they pass.** Run: `uv run pytest tests/checkers -v && uv run pyright`. Expected: pass, `0 errors`.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: structure checker"`

---

## Task 11: Safe arithmetic evaluator and arithmetic checker

**Files:**
- Create: `src/kengine/checkers/arith_eval.py`, `src/kengine/checkers/arithmetic.py`
- Test: `tests/checkers/test_arith_eval.py`, `tests/checkers/test_arithmetic.py`

**Interfaces:**
- Consumes: `parse_number`, `decimal_places` (Task 4), Task 9.
- Produces:
  - `class ArithError(ValueError)`
  - `safe_eval(expression: str) -> Fraction | float`
  - `results_match(actual: Fraction | float, claimed: str) -> bool | None`
  - `format_number(x: Fraction | float) -> str`
  - `ArithmeticChecker()`, with `id = "arithmetic"` and `kind = deterministic`. It applies to `type == computation`.

**`safe_eval` rules** (§5.5 #2: no `eval`):
- **Before parsing:**
  - Rewrite `×`→`*`, `÷`→`/`, `^`→`**` and `−`→`-`.
  - Remove a `,` only when it sits between a digit and exactly 3 digits followed by a word boundary.
  - Rewrite `N%` → `(N/100)`.
  - Reject input longer than 500 characters.
- **Parsing:** `ast.parse(mode="eval")`, then a whitelist walk:
  - `BinOp` with `Add Sub Mult Div FloorDiv Mod Pow`
  - `UnaryOp` with `UAdd USub`
  - `Constant` (int or float only, not bool)
  - `Call` to the names `sqrt abs round min max` only, with positional arguments only
- **Arithmetic:**
  - Integers become `Fraction`; floats become `Fraction(repr(value))`. `sqrt` returns a float, and so does a fractional power.
- **Power limits:** the exponent must be integral with absolute value ≤ 1000, and `|exp| * log10(max(|base|, 2)) <= 10_000`, otherwise `ArithError("result too large")`.
- **Errors:** division by zero and anything not on the whitelist raise `ArithError` with a short reason.

**`results_match`:**
- Parse `claimed` with `parse_number`. If that fails, return `None`.
- With `d = decimal_places(claimed)`:
  - if `d == 0`, it matches when `|actual - claimed| <= 1e-9 * max(1, |actual|)`
  - otherwise it matches when `|actual - claimed| <= 0.5 * 10**-d`

**Verdicts:**

| Situation | Outcome | Status | Explanation |
|---|---|---|---|
| Match | `pass` | `verified` | `` `{expr}` = {format_number(actual)} `` |
| Mismatch | `fail` | `contradicted` | `` `{expr}` evaluates to {format_number(actual)}, not {claimed} `` |
| `ArithError` or unparseable `claimed` | `unknown` | `None` | `"could not evaluate: {reason}"` |

`format_number` renders a whole-number `Fraction` as an integer string. Anything else is rounded to 6 decimal places, with trailing zeros removed.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("expr,value", [("12*7", 84), ("1,000 + 234", 1234), ("200*15%", 30),
    ("2^10", 1024), ("7 × 6", 42), ("(3 - −2) ÷ 5", 1), ("max(3, 9) - abs(-1)", 8), ("0.1 + 0.2", Fraction(3, 10))])
def test_safe_eval_values(expr, value): assert safe_eval(expr) == value

@pytest.mark.parametrize("expr", ["__import__('os')", "x + 1", "(1).real", "[1, 2]", "True + 1", "1/0", "lambda: 1"])
def test_safe_eval_rejects(expr):
    with pytest.raises(ArithError): safe_eval(expr)

def test_pathological_inputs_fail_fast():                # Review Focus #4
    start = time.monotonic()
    for expr in ["9**9**9", "2**100000", "1+" * 50_000 + "1"]:
        with pytest.raises(ArithError): safe_eval(expr)
    assert time.monotonic() - start < 1.0

@pytest.mark.parametrize("actual,claimed,ok", [
    (Fraction(10, 3), "3.33", True), (Fraction(10, 3), "3.34", False), (Fraction(10, 3), "3", False),
    (Fraction(84), "84", True), (Fraction(84), "82", False), (Fraction(30), "$30.00", True),
    (Fraction(1234), "1,234", True), (Fraction(-5), "−5", True), (Fraction(84), "eighty", None)])
def test_results_match(actual, claimed, ok): assert results_match(actual, claimed) is ok

def test_checker_verdicts():
    # 12*7 claimed 84 -> PASS/VERIFIED "`12*7` = 84"
    # 12*7 claimed 82 -> FAIL/CONTRADICTED "`12*7` evaluates to 84, not 82"
    # "x*2" claimed 4 -> UNKNOWN/None, explanation starts with "could not evaluate:"
    # 10/3 claimed 3.3333333 -> PASS; format_number(Fraction(10, 3)) == "3.333333"
```

- [ ] **Step 2: Run them and confirm they fail.** Run: `uv run pytest tests/checkers -v`. Expected: FAIL (`ImportError`).
- [ ] **Step 3: Implement `safe_eval`, `results_match`, `format_number` and `ArithmeticChecker`.** Python's own parser may fail on deeply nested input with `RecursionError` or `MemoryError`; catch those and wrap them as `ArithError`.
- [ ] **Step 4: Run the tests and confirm they pass.** Run: `uv run pytest tests/checkers -v && uv run pyright`. Expected: pass, `0 errors`.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: safe arithmetic evaluator and arithmetic checker"`

---

## Task 12: Cross-sample voting

**Files:**
- Create: `src/kengine/claims/voting.py`
- Test: `tests/claims/test_voting.py`

**Interfaces:**
- Consumes: `Extraction`, `ExtractedClaim`, `ProposedAction` (Task 8), `normalize_text` (Task 4).
- Produces:
  - `default_k(n: int) -> int`, which is `n // 2 + 1`
  - `VoteResult(claims: list[ExtractedClaim], action: ProposedAction | None, dropped: list[str], n_valid: int)`
  - `vote(samples: Sequence[Extraction], k: int, similarity: float = 0.8) -> VoteResult`

**Algorithm** (*plan's call*; §9 left this to the builder):
- **Matching:** two claims *match* when they have the same `type` and `difflib.SequenceMatcher(None, normalize_text(a.text), normalize_text(b.text)).ratio() >= similarity`.
- **Support:** a claim's *support* is the number of samples, including its own, that contain at least one matching claim.
- **Anchor:** the sample with the most claims whose support is ≥ k. Ties go to the earliest sample.
- **Kept claims:** the anchor's claims with support ≥ k, in the anchor's order. `dropped` lists the anchor refs that were removed.
- **Premises:** for kept claims, premise refs `c#` that point at dropped refs are removed. `b#` refs are kept as they are.
- **Action:** the kind is decided by majority across samples, with ties going to the anchor's kind. The answer and preconditions come from the anchor if its kind wins, otherwise from the earliest sample with the winning kind. Preconditions are filtered the same way as premises.
- **No samples:** `vote([])` returns `VoteResult([], None, [], 0)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_single_sample_k1_keeps_everything(): ...
def test_claim_in_one_of_three_samples_is_dropped_and_links_removed():
    # s1: c1 fact "Tom has 3 apples", c2 inference "Tom has 6 apples" premises [c1]
    # s2, s3: only a paraphrase of c2 ("tom has six apples" won't match; use "Tom has 6 apples.")
    # k=2 -> kept [c2] with premises == [], dropped == ["c1"]
def test_type_mismatch_does_not_match():  # same text, fact vs assumption -> separate
def test_anchor_is_most_supported_sample_not_first():
    # s1 is an outlier with 3 unique claims; s2 and s3 agree on 2 claims -> anchor s2; kept refs from s2
def test_action_majority_and_precondition_filtering():
    # kinds [continue, final_answer, final_answer] -> final_answer; answer from s2; preconditions
    # referencing a dropped ref are removed, "b4" kept
def test_empty_samples(): assert vote([], 1) == VoteResult(claims=[], action=None, dropped=[], n_valid=0)
def test_default_k(): assert [default_k(n) for n in (1, 3, 5)] == [1, 2, 3]
```

- [ ] **Step 2: Run them and confirm they fail.** Run: `uv run pytest tests/claims/test_voting.py -v`. Expected: FAIL (`ImportError`).
- [ ] **Step 3: Implement `vote` and `default_k`.**
- [ ] **Step 4: Run the tests and confirm they pass.** Run: `uv run pytest tests/claims -v && uv run pyright`. Expected: pass, `0 errors`.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: cross-sample claim voting"`

---

## Task 13: Task model, effort profiles and final-answer gate

**Files:**
- Create: `src/kengine/loop/__init__.py`, `loop/task.py`, `loop/profiles.py`, `loop/gate.py`
- Test: `tests/loop/test_profiles.py`, `tests/loop/test_gate.py`

**Interfaces:**
- Consumes: Task 9.
- Produces:
  - `Task(goal: str, documents: list[str] = [], answer_format: str | None = None)`
  - `CheckScope` (StrEnum): `load_bearing`, `all`
  - `GatePolicy` (StrEnum): `strict`, `standard`, `permissive`
  - `Profile(name: str, n_samples: int, check_scope: CheckScope, max_steps: int, final_gate: GatePolicy)`
  - `PROFILES: dict[str, Profile]`
  - `get_profile(name: str) -> Profile`. An unknown name raises `KeyError`, with the valid names in the message.
  - `GateDecision(allowed: bool, blocking: list[str], acted_on_assumptions: list[str], reasons: list[str])`
  - `evaluate_gate(preconditions: Sequence[str], view: BeliefView, policy: GatePolicy) -> GateDecision`

**Pinned profile values.** These come from the §5.2 table. The `max_steps` values are this plan's call. The critic and fact-check settings get added in later plans.

| name | n_samples | check_scope | max_steps | final_gate |
|---|---|---|---|---|
| quick | 1 | load_bearing | 8 | standard |
| standard | 3 | all | 16 | standard |
| thorough | 3 | all | 32 | standard |
| exhaustive | 5 | all | 64 | standard |

**Gate rules** (§5.6):
- **What gets checked:** the precondition ids **and all their transitive premises** (`view.premise_closure`). A conclusion is only as good as what it rests on; *that's the plan's call*.
- **Unknown id:** blocks, with reason `"{id}: unknown claim"`.
- **Terminal status:** blocks under every policy, with reason `"{id} is {status}"`.
- **`strict`:** acceptable means status `verified`, or type `given`.
- **`standard`:** acceptable means `verified`, `supported`, `critic_accepted`, `assumed`, or type `given`. Each `assumed` goes into `acted_on_assumptions`. `unchecked` blocks with reason `"{id} is unchecked"`.
- **`permissive`:** anything that isn't terminal is acceptable. Both `assumed` and `unchecked` go into `acted_on_assumptions`.
- **No preconditions:** blocks under `strict` and `standard`, with reason `"final answer cites no supporting claims"`. Allowed under `permissive`.
- `blocking` and `acted_on_assumptions` are in insertion order.

- [ ] **Step 1: Write the failing tests**

```python
def test_profiles_pinned():
    assert {n: (p.n_samples, p.check_scope, p.max_steps) for n, p in PROFILES.items()} == {
        "quick": (1, CheckScope.LOAD_BEARING, 8), "standard": (3, CheckScope.ALL, 16),
        "thorough": (3, CheckScope.ALL, 32), "exhaustive": (5, CheckScope.ALL, 64)}
def test_get_profile_unknown_lists_names():
    with pytest.raises(KeyError, match="quick"): get_profile("turbo")

# store fixture: b1 fact verified, b2 assumption (assumed), b3 inference(b1, b2) verified, b4 fact unchecked,
#                b5 inference(b4) verified, b6 fact contradicted
def test_standard_allows_assumed_premise_and_records_it():
    d = evaluate_gate(["b3"], s, GatePolicy.STANDARD)
    assert d.allowed and d.acted_on_assumptions == ["b2"]
def test_strict_blocks_assumed_premise():
    d = evaluate_gate(["b3"], s, GatePolicy.STRICT)
    assert not d.allowed and d.blocking == ["b2"]
def test_unchecked_premise_blocks_verified_conclusion_in_standard():
    d = evaluate_gate(["b5"], s, GatePolicy.STANDARD)
    assert not d.allowed and d.blocking == ["b4"] and d.reasons == ["b4 is unchecked"]
def test_terminal_blocks_even_permissive():
    assert not evaluate_gate(["b6"], s, GatePolicy.PERMISSIVE).allowed
def test_permissive_records_unchecked():
    assert evaluate_gate(["b5"], s, GatePolicy.PERMISSIVE).acted_on_assumptions == ["b4"]
def test_empty_preconditions():
    assert evaluate_gate([], s, GatePolicy.STANDARD).reasons == ["final answer cites no supporting claims"]
    assert evaluate_gate([], s, GatePolicy.PERMISSIVE).allowed
def test_unknown_id_blocks():
    assert evaluate_gate(["b99"], s, GatePolicy.PERMISSIVE).reasons == ["b99: unknown claim"]
```

- [ ] **Step 2: Run them and confirm they fail.** Run: `uv run pytest tests/loop -v`. Expected: FAIL (`ImportError`).
- [ ] **Step 3: Implement `task.py`, `profiles.py` and `gate.py`.**
- [ ] **Step 4: Run the tests and confirm they pass.** Run: `uv run pytest tests/loop -v && uv run pyright`. Expected: pass, `0 errors`.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: task model, effort profiles, final-answer gate"`

---

## Task 14: Context assembly, feedback and step prompts

**Files:**
- Create: `src/kengine/loop/context.py`, `src/kengine/loop/feedback.py`, `src/kengine/prompts/think.md`, `src/kengine/prompts/extract.md`
- Test: `tests/loop/test_context.py`, `tests/loop/test_feedback.py`, `tests/test_prompts.py`

**Interfaces:**
- Consumes: Tasks 6 (`load_prompt`), 9 and 13.
- Produces:
  - `belief_line(rec: BeliefRecord) -> str`. Format: `"{id} [{status}] ({type}) {text}"`. For computation claims, append `" :: {expression} = {claimed_result}"`.
  - `StepSummary(step: int, action_kind: str, note: str)`
  - `StepContext(messages: list[Message], included: list[str], dropped: list[str], prompt_tokens: int)`
  - `class ContextOverflowError(ValueError)`
  - `assemble_think_context(task: Task, view: BeliefView, feedback: Sequence[str], history: Sequence[StepSummary], budget_tokens: int, count_tokens: Callable[[str], int]) -> StepContext`
  - `assemble_extract_context(view: BeliefView, think_text: str, budget_tokens: int, count_tokens: Callable[[str], int]) -> StepContext`
  - `approx_tokens(text: str) -> int`, which is `ceil(len(text) / 3.5)`
  - `build_feedback(changes: Sequence[StatusChange], verdicts: Sequence[tuple[str, Verdict]], issues: Sequence[str], gate: GateDecision | None, max_items: int = 5) -> list[str]`

**THINK context** (§5.8): a system message from `think.md`, then one user message with these sections in order. Empty sections are left out.
1. `## Task`: the goal, plus `answer_format` if set
2. `## Documents`: `[i] text`
3. `## Problems found last step`: the feedback lines
4. `## Established`: beliefs with status `verified`, or of type `given`
5. `## Assumptions in play`: status `assumed`
6. `## Unchecked claims`: status `unchecked`, `supported` or `critic_accepted`
7. `## Your previous steps`: the last 5 `StepSummary` lines

Beliefs with a terminal status are never included.

**Budget rule:** the system message, Task, Documents and Problems are mandatory. If they alone exceed the budget, raise `ContextOverflowError`. Belief lines are then added in priority order (Established, then Assumptions, then Unchecked), newest first within each group, until the next line wouldn't fit. Lines that don't fit are recorded in `dropped`. History is added last and only if it fits. Lines are rendered in id order inside each section.

**EXTRACT context:** a system message from `extract.md`, then a user message containing:
- `## Existing claims you may reference`: every non-terminal belief line, newest first, within the budget
- `## Reasoning to convert`: `think_text`

**Feedback order and format** (§5.7). It's capped at `max_items`, in this order:
1. gate block reasons, as `"final answer blocked: {reason}"`
2. `contradicted` changes, as `"{id} contradicted: {explanation}"`, using the explanation from the verdict on the change
3. `invalidated`, one line per root cause, as `"{n} claim(s) invalidated because {root} is {status}"`
4. `fail` verdicts with no status change, as `"{id}: {explanation}"`
5. `unknown` verdicts, as `"{id}: {explanation}"`
6. `issues`, as they are

`pass` verdicts produce nothing.

**What the prompts must say.** Write the copy yourself; these points must all be covered.

`think.md` must tell the model to:
- reason freely
- refer to existing claims by their `b#` id and not restate them
- finish with a `Working:` section: a numbered list of one-sentence claims, with every assumption labelled `ASSUMPTION:` and every calculation written as `expression = result` using only numbers and operators
- end with either `Next: continue` or `Final answer: <answer>`, followed by `Rests on: <ids or item numbers>`

`extract.md` must:
- turn each numbered `Working` item into one claim (refs `c1`… in order)
- define the six extractable types in one line each
- put calculations in `computation`, with the expression using numbers and operators only
- point `premises` at `c#` or `b#`
- use `final_answer` only if the reasoning states one, with `preconditions` taken from `Rests on`
- include one short worked example: input `Working` → output JSON

- [ ] **Step 1: Write the failing tests**

```python
def test_think_sections_order_and_exclusions():
    # store: b1 given, b2 assumption, b3 fact unchecked, b4 fact contradicted
    ctx = assemble_think_context(task, s, ["b9 contradicted: x"], [], 10_000, approx_tokens)
    user = ctx.messages[-1].content
    assert user.index("## Task") < user.index("## Problems found last step") < user.index("## Established") \
        < user.index("## Assumptions in play") < user.index("## Unchecked claims")
    assert "b4" not in user and ctx.included == ["b1", "b2", "b3"]

def test_budget_drops_lowest_priority_oldest_first():
    # 1 established + 20 unchecked; budget fits mandatory + established + ~5 unchecked lines
    # -> b1 included; newest unchecked included; oldest unchecked in ctx.dropped; prompt_tokens <= budget

def test_mandatory_overflow_raises():
    with pytest.raises(ContextOverflowError): assemble_think_context(Task(goal="x" * 10_000), s, [], [], 100, approx_tokens)

def test_belief_line_format():
    assert belief_line(rec_comp) == "b2 [verified] (computation) Total cost :: 12*7 = 84"

def test_feedback_priority_and_cap():
    # inputs: gate block, 1 contradicted change, 2 invalidated (same root b1), 1 unknown verdict, 3 issues
    fb = build_feedback(changes, verdicts, issues, gate, max_items=5)
    assert fb[0] == "final answer blocked: b4 is unchecked"
    assert fb[1] == "b1 contradicted: `12*7` evaluates to 84, not 82"
    assert fb[2] == "2 claim(s) invalidated because b1 is contradicted"
    assert len(fb) == 5

def test_prompts_load_and_contain_required_markers():
    t, e = load_prompt("think"), load_prompt("extract")
    assert all(m in t for m in ["Working:", "ASSUMPTION:", "Final answer:", "Rests on:", "Next: continue"])
    assert all(m in e for m in ["computation", "premises", "preconditions", "final_answer"])
```

- [ ] **Step 2: Run them and confirm they fail.** Run: `uv run pytest tests/loop/test_context.py tests/loop/test_feedback.py tests/test_prompts.py -v`. Expected: FAIL (`ImportError`).
- [ ] **Step 3: Implement `context.py`, `feedback.py`, `think.md` and `extract.md`.**
- [ ] **Step 4: Run the tests and confirm they pass.** Run: `uv run pytest tests/loop tests/test_prompts.py -v && uv run pyright`. Expected: pass, `0 errors`.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: budgeted context assembly, feedback, step prompts"`

---

## Task 15: Agent loop

**Files:**
- Create: `src/kengine/loop/agent.py`
- Test: `tests/loop/test_agent.py`, using `ScriptedBackend` only

**Interfaces:**
- Consumes: everything from Tasks 2–14.
- Produces:
  - `Limits(max_steps: int | None = None, max_completion_tokens: int | None = None, max_wall_s: float | None = None)`
  - `RunStatus = Literal["answered","blocked","partial","budget_exhausted","error"]`
  - `RunResult(status: RunStatus, answer: str | None, blocked_answer: str | None, session_dir: Path, steps: int, model_calls: int, prompt_tokens: int, completion_tokens: int, wall_s: float, acted_on_assumptions: list[str], error: str | None = None)`
  - `Agent(backend: ModelBackend, roles: RolesConfig, profile: Profile, context_budget: int, checkers: Sequence[Checker], session: Session, limits: Limits = Limits(), clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep, retry_delay_s: float = 5.0)`
  - `Agent.run(self, task: Task) -> RunResult`

**Step algorithm** (§5.2). Every stage logs its event (Task 9).

0. Log `session_started` at the beginning. Before each step, check the limits:
   - **steps:** `min(profile.max_steps, limits.max_steps)`
   - **completion tokens:** the running total of completion tokens
   - **wall clock:** time since `run` began

   If a limit is exceeded, stop. The status is `blocked` if the last action was a blocked final answer, otherwise `budget_exhausted`.
1. **THINK.** Use `assemble_think_context(..., count_tokens=approx_tokens)`, then `backend.complete(messages, roles.think)`.
   - If `content.strip() == ""` and `finish_reason == "length"`: add the issue `"your reasoning was cut off before the Working section; be more concise"`, skip to step 7, and don't extract (Review Focus #2).
2. **EXTRACT.** Use `assemble_extract_context`, then make `profile.n_samples` calls to `complete_structured(..., roles.extract, Extraction)`. A `StructuredOutputError` logs `extraction(ok=False)` and that sample is skipped. If no samples are valid: add the issue `"could not turn your reasoning into claims; keep the Working list short and numbered"` and skip to step 7 (Review Focus #1).
3. **VOTE.** Call `vote(samples, default_k(len(valid_samples)))`.
4. **INSERT.**
   - Map each kept ref to a new belief id, inserting in an order where every premise is already inserted.
   - Premises that can't be resolved (an unknown `b#`, or a `c#` that's dropped or only reachable through a cycle) are removed, with the issue `"{ref} cited unknown {p}; link removed"`.
   - Source is `Source(kind=model, ref=f"step:{n}")`.
   - Preconditions are translated the same way.
5. **CHECK.**
   - The scope is all new beliefs (`all`), or the new beliefs inside `premise_closure(action preconditions)` (`load_bearing`).
   - Run each checker whose `applies_to(claim)` is true, in the order given, and call `apply_verdict` for each.
   - Once a belief becomes terminal, later checkers skip it.
6. **ACT.**
   - **`final_answer` with an empty answer:** issue `"final answer had no text"`.
   - **Otherwise `final_answer`:** call `evaluate_gate(preconditions, store, profile.final_gate)` and log it. If allowed, finish with `answered` and `acted_on_assumptions` from the decision. If not, remember `blocked_answer`.
   - **`continue`:** no action.
7. **FEEDBACK.** Call `build_feedback(...)` and log it. A `StepSummary` is added to the history.
   - **Loop detection:** if the feedback, with every `b\d+` replaced by `b#`, is non-empty and identical as a set for 3 consecutive steps, stop with `partial`. If a blocked answer exists, the status is `blocked` instead.
8. **Backend failures.** A `BackendError` that isn't a `StructuredOutputError`, raised by THINK or by every EXTRACT call, gets one retry after `sleep(retry_delay_s)`. If it fails again, finish with `error` and the message; the session stays intact (Review Focus #3).

Log `run_finished` with the result in every exit path. Token and call counts add up the usage from every completion, including failed structured calls whose completion is available.

- [ ] **Step 1: Write the failing tests**

`think(...)` and `extract_json(...)` below are small test helpers that build `ScriptedReply` objects.

```python
def test_happy_path_verified_computation_answers(tmp_path):
    script = [think("Working:\n1. 12*7 = 84\nFinal answer: 84\nRests on: 1"),
              *[extract_json(claims=[comp("c1", "12*7", "84")], action=final("84", ["c1"]))] * 3]
    r = make_agent(script, profile="standard", tmp=tmp_path).run(Task(goal="12*7?"))
    assert (r.status, r.answer, r.steps, r.model_calls) == ("answered", "84", 1, 4)
    assert Session.open(r.session_dir).store.get("b1").status == Status.VERIFIED

def test_wrong_arithmetic_is_contradicted_and_fed_back(tmp_path):
    # step 1: claims 12*7 = 82, final 82 -> contradicted -> gate blocks
    # step 2 THINK prompt contains "b1 contradicted: `12*7` evaluates to 84, not 82"
    #   and "final answer blocked: b1 is contradicted"
    # step 2 claims 12*7 = 84 -> answered "84"; r.steps == 2

def test_assumption_answer_recorded_under_standard(tmp_path):
    # assumption c1 + inference c2(c1), final rests on c2 -> answered; acted_on_assumptions == ["b1"]

def test_all_extractions_invalid_gives_feedback_and_continues(tmp_path):   # Review Focus #1
    # step 1: think, 3 malformed extraction replies; step 2: valid -> answered;
    # step-2 think prompt contains "could not turn your reasoning into claims"

def test_think_cutoff_skips_extraction(tmp_path):                         # Review Focus #2
    # step 1 think reply content "", finish_reason "length"; next scripted call must be step-2 THINK
    # (assert backend.calls[1].schema_name is None) and its prompt contains "reasoning was cut off"

def test_backend_failure_retries_once_then_errors(tmp_path):               # Review Focus #3
    # script: error, error -> r.status == "error"; sleep called once with 5.0;
    # events.jsonl ends with run_finished; beliefs.sqlite exists

def test_backend_failure_recovers_after_retry(tmp_path):
    # script: error, then a normal happy path -> answered

def test_loop_detection_stops_partial(tmp_path):
    # 3 steps each producing an inference with no premises and action continue
    # -> r.status == "partial", r.steps == 3

def test_step_limit_budget_exhausted(tmp_path):
    # quick profile, Limits(max_steps=2), model keeps saying continue -> "budget_exhausted", steps == 2

def test_blocked_answer_reported_when_limit_hit(tmp_path):
    # max_steps=1, final answer resting on an unchecked fact -> status "blocked",
    # answer None, blocked_answer == "Paris"

def test_load_bearing_scope_only_checks_answer_closure(tmp_path):
    # quick profile: claims c1 comp 2+2=4 (unrelated), c2 comp 3*3=9 (precondition) ->
    # b2 VERIFIED, b1 stays UNCHECKED

def test_unresolvable_premise_link_removed_with_issue(tmp_path):
    # extraction cites premise "b42" that doesn't exist -> belief inserted without it;
    # next think prompt contains "cited unknown b42; link removed"
```

- [ ] **Step 2: Run them and confirm they fail.** Run: `uv run pytest tests/loop/test_agent.py -v`. Expected: FAIL (`ImportError`).
- [ ] **Step 3: Implement `Agent`.** Keep `run` as a short driver over private methods, one per stage: `_think`, `_extract`, `_insert`, `_check`, `_act`, `_feedback`.
- [ ] **Step 4: Run the tests and confirm they pass.** Run: `uv run pytest -v && uv run pyright && uv run ruff check`. Expected: all pass, `0 errors`.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: kengine agent loop"`

---

## Task 16: kengine solver, `kengine run`, LLM smoke test and runbook checkpoint B

**Files:**
- Create: `src/kengine/eval/kengine_solver.py`, `tests/integration/__init__.py`, `tests/integration/test_llm_smoke.py`
- Modify: `src/kengine/cli/main.py` (wire up `--solver kengine`; add `run`), `docs/real-machine-runbook.md` (add Checkpoint B)
- Test: `tests/eval/test_kengine_solver.py`, `tests/cli/test_cli.py`

**Interfaces:**
- Consumes: `Agent`, `RunResult`, `Limits` (Task 15), `Solver`, `SolveResult` (Task 4), `get_profile` (Task 13), `StructureChecker`, `ArithmeticChecker`.
- Produces:
  - `default_checkers() -> list[Checker]`, which returns `[StructureChecker(), ArithmeticChecker()]`, in `checkers/__init__.py`
  - `KengineSolver(backend: ModelBackend, config: KengineConfig, profile: Profile, sessions_root: Path, token_budget: int)`, with `name = "kengine"`
  - CLI: `kengine run TASK --profile NAME [--max-steps N] [--max-tokens N] [--max-wall-s S]`

**Status mapping:**

| RunStatus | SolveStatus |
|---|---|
| answered | answered |
| blocked | blocked |
| partial | partial |
| budget_exhausted | partial |
| error | error |

- `transcript_excerpt` holds the last THINK content (up to 2000 characters), then `\n---\n`, then the last feedback items.
- Eval sessions go under `<out_dir>/sessions/`. `kengine run` sessions go under `config.data_dir / "sessions"`.
- `kengine run` prints:
  - the status
  - the answer, or `blocked answer (unverified): ...`
  - each acted-on assumption, as `ASSUMED: {id} {text}`
  - the session directory

  It exits 0 for `answered` and 3 for anything else.

- [ ] **Step 1: Write the failing tests**

```python
def test_solver_maps_status_and_passes_budget(tmp_path):
    # ScriptedBackend happy path -> SolveResult(status="answered", answer="84", model_calls=4)
    # session_dir is under tmp_path / "sessions"
def test_solver_budget_exhausted_maps_to_partial(tmp_path): ...
def test_cli_eval_kengine_writes_results(monkeypatch, tmp_path):
    # -> dir "*-gsm8k-kengine-smoke-standard" with metrics.json
def test_cli_run_prints_answer_and_assumptions(monkeypatch, tmp_path):
    # answered with acted_on_assumptions ["b1"] -> exit 0; output contains "ASSUMED: b1"
def test_cli_run_blocked_exit_code(monkeypatch, tmp_path):
    # -> exit 3; output contains "blocked answer (unverified): Paris"
```

`tests/integration/test_llm_smoke.py`, marked `llm`, is skipped unless `KENGINE_TEST_LLM_URL` is set. It runs `Agent` with `LlamaCppBackend` and the `quick` profile on `"What is 17 * 23?"`, and asserts:
- `result.status` is one of the five `RunStatus` values
- the first event in `events.jsonl` is `session_started` and the last is `run_finished`

It doesn't assert correctness, because it's meant to run against a tiny CPU model.

- [ ] **Step 2: Run them and confirm they fail.** Run: `uv run pytest tests/eval/test_kengine_solver.py tests/cli -v`. Expected: FAIL (`ImportError` / exit code 2).
- [ ] **Step 3: Implement `KengineSolver`, `default_checkers` and the CLI wiring.**
- [ ] **Step 4: Run the tests and confirm they pass.** Run: `uv run pytest -v && uv run pyright && uv run ruff check`. Expected: all pass (the `llm` test deselected), `0 errors`.
- [ ] **Step 5: Optional dev smoke test.** If a llama-server with a tiny model is running locally, run `KENGINE_TEST_LLM_URL=http://127.0.0.1:8080 uv run pytest -m llm -v`. Expected: PASS. If no server is available, skip this step and say so in the task report.
- [ ] **Step 6: Add Checkpoint B to `docs/real-machine-runbook.md`**
  1. `git pull && uv sync --extra eval && kengine doctor`.
  2. Run `kengine run "A shop sells pens at $1.25. How much do 12 pens cost?" --profile standard`. Record the output, and check that the `beliefs.sqlite` and `events.jsonl` files in the session directory look sensible.
  3. For each suite in gsm8k, folio, hotpotqa: run `kengine eval <suite> --solver kengine --size small --profile quick`, then `--profile standard`.
  4. Write `eval/results/CHECKPOINT-B.md`. It needs:
     - a table of headline accuracy, abstention rate and answered accuracy per suite and profile, compared with the Checkpoint A baseline
     - mean wall-clock time per item
     - 3 example audit trails (one helped, one hurt, one blocked), as excerpts from `events.jsonl`
  5. Commit everything except `sessions/` and push.
- [ ] **Step 7: Commit, and tell the owner Checkpoint B can run**

```bash
git add -A && git commit -m "feat: kengine eval solver, run command, runbook checkpoint B"
```

---

## After this plan

- **Owner review point 1** uses Checkpoint A's error taxonomy (§7). It may change the order of M2 and M3, but not the scope.
- **Plan 2** (M2: the formal fragment, the Z3 logic checker and counterexample rendering) gets written after that review. It builds on `Claim`, `Checker`, `BeliefStore` and `Agent` from this plan without changing their interfaces. It adds a `formal` field to `Claim` and `ExtractedClaim`.
- **Owner review point 2** follows Plan 2's measured with-vs-without comparison.
