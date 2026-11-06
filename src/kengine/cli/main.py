"""The `kengine` command."""

from __future__ import annotations

import datetime
import json
import random
import subprocess
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

import typer

from kengine.audit.session import Session
from kengine.backend import make_backend
from kengine.backend.types import ModelBackend
from kengine.checkers import task_checkers
from kengine.claims.formal import FormalSyntax
from kengine.cli.doctor import nvidia_smi, render_report, run_doctor
from kengine.config import BackendConfig, KengineConfig, default_config_paths, load_config
from kengine.eval import suites
from kengine.eval.adapters import gsm8k as gsm8k_adapter
from kengine.eval.baseline import BaselineSolver
from kengine.eval.compare import compare_runs
from kengine.eval.kengine_solver import KengineSolver
from kengine.eval.probe_formal import run_probe
from kengine.eval.runner import merge_shards, rescore, results_dir_name, run_eval, shard_dir_name
from kengine.eval.taxonomy import summarize_taxonomy
from kengine.eval.types import RunMetrics, Size, Solver, SuiteName
from kengine.finetune.chains import build_chain_pairs
from kengine.finetune.data import build_pairs, dev_items, distill_gsm8k
from kengine.loop.agent import Agent, Limits
from kengine.loop.profiles import get_profile
from kengine.loop.task import JUDGE_FORMAT, Task, judge_question, split_sentences
from kengine.retrieval.models import Models, ModelsMissing, fetch_models, load_models
from kengine.retrieval.web import SearxngSearch

app = typer.Typer(no_args_is_help=True, help="kengine: check a small model's reasoning.")

# Module-level so tests can swap in a scripted backend.
backend_factory: Callable[[BackendConfig], ModelBackend] = make_backend


def _config() -> tuple[KengineConfig, list[Path]]:
    paths = default_config_paths()
    return load_config(paths), paths


def _models(cfg: KengineConfig) -> Models | None:
    if not cfg.facts.enabled:
        return None
    try:
        return load_models(cfg.models_dir)
    except ModelsMissing as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


@app.command()
def doctor(
    json_path: Annotated[
        Path | None, typer.Option("--json", help="Also write the report as JSON here.")
    ] = None,
) -> None:
    """Check the backend, structured output, thinking separation and GPU."""
    cfg, paths = _config()
    rep = run_doctor(cfg, paths, backend_factory(cfg.backend), nvidia_smi)
    typer.echo(render_report(rep))
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(rep.model_dump_json(indent=2) + "\n", encoding="utf-8")
    raise typer.Exit(0 if rep.ok else 1)


@app.command("fetch-models")
def fetch_models_cmd() -> None:
    """Download the entailment and embedding models (about 870 MB) for the fact checks."""
    cfg, _ = _config()
    for path in fetch_models(cfg.models_dir):
        typer.echo(str(path))


@app.command("eval")
def eval_cmd(
    suite: Annotated[SuiteName, typer.Argument(help="gsm8k | hotpotqa | folio")],
    solver: Annotated[Literal["baseline", "kengine"], typer.Option()] = "baseline",
    size: Annotated[Size, typer.Option()] = "smoke",
    profile: Annotated[
        str | None, typer.Option(help="Effort profile (kengine solver only).")
    ] = None,
    seed: Annotated[int, typer.Option()] = 0,
    out: Annotated[Path, typer.Option(help="Parent directory for results.")] = Path("eval/results"),
    limit: Annotated[int | None, typer.Option(min=1, help="Only the first N items.")] = None,
    shard: Annotated[
        str | None,
        typer.Option(help="K/N: run items K, K+N, ... into a shard-KofN subdir; see eval-merge."),
    ] = None,
    split: Annotated[
        Literal["validation", "train-dev"],
        typer.Option(help="train-dev: FOLIO-train's held-back stories (folio only; ignores size)"),
    ] = "validation",
    no_facts: Annotated[
        bool,
        typer.Option(
            "--no-facts", help="Run without the M3 fact, consistency and inference checks."
        ),
    ] = False,
) -> None:
    """Run a suite; results go to OUT/<date>-<suite>-<solver>-<size>-<profile>[-firstN]."""
    if split == "train-dev" and suite != "folio":
        typer.echo("--split train-dev applies to folio only", err=True)
        raise typer.Exit(2)
    cfg, _ = _config()
    if no_facts:
        cfg = cfg.model_copy(update={"facts": cfg.facts.model_copy(update={"enabled": False})})
    k, n = _parse_shard(shard) if shard is not None else (1, 1)
    profile_name = "none"
    if solver == "kengine":
        profile_name = profile or cfg.default_profile
        _profile_or_exit(profile_name)
    elif profile is not None:
        typer.echo("--profile applies to the kengine solver only", err=True)
        raise typer.Exit(2)
    where = "train-dev" if split == "train-dev" else size
    out_dir = out / results_dir_name(
        datetime.date.today(), suite, solver, where, profile_name, limit
    )
    if shard is not None:
        out_dir = out_dir / shard_dir_name(k, n)
    backend = backend_factory(cfg.backend)
    budget = cfg.eval.token_budget_per_task
    the_solver: Solver
    if solver == "kengine":
        the_solver = KengineSolver(
            backend,
            cfg,
            get_profile(profile_name),
            out_dir / "sessions",
            budget,
            models=_models(cfg),
        )
    else:
        the_solver = BaselineSolver(backend, cfg.roles.baseline, budget)
    health = backend.health()
    if split == "train-dev":
        items = dev_items(suites.load_rows("tasksource/folio", None, "train"))
    else:
        items = suites.load_items(suite, size, seed)
    if limit is not None:
        items = items[:limit]
    items = items[k - 1 :: n]
    run_config: dict[str, object] = {
        "suite": suite,
        "solver": solver,
        "size": where,
        "profile": profile_name,
        "seed": seed,
        "limit": limit,
        "n_items": len(items),
        "git_commit": _git_commit(),
        "backend": cfg.backend.model_dump(mode="json"),
        "loaded_model": health.model,
        "n_ctx": health.n_ctx,
        "roles": cfg.roles.model_dump(mode="json"),
        "token_budget_per_task": cfg.eval.token_budget_per_task,
    }
    if shard is not None:
        run_config["shard"] = f"{k}/{n}"
    typer.echo(f"{suite}/{solver}/{size}: {len(items)} items -> {out_dir}")
    # A shard keeps every failure, so eval-merge can draw the same sample an unsharded run would.
    sample = len(items) if shard is not None else 25
    m = run_eval(suites.SUITES[suite], items, the_solver, out_dir, run_config, sample, seed=seed)
    typer.echo(_metrics_line(m))


def _parse_shard(shard: str) -> tuple[int, int]:
    k, _, n = shard.partition("/")
    if not (k.isdigit() and n.isdigit() and 1 <= int(k) <= int(n)):
        typer.echo(f"--shard must be K/N with 1 <= K <= N, got {shard!r}", err=True)
        raise typer.Exit(2)
    return int(k), int(n)


@app.command("eval-merge")
def eval_merge(
    results_dir: Annotated[Path, typer.Argument(help="A results dir holding shard-KofN runs")],
) -> None:
    """Join RESULTS_DIR's finished shards into one run in RESULTS_DIR."""
    first = min(results_dir.glob("shard-*of*"), default=None)
    if first is None:
        typer.echo(f"no shard-KofN dirs in {results_dir}", err=True)
        raise typer.Exit(2)
    config = json.loads((first / "config.json").read_text(encoding="utf-8"))
    try:
        m = merge_shards(suites.SUITES[config["suite"]], results_dir, seed=int(config["seed"]))
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(_metrics_line(m))


def _metrics_line(m: RunMetrics) -> str:
    answered = "n/a" if m.answered_accuracy is None else f"{m.answered_accuracy:.3f}"
    return (
        f"n={m.n} headline={m.headline_accuracy:.3f} exact={m.exact_accuracy:.3f} "
        f"abstention={m.abstention_rate:.3f} answered_acc={answered} "
        f"statuses={m.status_counts} mean_wall_s={m.mean_wall_s:.1f}"
    )


@app.command("eval-rescore")
def eval_rescore(
    results_dir: Annotated[Path, typer.Argument(help="A finished results dir")],
) -> None:
    """Re-score RESULTS_DIR's answers with the suite's current scorer."""
    suite = json.loads((results_dir / "config.json").read_text(encoding="utf-8"))["suite"]
    typer.echo(_metrics_line(rescore(suites.SUITES[suite], results_dir)))


def _profile_or_exit(name: str) -> None:
    try:
        get_profile(name)
    except KeyError as e:
        typer.echo(str(e.args[0]), err=True)
        raise typer.Exit(2) from None


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="What to solve.")],
    profile: Annotated[
        str | None, typer.Option(help="quick | standard | thorough | exhaustive")
    ] = None,
    max_steps: Annotated[int | None, typer.Option(min=1)] = None,
    max_tokens: Annotated[
        int | None, typer.Option(min=1, help="Completion-token budget for the whole run.")
    ] = None,
    max_wall_s: Annotated[float | None, typer.Option(min=0)] = None,
    conclusion: Annotated[
        str | None,
        typer.Option(
            help="Judge this statement True, False or Uncertain from TASK's sentences, "
            "and check the answer against it."
        ),
    ] = None,
    doc: Annotated[
        list[Path] | None,
        typer.Option(help="A document to check facts against; repeatable."),
    ] = None,
) -> None:
    """Solve TASK with the agent loop. Exits 0 when answered, 3 otherwise."""
    cfg, _ = _config()
    profile_name = profile or cfg.default_profile
    _profile_or_exit(profile_name)
    documents = [f"{p.name}: {p.read_text()}" for p in doc or []]
    web = SearxngSearch(cfg.facts.searxng_url) if cfg.facts.searxng_url else None
    checkers = task_checkers(
        documents, has_givens=True, models=_models(cfg), facts=cfg.facts, web=web
    )
    session = Session.create(cfg.data_dir / "sessions")
    try:
        agent = Agent(
            backend=backend_factory(cfg.backend),
            roles=cfg.roles,
            profile=get_profile(profile_name),
            context_budget=cfg.context.step_prompt_tokens,
            checkers=checkers,
            session=session,
            limits=Limits(
                max_steps=max_steps, max_completion_tokens=max_tokens, max_wall_s=max_wall_s
            ),
            formal_syntax=cfg.logic.syntax,
            formalize_givens=cfg.logic.formalize_givens,
            answer_from_givens=cfg.logic.answer_from_givens,
        )
        givens = split_sentences(task)
        if conclusion is None:
            r = agent.run(Task(goal=task, documents=documents, givens=givens))
        else:
            r = agent.run(
                Task(
                    goal=judge_question(givens, conclusion),
                    answer_format=JUDGE_FORMAT,
                    documents=documents,
                    givens=givens,
                    conclusion=conclusion,
                )
            )
        typer.echo(f"status: {r.status}")
        if r.answer is not None:
            typer.echo(f"answer: {r.answer}")
        elif r.blocked_answer is not None:
            typer.echo(f"blocked answer (unverified): {r.blocked_answer}")
        if r.error is not None:
            typer.echo(f"error: {r.error}")
        for id in r.acted_on_assumptions:
            typer.echo(f"ASSUMED: {id} {session.store.get(id).claim.text}")
        typer.echo(f"session: {r.session_dir}")
    finally:
        session.close()
    raise typer.Exit(0 if r.status == "answered" else 3)


@app.command("eval-compare")
def eval_compare(
    base: Annotated[Path, typer.Argument(help="Results dir of the reference run (baseline)")],
    other: Annotated[Path, typer.Argument(help="Results dir to compare with it")],
) -> None:
    """Compare two runs item by item: errors caught, fixed, broken and false alarms."""
    c = compare_runs(base, other)
    typer.echo(c.line())
    out = other / f"compare-{base.name}.json"
    out.write_text(c.model_dump_json(indent=2) + "\n", encoding="utf-8")
    typer.echo(f"wrote {out}")


@app.command("finetune-data")
def finetune_data(
    out: Annotated[Path, typer.Option()] = Path("finetune/data"),
    gsm8k: Annotated[
        int, typer.Option(min=0, help="gsm8k-train items to translate with today's model")
    ] = 100,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Write FORMALIZE training pairs: FOLIO-train's expert logic, plus gsm8k-train items
    answered by the current model."""
    rows = suites.load_rows("tasksource/folio", None, "train")
    pairs, dropped = build_pairs(rows)
    if gsm8k:
        cfg, _ = _config()
        math = suites.load_rows("openai/gsm8k", "main", "train")
        picked = random.Random(seed).sample(range(len(math)), gsm8k)
        items = [gsm8k_adapter.parse_row(math[i], i) for i in picked]
        pairs += distill_gsm8k(backend_factory(cfg.backend), cfg.roles.formalize, items)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        lines = [p.model_dump_json() + "\n" for p in pairs if p.split == split]
        (out / f"{split}.jsonl").write_text("".join(lines), encoding="utf-8")
    report = {
        "folio_rows": len(rows),
        "kept": {s: sum(p.split == s and p.label != "" for p in pairs) for s in ("train", "dev")},
        "gsm8k": sum(p.label == "" for p in pairs),
        "labels": dict(Counter(p.label for p in pairs if p.label)),
        "dropped": dict(dropped.most_common()),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(report))


@app.command("finetune-chains")
def finetune_chains(
    out: Annotated[Path, typer.Option()] = Path("finetune/data-chains"),
    uncertain_percent: Annotated[
        int, typer.Option(min=0, max=100, help="share of the Uncertain train rows to keep")
    ] = 100,
) -> None:
    """Write THINK and extract training pairs: worked solutions to FOLIO-train rows, built
    from the expert logic and checked by the logic checker."""
    cfg, _ = _config()
    rows = suites.load_rows("tasksource/folio", None, "train")
    pairs, dropped = build_chain_pairs(
        rows, cfg.context.step_prompt_tokens, uncertain_percent=uncertain_percent
    )
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        lines = [p.model_dump_json() + "\n" for p in pairs if p.split == split]
        (out / f"{split}.jsonl").write_text("".join(lines), encoding="utf-8")
    think = [p for p in pairs if p.role == "think"]
    extract = [p for p in pairs if p.role == "extract"]
    report = {
        "folio_rows": len(rows),
        "kept": {s: sum(p.split == s for p in think) for s in ("train", "dev")},
        "labels": dict(Counter(p.label for p in think)),
        "steps": dict(
            sorted(Counter(len(json.loads(p.completion)["claims"]) for p in extract).items())
        ),
        "dropped": dict(dropped.most_common()),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(report))


@app.command("probe-formal")
def probe_formal(
    syntax: Annotated[FormalSyntax, typer.Option(help="text | json")] = "text",
    size: Annotated[Size, typer.Option()] = "small",
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    seed: Annotated[int, typer.Option()] = 0,
    split: Annotated[
        Literal["validation", "train-dev"],
        typer.Option(help="train-dev: FOLIO-train's held-back stories (all of them; ignores size)"),
    ] = "validation",
    out: Annotated[Path, typer.Option()] = Path("eval/results"),
) -> None:
    """Formalize FOLIO items in SYNTAX and score the Z3 verdicts (spec §9)."""
    cfg, _ = _config()
    if split == "train-dev":
        items = dev_items(suites.load_rows("tasksource/folio", None, "train"))
    else:
        items = suites.load_items("folio", size, seed)
    if limit is not None:
        items = items[:limit]
    where = "train-dev" if split == "train-dev" else size
    name = f"{datetime.date.today().isoformat()}-folio-probe-{syntax}-{where}"
    out_dir = out / (name + (f"-first{limit}" if limit is not None else ""))
    s = run_probe(backend_factory(cfg.backend), cfg.roles.formalize, syntax, items, out_dir)
    typer.echo(
        f"{syntax}: n={s.n} all_formalized={s.all_formalized} decided={s.decided} "
        f"correct={s.correct} mean_wall_s={s.mean_wall_s:.1f} -> {out_dir}"
    )


@app.command("eval-summarize")
def eval_summarize(
    results_dir: Annotated[Path, typer.Argument(help="A results dir containing taxonomy.jsonl")],
) -> None:
    """Count the hand-assigned error labels in RESULTS_DIR/taxonomy.jsonl."""
    counts = summarize_taxonomy(results_dir / "taxonomy.jsonl")
    total = sum(counts.values())
    for label, n in counts.items():
        typer.echo(f"{n:4d}  {n / total:6.1%}  {label}")
    out = results_dir / "taxonomy_summary.json"
    out.write_text(json.dumps({str(k): v for k, v in counts.items()}, indent=2) + "\n")
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
