"""The `kengine` command."""

from __future__ import annotations

import datetime
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

import typer

from kengine.audit.session import Session
from kengine.backend import make_backend
from kengine.backend.types import ModelBackend
from kengine.checkers import default_checkers
from kengine.cli.doctor import nvidia_smi, render_report, run_doctor
from kengine.config import BackendConfig, KengineConfig, default_config_paths, load_config
from kengine.eval import suites
from kengine.eval.baseline import BaselineSolver
from kengine.eval.kengine_solver import KengineSolver
from kengine.eval.runner import results_dir_name, run_eval
from kengine.eval.taxonomy import summarize_taxonomy
from kengine.eval.types import Size, Solver, SuiteName
from kengine.loop.agent import Agent, Limits
from kengine.loop.profiles import get_profile
from kengine.loop.task import Task

app = typer.Typer(no_args_is_help=True, help="kengine: check a small model's reasoning.")

# Module-level so tests can swap in a scripted backend.
backend_factory: Callable[[BackendConfig], ModelBackend] = make_backend


def _config() -> tuple[KengineConfig, list[Path]]:
    paths = default_config_paths()
    return load_config(paths), paths


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
) -> None:
    """Run a suite; results go to OUT/<date>-<suite>-<solver>-<size>-<profile>."""
    cfg, _ = _config()
    profile_name = "none"
    if solver == "kengine":
        profile_name = profile or cfg.default_profile
        _profile_or_exit(profile_name)
    elif profile is not None:
        typer.echo("--profile applies to the kengine solver only", err=True)
        raise typer.Exit(2)
    out_dir = out / results_dir_name(datetime.date.today(), suite, solver, size, profile_name)
    backend = backend_factory(cfg.backend)
    budget = cfg.eval.token_budget_per_task
    the_solver: Solver
    if solver == "kengine":
        the_solver = KengineSolver(
            backend, cfg, get_profile(profile_name), out_dir / "sessions", budget
        )
    else:
        the_solver = BaselineSolver(backend, cfg.roles.baseline, budget)
    health = backend.health()
    items = suites.load_items(suite, size, seed)
    run_config: dict[str, object] = {
        "suite": suite,
        "solver": solver,
        "size": size,
        "profile": profile_name,
        "seed": seed,
        "n_items": len(items),
        "git_commit": _git_commit(),
        "backend": cfg.backend.model_dump(mode="json"),
        "loaded_model": health.model,
        "n_ctx": health.n_ctx,
        "roles": cfg.roles.model_dump(mode="json"),
        "token_budget_per_task": cfg.eval.token_budget_per_task,
    }
    typer.echo(f"{suite}/{solver}/{size}: {len(items)} items -> {out_dir}")
    m = run_eval(suites.SUITES[suite], items, the_solver, out_dir, run_config, seed=seed)
    answered = "n/a" if m.answered_accuracy is None else f"{m.answered_accuracy:.3f}"
    typer.echo(
        f"n={m.n} headline={m.headline_accuracy:.3f} abstention={m.abstention_rate:.3f} "
        f"answered_acc={answered} statuses={m.status_counts} "
        f"mean_wall_s={m.mean_wall_s:.1f}"
    )


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
) -> None:
    """Solve TASK with the agent loop. Exits 0 when answered, 3 otherwise."""
    cfg, _ = _config()
    profile_name = profile or cfg.default_profile
    _profile_or_exit(profile_name)
    session = Session.create(cfg.data_dir / "sessions")
    try:
        agent = Agent(
            backend=backend_factory(cfg.backend),
            roles=cfg.roles,
            profile=get_profile(profile_name),
            context_budget=cfg.context.step_prompt_tokens,
            checkers=default_checkers(),
            session=session,
            limits=Limits(
                max_steps=max_steps, max_completion_tokens=max_tokens, max_wall_s=max_wall_s
            ),
        )
        r = agent.run(Task(goal=task))
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
