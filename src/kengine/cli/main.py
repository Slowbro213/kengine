"""The `kengine` command."""

from __future__ import annotations

import datetime
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

import typer

from kengine.backend import make_backend
from kengine.backend.types import ModelBackend
from kengine.cli.doctor import nvidia_smi, render_report, run_doctor
from kengine.config import BackendConfig, KengineConfig, default_config_paths, load_config
from kengine.eval import suites
from kengine.eval.baseline import BaselineSolver
from kengine.eval.runner import results_dir_name, run_eval
from kengine.eval.taxonomy import summarize_taxonomy
from kengine.eval.types import Size, Solver, SuiteName

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
    if solver == "kengine":
        typer.echo("--solver kengine is available after Task 16", err=True)
        raise typer.Exit(2)
    profile_name = "none"
    backend = backend_factory(cfg.backend)
    the_solver: Solver = BaselineSolver(backend, cfg.roles.baseline, cfg.eval.token_budget_per_task)
    health = backend.health()
    items = suites.load_items(suite, size, seed)
    out_dir = out / results_dir_name(datetime.date.today(), suite, solver, size, profile_name)
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
