from __future__ import annotations

import subprocess
from enum import Enum
from pathlib import Path

import httpx
import openai
import typer

from .config import load_config, resolve_model
from .ollama_client import OllamaClient
from .review.harness import resolve_tier, review_harness
from .tools import RepoTools

app = typer.Typer(add_completion=False, help="Cost-free Ollama code review.")


class OutputFormat(str, Enum):
    md = "md"
    json = "json"


class Effort(str, Enum):
    low = "low"
    med = "med"
    high = "high"


def client_factory(spec, timeout: float):
    return OllamaClient(
        base_url=spec.base_url,
        model=spec.ollama_model,
        num_ctx=spec.num_ctx,
        timeout=timeout,
    )


@app.command()
def main(
    base: str = typer.Option("main", help="Base ref to diff against."),
    head: str = typer.Option("HEAD", help="Head ref."),
    repo: Path = typer.Option(Path("."), help="Repository root."),
    paths: list[str] = typer.Option(None, help="Limit diff to these paths."),
    context_file: Path = typer.Option(None, help="Memory context file to inject."),
    model: str = typer.Option(None, help="Model alias (see config.toml)."),
    local: bool = typer.Option(False, help="Use the local model alias."),
    fmt: OutputFormat = typer.Option(OutputFormat.md, "--format", help="Output format: md|json."),
    out: Path = typer.Option(None, help="Write the report to this file."),
    timeout: float = typer.Option(120.0, help="Per-call timeout (seconds)."),
    max_iters: int = typer.Option(12, help="Max agent iterations."),
    effort: Effort = typer.Option(Effort.med, "--effort", help="Review effort: low|med|high."),
    doctor: bool = typer.Option(False, "--doctor", help="Check ollama daemon and model availability, then exit."),
) -> None:
    spec = resolve_model(model, local=local)

    if doctor:
        from . import doctor as _doctor
        results = [_doctor.check_ollama(spec.base_url), _doctor.check_model(spec.base_url, spec.ollama_model)]
        for r in results:
            mark = "OK " if r.ok else "FAIL"
            typer.echo(f"[{mark}] {r.name}: {r.detail}")
        raise typer.Exit(code=0 if all(r.ok for r in results) else 1)

    tools = RepoTools(repo)
    try:
        diff = tools.git_diff(base, head, paths or None)
    except subprocess.CalledProcessError as exc:
        typer.echo(f"git diff failed (check refs '{base}'/'{head}'): {exc}", err=True)
        raise typer.Exit(code=2)
    if not diff.strip():
        typer.echo("No changes to review.")
        raise typer.Exit(code=0)

    context = context_file.read_text() if context_file and context_file.is_file() else None
    client = client_factory(spec, timeout)

    tier = resolve_tier(effort.value, config=load_config())
    try:
        report = review_harness(
            client, client, tools, diff=diff, context=context,
            tier=tier, max_iters=max_iters,
        )
    except (openai.OpenAIError, httpx.HTTPError, ConnectionError, TimeoutError) as exc:
        typer.echo(
            f"Ollama request failed: {exc}\n"
            f"Is the ollama daemon running and is model '{spec.ollama_model}' pulled? "
            f"Try: kagura-code-review --doctor",
            err=True,
        )
        raise typer.Exit(code=3)

    rendered = report.to_json() if fmt is OutputFormat.json else report.to_markdown()

    if out:
        out.write_text(rendered)
    typer.echo(rendered)
    raise typer.Exit(code=report.exit_code())
