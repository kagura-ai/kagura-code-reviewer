from __future__ import annotations

import subprocess
from enum import Enum
from pathlib import Path

import typer

from .config import resolve_model
from .ollama_client import OllamaClient
from .review.skill import review
from .tools import RepoTools

app = typer.Typer(add_completion=False, help="Cost-free Ollama code review.")


class OutputFormat(str, Enum):
    md = "md"
    json = "json"


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
) -> None:
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
    spec = resolve_model(model, local=local)
    client = client_factory(spec, timeout)

    report = review(client, tools, diff=diff, context=context, max_iters=max_iters)
    rendered = report.to_json() if fmt is OutputFormat.json else report.to_markdown()

    if out:
        out.write_text(rendered)
    typer.echo(rendered)
    raise typer.Exit(code=report.exit_code())
