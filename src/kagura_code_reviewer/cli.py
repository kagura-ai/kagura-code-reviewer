from __future__ import annotations

import subprocess
from enum import Enum
from pathlib import Path

import httpx
import openai
import typer

from .advisor import detect_hardware, list_models, lookup_cap, recommend
from .config import _USER, load_config, resolve_model, spec_from_model_name
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


def _local_base_url() -> str:
    models = load_config().get("models", {})
    entry = models.get("review-local") or next(iter(models.values()), {})
    return entry.get("base_url", "http://localhost:11434/v1")


def _no_user_config() -> bool:
    return not _USER.is_file()


def _resolve_spec(model: str | None, local: bool, cloud: bool):
    # 1. explicit alias wins
    if model is not None:
        return resolve_model(model, local=local)
    # 2/3. explicit backend choice OR 5. zero-config default -> advisor
    if cloud or local or _no_user_config():
        base_url = _local_base_url()
        rec = recommend(detect_hardware(), list_models(base_url), prefer_local=not cloud)
        if rec.finder is None:
            typer.echo(rec.reason, err=True)
            raise typer.Exit(code=2)
        typer.echo(f"Auto-selected {rec.finder}: {rec.reason}", err=True)
        return spec_from_model_name(rec.finder, base_url, num_ctx=lookup_cap(rec.finder).ctx)
    # 4. user config default alias
    return resolve_model(None, local=False)


@app.command()
def main(
    base: str = typer.Option("main", help="Base ref to diff against."),
    head: str = typer.Option("HEAD", help="Head ref."),
    repo: Path = typer.Option(Path("."), help="Repository root."),
    paths: list[str] = typer.Option(None, help="Limit diff to these paths."),
    context_file: Path = typer.Option(None, help="Memory context file to inject."),
    model: str = typer.Option(None, help="Model alias (see config.toml)."),
    local: bool = typer.Option(False, help="Use the local model alias."),
    cloud: bool = typer.Option(False, help="Use a cloud model (paid) instead of local."),
    fmt: OutputFormat = typer.Option(OutputFormat.md, "--format", help="Output format: md|json."),
    out: Path = typer.Option(None, help="Write the report to this file."),
    timeout: float = typer.Option(120.0, help="Per-call timeout (seconds)."),
    max_iters: int = typer.Option(12, help="Max agent iterations."),
    effort: Effort = typer.Option(Effort.med, "--effort", help="Review effort: low|med|high."),
    doctor: bool = typer.Option(False, "--doctor", help="Check ollama daemon and model availability, then exit."),
) -> None:
    if doctor:
        from . import doctor as _doctor
        base_url = _local_base_url()
        hw = detect_hardware()
        rec = recommend(hw, list_models(base_url), prefer_local=not cloud)
        results = [_doctor.check_ollama(base_url)]
        if rec.finder:
            results.append(_doctor.check_model(base_url, rec.finder))
        for r in results:
            mark = "OK " if r.ok else "FAIL"
            typer.echo(f"[{mark}] {r.name}: {r.detail}")
        typer.echo(_doctor.format_hardware_report(hw, rec))
        raise typer.Exit(code=0 if all(r.ok for r in results) else 1)

    spec = _resolve_spec(model, local, cloud)
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
            f"Try: kagura-code-reviewer --doctor",
            err=True,
        )
        raise typer.Exit(code=3)

    rendered = report.to_json() if fmt is OutputFormat.json else report.to_markdown()

    if out:
        out.write_text(rendered)
    typer.echo(rendered)
    raise typer.Exit(code=report.exit_code())
