# Model + Hardware Advisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-select the best free local Ollama model for review/verify from detected hardware + installed models, making free-local the zero-config default; cloud opt-in; `--doctor` shows hardware + recommendation.

**Architecture:** New `advisor.py` with injectable hardware probes (`detect_hardware`), a curated `MODEL_CAPABILITIES` table, and a pure `recommend()` function. `cli.py` gains model-selection precedence (explicit `--model` > `--cloud` > `--local` > user config > advisor auto). `doctor.py` prints hardware + recommendation.

**Tech Stack:** Python 3.11, dataclasses, `subprocess`/`httpx` (probes), pytest + Typer CliRunner. Builds on `config.py` (`ModelSpec`, `resolve_model`), `doctor.py` (`_ollama_root`), `cli.py`, `review/harness.py`.

**Spec:** `docs/superpowers/specs/2026-06-07-model-hardware-advisor-design.md`

---

## File Structure

- **Create** `src/kagura_code_reviewer/advisor.py` — `Hardware`, `detect_hardware`, `ModelCap`, `MODEL_CAPABILITIES`, `is_cloud`, `lookup_cap`, `list_models`, `Recommendation`, `recommend`.
- **Modify** `src/kagura_code_reviewer/config.py` — `spec_from_model_name()` helper.
- **Modify** `src/kagura_code_reviewer/cli.py` — `--cloud` flag, model-selection precedence using the advisor, `None`-recommendation handling.
- **Modify** `src/kagura_code_reviewer/doctor.py` — `format_hardware_report()` used by the `--doctor` path.
- **Tests** `tests/test_advisor.py` (new); additions to `tests/test_config.py`, `tests/test_cli.py`, `tests/test_doctor.py`.

---

## Task 1: `Hardware` + injectable `detect_hardware`

**Files:**
- Create: `src/kagura_code_reviewer/advisor.py`
- Test: `tests/test_advisor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_advisor.py
from kagura_code_reviewer.advisor import Hardware, detect_hardware


def test_detect_hardware_uses_injected_probes():
    hw = detect_hardware(
        vram_reader=lambda: 24564,
        ram_reader=lambda: 96000,
        cpu_reader=lambda: 32,
    )
    assert hw == Hardware(vram_mb=24564, ram_mb=96000, cpu_threads=32, has_gpu=True)


def test_detect_hardware_no_gpu_when_vram_zero():
    hw = detect_hardware(vram_reader=lambda: 0, ram_reader=lambda: 8000, cpu_reader=lambda: 4)
    assert hw.has_gpu is False
    assert hw.vram_mb == 0


def test_detect_hardware_swallows_probe_errors():
    def boom():
        raise RuntimeError("no tool")
    hw = detect_hardware(vram_reader=boom, ram_reader=boom, cpu_reader=boom)
    assert hw == Hardware(vram_mb=0, ram_mb=0, cpu_threads=1, has_gpu=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_advisor.py -q`
Expected: FAIL (`ModuleNotFoundError: advisor`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/kagura_code_reviewer/advisor.py
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass
class Hardware:
    vram_mb: int
    ram_mb: int
    cpu_threads: int
    has_gpu: bool


def _read_vram_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(out.strip().splitlines()[0])


def _read_ram_mb() -> int:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024  # kB -> MB
    return 0


def _read_cpu_threads() -> int:
    return os.cpu_count() or 1


def _safe(reader, default):
    try:
        return reader()
    except Exception:
        return default


def detect_hardware(vram_reader=_read_vram_mb, ram_reader=_read_ram_mb,
                    cpu_reader=_read_cpu_threads) -> Hardware:
    vram = _safe(vram_reader, 0)
    ram = _safe(ram_reader, 0)
    cpu = _safe(cpu_reader, 1)
    return Hardware(vram_mb=vram, ram_mb=ram, cpu_threads=cpu, has_gpu=vram > 0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_advisor.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kagura_code_reviewer/advisor.py tests/test_advisor.py
git commit -m "feat: Hardware dataclass + injectable detect_hardware probes"
```

---

## Task 2: Capability table + `is_cloud` + `lookup_cap`

**Files:**
- Modify: `src/kagura_code_reviewer/advisor.py`
- Test: `tests/test_advisor.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_advisor.py
from kagura_code_reviewer.advisor import ModelCap, is_cloud, lookup_cap


def test_is_cloud():
    assert is_cloud("qwen3-coder:480b-cloud") is True
    assert is_cloud("kimi-k2.6:cloud") is True
    assert is_cloud("qwen2.5-coder:14b") is False


def test_lookup_cap_exact_then_family_then_default():
    exact = lookup_cap("qwen2.5-coder:14b")
    assert exact.tool_calling == "good" and exact.vram_mb == 9000
    family = lookup_cap("qwen2.5-coder:3b")          # unknown size, known family
    assert family.tool_calling in {"good", "fair"}
    default = lookup_cap("totally-unknown:1b")
    assert isinstance(default, ModelCap) and default.review <= 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_advisor.py::test_is_cloud -q`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

Add to `advisor.py`:

```python
@dataclass
class ModelCap:
    review: float
    tool_calling: str   # "good" | "fair" | "poor"
    vram_mb: int        # approx footprint (0 for cloud)
    ctx: int


MODEL_CAPABILITIES: dict[str, ModelCap] = {
    "qwen3.5:27b": ModelCap(0.85, "good", 17000, 32768),
    "qwen3:30b": ModelCap(0.82, "good", 18000, 32768),
    "qwen2.5-coder:14b": ModelCap(0.80, "good", 9000, 32768),
    "qwen3:14b": ModelCap(0.74, "good", 9000, 32768),
    "qwen2.5-coder:7b": ModelCap(0.55, "good", 5000, 16384),
    "qwen3.5:9b": ModelCap(0.55, "fair", 6600, 16384),
    "gemma4:31b": ModelCap(0.60, "fair", 19000, 8192),
    "qwen3-coder:480b-cloud": ModelCap(0.95, "good", 0, 32768),
    "qwen3.5:397b-cloud": ModelCap(0.93, "good", 0, 32768),
}

# Family-prefix fallbacks (matched by longest prefix before exact-miss default).
_FAMILY_CAPS: dict[str, ModelCap] = {
    "qwen2.5-coder": ModelCap(0.6, "good", 6000, 16384),
    "qwen3.5": ModelCap(0.6, "good", 8000, 16384),
    "qwen3-coder": ModelCap(0.9, "good", 0, 32768),
    "qwen3": ModelCap(0.6, "good", 8000, 16384),
    "gemma4": ModelCap(0.55, "fair", 12000, 8192),
    "deepseek-r1": ModelCap(0.40, "poor", 9000, 8192),
}

_DEFAULT_CAP = ModelCap(0.3, "fair", 6000, 8192)


def is_cloud(name: str) -> bool:
    return "cloud" in name


def lookup_cap(name: str) -> ModelCap:
    if name in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[name]
    family = name.split(":", 1)[0]
    for prefix in sorted(_FAMILY_CAPS, key=len, reverse=True):
        if family.startswith(prefix):
            return _FAMILY_CAPS[prefix]
    return _DEFAULT_CAP
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_advisor.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kagura_code_reviewer/advisor.py tests/test_advisor.py
git commit -m "feat: model capability table + is_cloud + lookup_cap (exact/family/default)"
```

---

## Task 3: `list_models` (installed tags via /api/tags)

**Files:**
- Modify: `src/kagura_code_reviewer/advisor.py`
- Test: `tests/test_advisor.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_advisor.py
from pytest_httpserver import HTTPServer
from kagura_code_reviewer.advisor import list_models


def test_list_models_parses_tags(httpserver: HTTPServer):
    httpserver.expect_request("/api/tags").respond_with_json(
        {"models": [{"name": "qwen2.5-coder:14b"}, {"name": "qwen3-coder:480b-cloud"}]}
    )
    names = list_models(httpserver.url_for("/v1"))
    assert names == ["qwen2.5-coder:14b", "qwen3-coder:480b-cloud"]


def test_list_models_returns_empty_on_error():
    assert list_models("http://127.0.0.1:9/v1") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_advisor.py::test_list_models_parses_tags -q`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

Add to `advisor.py`:

```python
import httpx

from .doctor import _ollama_root


def list_models(base_url: str) -> list[str]:
    try:
        resp = httpx.get(f"{_ollama_root(base_url)}/api/tags", timeout=3.0)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", []) if m.get("name")]
    except Exception:
        return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_advisor.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kagura_code_reviewer/advisor.py tests/test_advisor.py
git commit -m "feat: list_models reads installed tags via /api/tags (empty on error)"
```

---

## Task 4: `recommend()` (pure selection)

**Files:**
- Modify: `src/kagura_code_reviewer/advisor.py`
- Test: `tests/test_advisor.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_advisor.py
from kagura_code_reviewer.advisor import Recommendation, recommend

_INSTALLED = [
    "qwen3.5:27b", "qwen2.5-coder:14b", "qwen2.5-coder:7b",
    "deepseek-r1:32b", "qwen3-coder:480b-cloud",
]


def test_recommend_picks_strongest_local_that_fits_24gb():
    hw = Hardware(vram_mb=24564, ram_mb=96000, cpu_threads=32, has_gpu=True)
    rec = recommend(hw, _INSTALLED, prefer_local=True)
    assert rec.finder == "qwen3.5:27b"
    assert rec.verifier == "qwen3.5:27b"
    assert rec.fits is True


def test_recommend_excludes_models_over_vram_on_8gb():
    hw = Hardware(vram_mb=8000, ram_mb=32000, cpu_threads=8, has_gpu=True)
    rec = recommend(hw, _INSTALLED, prefer_local=True)
    assert rec.finder == "qwen2.5-coder:7b"   # 27b/14b excluded, 7b fits 7.2GB
    assert rec.fits is True


def test_recommend_ram_fallback_when_no_gpu():
    hw = Hardware(vram_mb=0, ram_mb=40000, cpu_threads=16, has_gpu=False)
    rec = recommend(hw, _INSTALLED, prefer_local=True)
    assert rec.finder == "qwen3.5:27b"   # fits RAM; nothing fits 0 VRAM
    assert rec.fits is False


def test_recommend_excludes_poor_tool_calling():
    hw = Hardware(vram_mb=24000, ram_mb=96000, cpu_threads=32, has_gpu=True)
    rec = recommend(hw, ["deepseek-r1:32b"], prefer_local=True)
    assert rec.finder is None
    assert "no suitable" in rec.reason.lower()


def test_recommend_empty_installed():
    hw = Hardware(vram_mb=24000, ram_mb=96000, cpu_threads=32, has_gpu=True)
    rec = recommend(hw, [], prefer_local=True)
    assert rec.finder is None


def test_recommend_cloud_mode_picks_best_cloud():
    hw = Hardware(vram_mb=0, ram_mb=8000, cpu_threads=8, has_gpu=False)
    rec = recommend(hw, _INSTALLED, prefer_local=False)
    assert rec.finder == "qwen3-coder:480b-cloud"
    assert rec.fits is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_advisor.py::test_recommend_picks_strongest_local_that_fits_24gb -q`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

Add to `advisor.py`:

```python
@dataclass
class Recommendation:
    finder: str | None
    verifier: str | None
    reason: str
    fits: bool


_VRAM_SAFETY = 0.9


def recommend(hardware: Hardware, installed: list[str], prefer_local: bool = True) -> Recommendation:
    candidates = [
        m for m in installed
        if (is_cloud(m) != prefer_local) and lookup_cap(m).tool_calling != "poor"
    ]
    if not candidates:
        return Recommendation(
            None, None,
            "no suitable {} model installed; pull one or use --cloud".format(
                "cloud" if not prefer_local else "local"),
            False,
        )

    if not prefer_local:
        best = max(candidates, key=lambda m: lookup_cap(m).review)
        cap = lookup_cap(best)
        return Recommendation(best, best, f"cloud model {best} (aptitude {cap.review})", True)

    usable_vram = hardware.vram_mb * _VRAM_SAFETY
    fitting = [m for m in candidates if lookup_cap(m).vram_mb <= usable_vram]
    pool, fits = (fitting, True) if fitting else (
        [m for m in candidates if lookup_cap(m).vram_mb <= hardware.ram_mb] or candidates,
        False,
    )
    best = max(pool, key=lambda m: lookup_cap(m).review)
    cap = lookup_cap(best)
    where = "GPU VRAM" if fits else "system RAM (no GPU fit; slower)"
    reason = f"local model {best} (aptitude {cap.review}, ~{cap.vram_mb} MB) fits {where}"
    return Recommendation(best, best, reason, fits)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_advisor.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kagura_code_reviewer/advisor.py tests/test_advisor.py
git commit -m "feat: recommend() pure model selection (VRAM fit, RAM fallback, cloud mode)"
```

---

## Task 5: `spec_from_model_name` in config

**Files:**
- Modify: `src/kagura_code_reviewer/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_config.py
from kagura_code_reviewer.config import ModelSpec, spec_from_model_name


def test_spec_from_model_name_builds_spec():
    spec = spec_from_model_name("qwen3.5:27b", "http://localhost:11434/v1", num_ctx=32768)
    assert isinstance(spec, ModelSpec)
    assert spec.ollama_model == "qwen3.5:27b"
    assert spec.base_url == "http://localhost:11434/v1"
    assert spec.num_ctx == 32768
    assert spec.alias == "auto"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py::test_spec_from_model_name_builds_spec -q`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Write minimal implementation**

Add to `config.py`:

```python
def spec_from_model_name(name: str, base_url: str, num_ctx: int = 8192) -> ModelSpec:
    return ModelSpec(alias="auto", ollama_model=name, base_url=base_url, num_ctx=num_ctx)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kagura_code_reviewer/config.py tests/test_config.py
git commit -m "feat: spec_from_model_name builds a ModelSpec from a bare model tag"
```

---

## Task 6: CLI precedence — `--cloud` + advisor default

**Files:**
- Modify: `src/kagura_code_reviewer/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_cli.py
from kagura_code_reviewer import advisor as advisor_mod
from kagura_code_reviewer.advisor import Recommendation


def _stub_diff(monkeypatch):
    monkeypatch.setattr(cli_mod.RepoTools, "git_diff", lambda self, b, h, p=None: "DIFF")


def test_cli_zero_config_uses_advisor(repo: Path, monkeypatch):
    from kagura_code_reviewer.report import Report
    captured = {}
    _stub_diff(monkeypatch)
    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout: object())
    monkeypatch.setattr(
        cli_mod, "recommend",
        lambda hw, installed, prefer_local=True: Recommendation("qwen3.5:27b", "qwen3.5:27b", "auto", True),
    )
    monkeypatch.setattr(cli_mod, "list_models", lambda base_url: ["qwen3.5:27b"])

    def fake_harness(fc, vc, repo_, diff, context, tier, max_iters=12, max_concurrency=1):
        captured["model"] = getattr(fc, "model", None)
        return Report(findings=[])

    # client_factory returns a plain object; assert the spec carried the advisor model instead
    def spy_factory(spec, timeout):
        captured["model"] = spec.ollama_model
        return object()
    monkeypatch.setattr(cli_mod, "client_factory", spy_factory)
    monkeypatch.setattr(cli_mod, "review_harness", fake_harness, raising=False)

    result = runner_invoke(repo)
    assert result.exit_code == 0
    assert captured["model"] == "qwen3.5:27b"
    assert "Auto-selected" in result.output


def test_cli_explicit_model_skips_advisor(repo: Path, monkeypatch):
    from kagura_code_reviewer.report import Report
    called = {"advisor": False}
    _stub_diff(monkeypatch)
    monkeypatch.setattr(cli_mod, "client_factory", lambda spec, timeout: object())
    monkeypatch.setattr(cli_mod, "review_harness",
                        lambda *a, **k: Report(findings=[]), raising=False)

    def boom_recommend(*a, **k):
        called["advisor"] = True
        raise AssertionError("advisor must not run when --model is given")
    monkeypatch.setattr(cli_mod, "recommend", boom_recommend)

    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["--base", "HEAD~1", "--repo", str(repo), "--model", "review-local"])
    assert called["advisor"] is False


def test_cli_advisor_none_exits_with_guidance(repo: Path, monkeypatch):
    _stub_diff(monkeypatch)
    monkeypatch.setattr(cli_mod, "list_models", lambda base_url: [])
    monkeypatch.setattr(cli_mod, "recommend",
                        lambda hw, installed, prefer_local=True: Recommendation(None, None, "no suitable local model", False))
    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["--base", "HEAD~1", "--repo", str(repo)])
    assert result.exit_code != 0
    assert "no suitable" in result.output
```

Add this helper near the top of `tests/test_cli.py` (after `runner` usages) to keep the zero-config test readable:

```python
def runner_invoke(repo: Path):
    return CliRunner().invoke(cli_mod.app, ["--base", "HEAD~1", "--repo", str(repo)])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py::test_cli_zero_config_uses_advisor -q`
Expected: FAIL (`--cloud`/advisor wiring absent; `cli_mod.recommend` not importable).

- [ ] **Step 3: Write minimal implementation**

In `cli.py`:

1. Imports:
```python
from .advisor import detect_hardware, list_models, recommend
from .config import load_config, resolve_model, spec_from_model_name
```

2. Add `--cloud` option to `main(...)` (after `local`):
```python
    cloud: bool = typer.Option(False, help="Use a cloud model (paid) instead of local."),
```

3. Replace `spec = resolve_model(model, local=local)` with a precedence helper call:
```python
    spec = _resolve_spec(model, local, cloud)
```

4. Add the helper above `main` (module level), using the shipped config's local
   base_url:
```python
def _local_base_url() -> str:
    models = load_config().get("models", {})
    entry = models.get("review-local") or next(iter(models.values()), {})
    return entry.get("base_url", "http://localhost:11434/v1")


def _resolve_spec(model: str | None, local: bool, cloud: bool):
    # 1. explicit alias
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
        cap_ctx = _ctx_for(rec.finder)
        return spec_from_model_name(rec.finder, base_url, num_ctx=cap_ctx)
    # 4. user config default alias
    return resolve_model(None, local=False)
```

5. Add the two small helpers `_no_user_config()` and `_ctx_for()`:
```python
from .advisor import lookup_cap
from .config import _USER  # user config path sentinel


def _no_user_config() -> bool:
    return not _USER.is_file()


def _ctx_for(model_name: str) -> int:
    return lookup_cap(model_name).ctx
```

(Note: `resolve_model`/`load_config`/`spec_from_model_name` are already imported in
Task 11 of the harness plan; ensure the final import block contains all of
`load_config, resolve_model, spec_from_model_name` and `advisor` symbols.)

Keep the `review_harness(client, client, ...)` call and the `--effort` wiring
from the harness sub-project unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cli.py -q`
Expected: PASS (new + all pre-existing CLI tests). The pre-existing tests pass
`--model`? No — they rely on the default path; they monkeypatch `client_factory`
and (for blocking) the harness. Because those tests run in a temp repo with no
user config, `_resolve_spec` now goes through the advisor. Update them in Step 5
if they break.

- [ ] **Step 5: Fix pre-existing CLI tests for the advisor default**

The pre-existing CLI tests (`test_cli_exits_nonzero_on_blocking`, etc.) now hit
the advisor path. Make them deterministic by monkeypatching the advisor in each
(or via an autouse fixture):

```python
@pytest.fixture(autouse=True)
def _stub_advisor(monkeypatch):
    monkeypatch.setattr(cli_mod, "list_models", lambda base_url: ["qwen2.5-coder:7b"], raising=False)
    monkeypatch.setattr(
        cli_mod, "recommend",
        lambda hw, installed, prefer_local=True: Recommendation("qwen2.5-coder:7b", "qwen2.5-coder:7b", "stub", True),
        raising=False,
    )
```

Place this fixture at the top of `tests/test_cli.py`. Re-run:
Run: `.venv/bin/pytest tests/test_cli.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/kagura_code_reviewer/cli.py tests/test_cli.py
git commit -m "feat: CLI model precedence with advisor default + --cloud flag"
```

---

## Task 7: `--doctor` shows hardware + recommendation

**Files:**
- Modify: `src/kagura_code_reviewer/doctor.py`
- Modify: `src/kagura_code_reviewer/cli.py` (doctor branch)
- Test: `tests/test_doctor.py`, `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_doctor.py
from kagura_code_reviewer.advisor import Hardware, Recommendation
from kagura_code_reviewer.doctor import format_hardware_report


def test_format_hardware_report_includes_hw_and_recommendation():
    hw = Hardware(vram_mb=24564, ram_mb=96000, cpu_threads=32, has_gpu=True)
    rec = Recommendation("qwen3.5:27b", "qwen3.5:27b", "fits GPU VRAM", True)
    text = format_hardware_report(hw, rec)
    assert "24564" in text and "32" in text
    assert "qwen3.5:27b" in text
    assert "fits GPU VRAM" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_doctor.py::test_format_hardware_report_includes_hw_and_recommendation -q`
Expected: FAIL (`ImportError: format_hardware_report`).

- [ ] **Step 3: Write minimal implementation**

Add to `doctor.py`:

```python
def format_hardware_report(hardware, recommendation) -> str:
    lines = [
        f"hardware: VRAM {hardware.vram_mb} MB, RAM {hardware.ram_mb} MB, "
        f"{hardware.cpu_threads} threads, GPU={'yes' if hardware.has_gpu else 'no'}",
    ]
    if recommendation.finder:
        lines.append(f"recommended local model: {recommendation.finder} — {recommendation.reason}")
    else:
        lines.append(f"recommendation: {recommendation.reason}")
    return "\n".join(lines)
```

Wire it into the `--doctor` branch of `cli.py.main` (after the two existing
checks, before `raise typer.Exit`):

```python
        from .advisor import detect_hardware, list_models, recommend
        hw = detect_hardware()
        rec = recommend(hw, list_models(spec.base_url), prefer_local=True)
        from .doctor import format_hardware_report
        typer.echo(format_hardware_report(hw, rec))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_doctor.py tests/test_cli.py::test_cli_doctor_flag -q`
Expected: PASS. (The existing doctor CLI test monkeypatches the checks; the
hardware lines are additive and detect_hardware/ list_models degrade gracefully.)

- [ ] **Step 5: Commit**

```bash
git add src/kagura_code_reviewer/doctor.py src/kagura_code_reviewer/cli.py tests/test_doctor.py
git commit -m "feat: --doctor reports hardware + model recommendation (closes 2/4 gap)"
```

---

## Task 8: Full suite + real smoke test

**Files:** none (verification)

- [ ] **Step 1: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: PASS (65 prior + new advisor/config/cli/doctor tests).

- [ ] **Step 2: Real doctor smoke**

Run: `.venv/bin/kagura-code-reviewer --doctor`
Expected: daemon OK, model line, and a `hardware:` + `recommended local model:`
line naming a real installed model (e.g. `qwen3.5:27b` on this machine).

- [ ] **Step 3: Real zero-config review smoke (free, advisor-selected model)**

Run: `timeout 600 .venv/bin/kagura-code-reviewer --base main --effort low`
Expected: stderr shows `Auto-selected <model>: ...` (a 14b/27b, not 7b); a
Markdown report prints; exit 0 if clean or 1 if a HIGH/CRITICAL finding.

- [ ] **Step 4: Commit (only if any verification fix was needed)**

```bash
git add -A && git commit -m "test: advisor end-to-end smoke verified"
```

---

## Self-Review

**Spec coverage:**
- Auto-select best free local → Task 4 `recommend` + Task 6 CLI default.
- Free-local default backend → Task 6 precedence (zero-config → advisor local).
- Respect explicit user choice → Task 6 (`--model`/user-config skip advisor) + test.
- `--doctor` hardware + recommendation → Task 7.
- Hardware detection (VRAM/RAM/CPU, graceful) → Task 1.
- Installed-model listing → Task 3.
- Capability table + cloud detection + fallback lookup → Task 2.
- Cloud opt-in (`--cloud`) → Task 6.
- Error paths (no model, probe failure, /api/tags failure) → Tasks 1/3/4/6.
- Existing 65 tests stay green → Task 6 Step 5 + Task 8.

**Placeholder scan:** No TBD/TODO. Every code step shows complete code. Task 6
Step 5 explicitly provides the autouse fixture to keep pre-existing tests
deterministic under the new advisor default.

**Type consistency:** `Hardware(vram_mb, ram_mb, cpu_threads, has_gpu)`,
`ModelCap(review, tool_calling, vram_mb, ctx)`, `Recommendation(finder, verifier,
reason, fits)`, `recommend(hardware, installed, prefer_local)`,
`lookup_cap(name)->ModelCap`, `list_models(base_url)->list[str]`,
`spec_from_model_name(name, base_url, num_ctx)->ModelSpec`, and the cli helpers
`_resolve_spec`/`_no_user_config`/`_ctx_for` are used consistently across tasks.
`config._USER` is the existing user-config path constant in `config.py`.
