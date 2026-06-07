from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

import httpx

from .doctor import _ollama_root


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

# Family-prefix fallbacks (matched by longest prefix before the default).
_FAMILY_CAPS: dict[str, ModelCap] = {
    "qwen2.5-coder": ModelCap(0.6, "good", 6000, 16384),
    "qwen3.5": ModelCap(0.6, "good", 8000, 16384),
    "qwen3-coder": ModelCap(0.9, "good", 0, 32768),
    "qwen3": ModelCap(0.6, "good", 8000, 16384),
    "gemma4": ModelCap(0.55, "fair", 12000, 8192),
    "deepseek-r1": ModelCap(0.40, "poor", 9000, 8192),
}

_DEFAULT_CAP = ModelCap(0.3, "fair", 6000, 8192)


def list_models(base_url: str) -> list[str]:
    try:
        resp = httpx.get(f"{_ollama_root(base_url)}/api/tags", timeout=3.0)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", []) if m.get("name")]
    except Exception:
        return []


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


@dataclass
class Recommendation:
    finder: str | None
    verifier: str | None
    reason: str
    fits: bool


_VRAM_SAFETY = 0.9

# KV-cache grows ~linearly with both context length and model size. This coarse
# heuristic (MB of KV per MB of weights per 1k tokens of context) is calibrated
# so a 27B model at 32k context overflows a 24GB GPU — matching the observed
# Ollama CPU-offload behaviour. Counting it stops the advisor recommending a
# model that only "fits" on paper (weights alone) and then offloads, running
# many times slower than a smaller model that fits entirely in VRAM.
_KV_MB_PER_WEIGHT_MB_PER_1K_CTX = 0.0126


def kv_cache_mb(cap: ModelCap) -> int:
    """Approximate KV-cache footprint at the model's full context length."""
    return int(cap.vram_mb * (cap.ctx / 1000.0) * _KV_MB_PER_WEIGHT_MB_PER_1K_CTX)


def effective_vram_mb(cap: ModelCap) -> int:
    """Approximate total VRAM at load: weights + KV-cache for the model's context."""
    return cap.vram_mb + kv_cache_mb(cap)


def recommend(hardware: Hardware, installed: list[str], prefer_local: bool = True) -> Recommendation:
    candidates = [
        m for m in installed
        if (is_cloud(m) != prefer_local) and lookup_cap(m).tool_calling != "poor"
    ]
    if not candidates:
        kind = "local" if prefer_local else "cloud"
        return Recommendation(
            None, None,
            f"no suitable {kind} model installed; pull one or use --cloud",
            False,
        )

    if not prefer_local:
        best = max(candidates, key=lambda m: lookup_cap(m).review)
        cap = lookup_cap(best)
        return Recommendation(best, best, f"cloud model {best} (aptitude {cap.review})", True)

    usable_vram = hardware.vram_mb * _VRAM_SAFETY
    fitting = [m for m in candidates if effective_vram_mb(lookup_cap(m)) <= usable_vram]
    if fitting:
        pool, fits = fitting, True
    else:
        pool = [m for m in candidates if effective_vram_mb(lookup_cap(m)) <= hardware.ram_mb] or candidates
        fits = False
    best = max(pool, key=lambda m: lookup_cap(m).review)
    cap = lookup_cap(best)
    where = "GPU VRAM" if fits else "system RAM (no GPU fit; slower)"
    reason = f"local model {best} (aptitude {cap.review}, ~{effective_vram_mb(cap)} MB incl. KV) fits {where}"
    return Recommendation(best, best, reason, fits)
