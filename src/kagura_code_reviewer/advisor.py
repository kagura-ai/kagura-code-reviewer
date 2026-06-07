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
