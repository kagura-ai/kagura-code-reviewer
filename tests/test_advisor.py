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


from kagura_code_reviewer.advisor import ModelCap, is_cloud, lookup_cap


def test_is_cloud():
    assert is_cloud("qwen3-coder:480b-cloud") is True
    assert is_cloud("kimi-k2.6:cloud") is True
    assert is_cloud("qwen2.5-coder:14b") is False


def test_lookup_cap_exact_then_family_then_default():
    exact = lookup_cap("qwen2.5-coder:14b")
    assert exact.tool_calling == "good" and exact.vram_mb == 9000
    family = lookup_cap("qwen2.5-coder:3b")
    assert family.tool_calling in {"good", "fair"}
    default = lookup_cap("totally-unknown:1b")
    assert isinstance(default, ModelCap) and default.review <= 0.4
