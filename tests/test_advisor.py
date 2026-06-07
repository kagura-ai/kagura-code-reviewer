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
