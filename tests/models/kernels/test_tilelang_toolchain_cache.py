"""Verify TileLang kernels cannot reuse binaries from another CUDA toolkit."""

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from btorch.models.kernels.backends import engine


@pytest.fixture
def mock_tilelang_toolchain(tmp_path, monkeypatch):
    tilelang_nvcc = pytest.importorskip("tilelang.contrib.nvcc")
    cuda_root = tmp_path / "cuda"
    bin_dir = cuda_root / "bin"
    libdevice = cuda_root / "nvvm" / "libdevice" / "libdevice.10.bc"
    bin_dir.mkdir(parents=True)
    libdevice.parent.mkdir(parents=True)
    nvcc = bin_dir / "nvcc"
    ptxas = bin_dir / "ptxas"
    nvcc.write_text("mock nvcc")
    ptxas.write_text("mock ptxas")
    libdevice.write_bytes(b"libdevice-a")
    versions = {nvcc: "nvcc 13.2", ptxas: "ptxas 13.2"}

    monkeypatch.setattr(tilelang_nvcc, "get_nvcc_compiler", lambda: str(nvcc))
    monkeypatch.setattr(tilelang_nvcc, "find_cuda_path", lambda: str(cuda_root))

    def run(command, **_kwargs):
        executable = Path(command[0]).resolve()
        return subprocess.CompletedProcess(command, 0, versions[executable], "")

    monkeypatch.setattr(engine.subprocess, "run", run)
    engine._tilelang_toolchain_stamp.cache_clear()
    yield SimpleNamespace(
        cuda_root=cuda_root,
        nvcc=nvcc,
        ptxas=ptxas,
        libdevice=libdevice,
        versions=versions,
    )
    engine._tilelang_toolchain_stamp.cache_clear()


def test_toolchain_stamp_tracks_compilers_and_libdevice(mock_tilelang_toolchain):
    toolchain = mock_tilelang_toolchain
    initial = engine._tilelang_toolchain_stamp()

    toolchain.versions[toolchain.nvcc.resolve()] = "nvcc 13.3"
    engine._tilelang_toolchain_stamp.cache_clear()
    changed_nvcc = engine._tilelang_toolchain_stamp()
    assert changed_nvcc != initial

    # A changed device library can alter transcendental coefficients even when
    # the CUDA source and compiler flags are identical. It must produce a new
    # cache identity rather than retrieving the previous cubin.
    toolchain.libdevice.write_bytes(b"libdevice-b")
    engine._tilelang_toolchain_stamp.cache_clear()
    changed_libdevice = engine._tilelang_toolchain_stamp()
    assert changed_libdevice not in (initial, changed_nvcc)

    # ptxas is a separate executable selected by NVCC. Mixed toolkit installs
    # are uncommon but valid, so its reported build also belongs in the stamp.
    toolchain.versions[toolchain.ptxas.resolve()] = "ptxas 13.3"
    engine._tilelang_toolchain_stamp.cache_clear()
    changed_ptxas = engine._tilelang_toolchain_stamp()
    assert changed_ptxas not in (initial, changed_nvcc, changed_libdevice)


def test_missing_toolchain_component_fails_with_path(mock_tilelang_toolchain):
    toolchain = mock_tilelang_toolchain
    toolchain.libdevice.unlink()
    engine._tilelang_toolchain_stamp.cache_clear()

    # Falling back to an "unknown" stamp would recreate the cross-toolchain
    # collision. Fail before compilation and identify the incomplete toolkit.
    with pytest.raises(RuntimeError, match=str(toolchain.libdevice)):
        engine._tilelang_toolchain_stamp()


def test_target_and_toolchain_separate_compiled_tilelang_kernels(
    monkeypatch,
):
    kernel_cache_module = pytest.importorskip("tilelang.cache.kernel_cache")
    binary_cache_module = pytest.importorskip("tilelang.cache.cuda_binary_cache")
    tilelang = ModuleType("tilelang")
    seen_compile_flags = []
    seen_targets = []

    def jit(*, compile_flags, target=None):
        def decorate(function):
            seen_compile_flags.append(tuple(compile_flags))
            seen_targets.append(target)
            return function

        return decorate

    tilelang.jit = jit
    monkeypatch.setitem(sys.modules, "tilelang", tilelang)
    stamps = iter(("a" * 64, "a" * 64, "a" * 64, "a" * 64, "b" * 64))
    targets = iter(
        (
            "cuda -arch=sm_80",
            "cuda -arch=sm_80",
            "cuda -arch=sm_80",
            "cuda -arch=sm_90",
            "cuda -arch=sm_90",
        )
    )
    monkeypatch.setattr(engine, "_current_tilelang_target", lambda: next(targets))
    monkeypatch.setattr(engine, "_tilelang_toolchain_stamp", lambda: next(stamps))
    engine._compile_cached.cache_clear()
    source = (
        "import tilelang\nclass Kernel: pass\n@tilelang.jit\n"
        "def build_kernel():\n    return Kernel()\n"
    )

    first = engine._compile("tilelang", source, False, True)
    repeated = engine._compile("tilelang", source, False, True)
    second_device = engine._compile("tilelang", source, False, True)
    second_target = engine._compile("tilelang", source, False, True)
    second_toolchain = engine._compile("tilelang", source, False, True)
    assert repeated is first
    # Equal targets reuse the backend's callable even on another device; the
    # launch guard selects its CUDA context. Architectures cannot share code.
    assert second_device is first
    assert second_target is not second_device
    assert second_toolchain is not first
    assert len(seen_compile_flags) == 3
    assert "--fmad=false" in seen_compile_flags[0]
    assert "--prec-div=true" in seen_compile_flags[0]
    assert seen_compile_flags[0] == seen_compile_flags[1]
    assert seen_compile_flags[1] != seen_compile_flags[2]

    # TileLang keys the serialized TIR cache on compile_flags and the lower
    # CUDA-binary cache on the resulting compiler options. Confirm the stamp
    # separates both layers that shared-cache qualification exercises.
    class StableFunction:
        def script(self, *, show_meta):
            assert show_meta
            return "stable TIR"

    kernel_cache = object.__new__(kernel_cache_module.KernelCache)
    outer_keys = [
        kernel_cache._generate_key(
            StableFunction(),
            [],
            args=(),
            target=target,
            pass_configs=None,
            compile_flags=list(flags),
        )
        for flags, target in zip(seen_compile_flags, seen_targets)
    ]
    binary_keys = [
        binary_cache_module.CUDABinaryCache.make_key(
            code="stable CUDA",
            target_kind="cuda",
            target_arch=target.rsplit("sm_", 1)[1],
            target_code=["sm_" + target.rsplit("sm_", 1)[1]],
            compile_format="cubin",
            options=list(flags),
        )
        for flags, target in zip(seen_compile_flags, seen_targets)
    ]
    assert outer_keys[0] != outer_keys[1] != outer_keys[2]
    assert binary_keys[0] != binary_keys[1] != binary_keys[2]


@pytest.mark.parametrize("backend", ("triton", "cupy"))
def test_non_tilelang_compilation_does_not_probe_nvcc(backend, monkeypatch):
    monkeypatch.setattr(
        engine,
        "_tilelang_toolchain_stamp",
        lambda: pytest.fail("non-TileLang compilation probed NVCC"),
    )
    engine._compile_cached.cache_clear()
    if backend == "triton":
        kernel = engine._compile("triton", "def cell_kernel():\n    return None\n")
        assert callable(kernel)
        return

    cupy = ModuleType("cupy")

    class RawKernel:
        def __init__(self, *_args, **_kwargs):
            self.compiled = False

        def compile(self):
            self.compiled = True

    cupy.RawKernel = RawKernel
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    kernel = engine._compile("cupy", "unused")
    assert kernel.compiled
