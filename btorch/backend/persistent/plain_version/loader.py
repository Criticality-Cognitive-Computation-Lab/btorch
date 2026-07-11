"""JIT loader for the plain persistent SNN CUDA extension."""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from shutil import which

from torch.utils.cpp_extension import load as load_extension


def _host_compiler() -> str | None:
    return which("g++-12") or which("g++")


@contextmanager
def _temporary_cxx(cxx: str | None):
    if cxx is None:
        yield
        return
    old = os.environ.get("CXX")
    os.environ["CXX"] = cxx
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("CXX", None)
        else:
            os.environ["CXX"] = old


@contextmanager
def _default_cuda_arch():
    old = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if old is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = _detect_cuda_arch_list()
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old


def _detect_cuda_arch_list() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            caps = {
                f"{major}.{minor}"
                for major, minor in (
                    torch.cuda.get_device_capability(i)
                    for i in range(torch.cuda.device_count())
                )
            }
            if caps:
                return ";".join(sorted(caps))
    except Exception:
        pass
    return "8.9"


@lru_cache(maxsize=1)
def load():
    """Build and load the plain persistent SNN CUDA extension."""

    if which("ninja") is None:
        raise RuntimeError(
            "Ninja is required to load the persistent SNN CUDA extension."
        )
    cxx = _host_compiler()
    if cxx is None:
        raise RuntimeError(
            "A C++ compiler is required to load the persistent SNN CUDA extension."
        )
    cuda_cflags = [
        "-O3",
        "--expt-relaxed-constexpr",
        "--target-directory=..",
    ]
    if cxx.endswith("g++-12"):
        cuda_cflags.append(f"-ccbin={cxx}")
    here = Path(__file__).resolve().parent
    build_directory = Path("/tmp/btorch_extensions/btorch_persistent_snn_plain")
    build_directory.mkdir(parents=True, exist_ok=True)
    with _temporary_cxx(cxx), _default_cuda_arch():
        return load_extension(
            name="btorch_persistent_snn_plain",
            sources=[
                str(here / "persistent_snn.cpp"),
                str(here / "persistent_snn_kernel.cu"),
            ],
            build_directory=str(build_directory),
            extra_cflags=["-O3"],
            extra_cuda_cflags=cuda_cflags,
            is_python_module=False,
            verbose=False,
        )
