"""JIT loader for the plain persistent SNN CUDA extension."""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from shutil import which

from torch.utils.cpp_extension import load as load_extension


_loaded_config: tuple[object, ...] | None = None


def _pipeline_enabled() -> bool:
    value = int(os.environ.get("BTORCH_PERSISTENT_PIPELINE", "0"))
    if value not in (0, 1):
        raise ValueError("BTORCH_PERSISTENT_PIPELINE must be 0 or 1.")
    return bool(value)


def _pipeline_role_ratio() -> str:
    value = os.environ.get("BTORCH_PIPELINE_ROLE_RATIO", "7:1")
    if value not in ("7:1", "3:1", "2:1", "1:1"):
        raise ValueError(
            "BTORCH_PIPELINE_ROLE_RATIO must be 7:1, 3:1, 2:1, or 1:1."
        )
    return value


def _pipeline_debug_counters() -> bool:
    value = int(os.environ.get("BTORCH_PIPELINE_DEBUG_COUNTERS", "0"))
    if value not in (0, 1):
        raise ValueError("BTORCH_PIPELINE_DEBUG_COUNTERS must be 0 or 1.")
    return bool(value)


def _pipeline_timing() -> bool:
    value = int(os.environ.get("BTORCH_PIPELINE_TIMING", "0"))
    if value not in (0, 1):
        raise ValueError("BTORCH_PIPELINE_TIMING must be 0 or 1.")
    return bool(value)


def _pipeline_component_timing() -> bool:
    value = int(os.environ.get("BTORCH_PIPELINE_COMPONENT_TIMING", "0"))
    if value not in (0, 1):
        raise ValueError("BTORCH_PIPELINE_COMPONENT_TIMING must be 0 or 1.")
    return bool(value)


def _pipeline_consumer_warps(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value not in (1, 2, 4, 8):
        raise ValueError(f"{name} must be 1, 2, 4, or 8.")
    return value


def _pipeline_ticket_chunk() -> int:
    value = int(os.environ.get("BTORCH_PIPELINE_TICKET_CHUNK", "1"))
    if value not in (1, 2, 4):
        raise ValueError("BTORCH_PIPELINE_TICKET_CHUNK must be 1, 2, or 4.")
    return value


def _pipeline_static_waves() -> int:
    value = int(os.environ.get("BTORCH_PIPELINE_STATIC_WAVES", "0"))
    if value not in (0, 1, 2, 4):
        raise ValueError(
            "BTORCH_PIPELINE_STATIC_WAVES must be 0, 1, 2, or 4."
        )
    return value


def _pipeline_binned_threshold() -> int:
    value = int(os.environ.get("BTORCH_PIPELINE_BINNED_THRESHOLD", "512"))
    if value not in (0, 128, 256, 512):
        raise ValueError(
            "BTORCH_PIPELINE_BINNED_THRESHOLD must be 0, 128, 256, or 512."
        )
    return value


def _pipeline_low_subwarp_size() -> int:
    value = int(os.environ.get("BTORCH_PIPELINE_LOW_SUBWARP_SIZE", "8"))
    if value not in (4, 8, 16):
        raise ValueError(
            "BTORCH_PIPELINE_LOW_SUBWARP_SIZE must be 4, 8, or 16."
        )
    return value


def _pipeline_high_low_ratio() -> int:
    value = int(os.environ.get("BTORCH_PIPELINE_HIGH_LOW_RATIO", "2"))
    if value not in (1, 2, 4):
        raise ValueError(
            "BTORCH_PIPELINE_HIGH_LOW_RATIO must be 1, 2, or 4."
        )
    return value


def _block_v4_config() -> tuple[int, int, int]:
    block_budget = int(os.environ.get("BTORCH_BLOCK_EDGE_BUDGET", "0"))
    long_segment = int(os.environ.get("BTORCH_LONG_SEGMENT_SIZE", "1024"))
    tile_reduce = int(os.environ.get("BTORCH_TILE_REDUCE_MODE", "0"))
    if block_budget not in (0, 128, 256, 512, 1024):
        raise ValueError("BTORCH_BLOCK_EDGE_BUDGET has an unsupported value.")
    if long_segment not in (128, 256, 512, 1024, 2048):
        raise ValueError("BTORCH_LONG_SEGMENT_SIZE has an unsupported value.")
    if tile_reduce not in (0, 1, 2):
        raise ValueError("BTORCH_TILE_REDUCE_MODE has an unsupported value.")
    return block_budget, long_segment, tile_reduce


def _block_hash_config() -> tuple[int, int, int, int, int]:
    aggregation = int(os.environ.get("BTORCH_BLOCK_HASH_AGGREGATION", "512"))
    capacity = int(os.environ.get("BTORCH_BLOCK_HASH_CAPACITY", "512"))
    max_probe = int(os.environ.get("BTORCH_BLOCK_HASH_MAX_PROBE", "4"))
    min_edges = int(os.environ.get("BTORCH_BLOCK_HASH_MIN_EDGES", "256"))
    used_slots = int(os.environ.get("BTORCH_BLOCK_HASH_USED_SLOTS", "1"))
    if aggregation not in (128, 256, 512):
        raise ValueError("BTORCH_BLOCK_HASH_AGGREGATION is unsupported.")
    if capacity not in (128, 256, 512):
        raise ValueError("BTORCH_BLOCK_HASH_CAPACITY is unsupported.")
    if max_probe not in (4, 8, 16):
        raise ValueError("BTORCH_BLOCK_HASH_MAX_PROBE is unsupported.")
    if min_edges not in (0, 64, 128, 192, 256):
        raise ValueError("BTORCH_BLOCK_HASH_MIN_EDGES is unsupported.")
    if used_slots not in (0, 1):
        raise ValueError("BTORCH_BLOCK_HASH_USED_SLOTS is unsupported.")
    return aggregation, capacity, max_probe, min_edges, used_slots


def _warp_spec_mode() -> int:
    mode = int(os.environ.get("BTORCH_WARP_SPEC_MODE", "0"))
    if mode not in (0, 1, 2, 3, 4):
        raise ValueError(
            "BTORCH_WARP_SPEC_MODE must be 0, 1, 2, 3, or 4."
        )
    return mode


def _long_warp_spec_config() -> tuple[bool, int, int, int, int]:
    enabled = "BTORCH_LONG_WARP_SPEC_MODE" in os.environ
    mode = int(os.environ.get("BTORCH_LONG_WARP_SPEC_MODE", "0"))
    chunk = int(os.environ.get("BTORCH_LONG_WARP_SPEC_CHUNK", "128"))
    stages = int(os.environ.get("BTORCH_LONG_WARP_SPEC_STAGES", "3"))
    threshold = int(
        os.environ.get("BTORCH_LONG_WARP_SPEC_THRESHOLD", "256")
    )
    if mode not in (0, 1, 2, 3):
        raise ValueError(
            "BTORCH_LONG_WARP_SPEC_MODE must be 0, 1, 2, or 3; "
            "S4/S5 remain gated on the S3 experiment."
        )
    if chunk not in (64, 128, 256):
        raise ValueError("BTORCH_LONG_WARP_SPEC_CHUNK is unsupported.")
    if stages not in (2, 3, 4):
        raise ValueError("BTORCH_LONG_WARP_SPEC_STAGES is unsupported.")
    if threshold not in (128, 256, 384, 512):
        raise ValueError("BTORCH_LONG_WARP_SPEC_THRESHOLD is unsupported.")
    return enabled, mode, chunk, stages, threshold


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


@lru_cache(maxsize=4)
def load(
    *, enable_block_stats: bool = False, enable_block_hash: bool = False
):
    """Build and load the plain persistent SNN CUDA extension.

    Set ``BTORCH_PERSISTENT_PIPELINE=1`` before the first load in a process
    to use the batch-one UPDATE--propagation pipeline kernel. Set
    ``BTORCH_PIPELINE_ROLE_RATIO`` to ``7:1``, ``3:1``, ``2:1``, or ``1:1``
    to select the initial UPDATE:PROPAGATION block ratio; UPDATE blocks become
    propagation helpers after finishing their neurons. Set
    ``BTORCH_PIPELINE_DEBUG_COUNTERS=1`` to return queue polling diagnostics in
    the sixth output tensor. ``BTORCH_PIPELINE_DEDICATED_WARPS`` and
    ``BTORCH_PIPELINE_HELPER_WARPS`` select 1, 2, 4, or 8 ticket consumers for
    dedicated propagation blocks and completed UPDATE blocks, respectively.
    ``BTORCH_PIPELINE_TICKET_CHUNK`` selects 1, 2, or 4 tasks per ticket
    allocation. ``BTORCH_PIPELINE_STATIC_WAVES`` selects 0, 1, 2, or 4
    deterministic startup claims per dedicated warp. The RTX 5090 FlyBrain
    scan defaults to a 7:1 role ratio, one dedicated warp, one helper warp,
    one task per allocation, zero static waves, a 512-edge HIGH threshold,
    eight LOW lanes, and a 2:1 HIGH/LOW schedule.
    ``BTORCH_PIPELINE_BINNED_THRESHOLD`` selects 0 (disabled), 128, 256, or
    512 edges for the two-ended HIGH/LOW pipeline queue, and
    ``BTORCH_PIPELINE_LOW_SUBWARP_SIZE`` selects 4, 8, or 16 lanes per LOW
    neuron. ``BTORCH_PIPELINE_HIGH_LOW_RATIO`` selects 1, 2, or 4 HIGH tasks
    per LOW group. ``BTORCH_PIPELINE_UPDATE_SMEM_KB`` caps the dynamic shared
    memory available to each UPDATE block (32 KiB by default); the launcher
    chooses a cooperative grid that can stage all neuron state within it.
    ``BTORCH_PIPELINE_TIMING=1`` builds opt-in overlap
    timestamps, while ``BTORCH_PIPELINE_COMPONENT_TIMING=1`` enables host-side
    CUDA Event component timings. Instrumentation modes are mutually exclusive.
    """

    global _loaded_config
    block_budget, long_segment, tile_reduce = _block_v4_config()
    (
        hash_aggregation,
        hash_capacity,
        hash_max_probe,
        hash_min_edges,
        hash_used_slots,
    ) = _block_hash_config()
    warp_spec_mode = _warp_spec_mode()
    (
        long_warp_spec_enabled,
        long_warp_spec_mode,
        long_warp_spec_chunk,
        long_warp_spec_stages,
        long_warp_spec_threshold,
    ) = _long_warp_spec_config()
    if long_warp_spec_enabled and warp_spec_mode:
        raise ValueError(
            "Block and long-segment warp specialization modes cannot "
            "be enabled together."
        )
    pipeline_enabled = _pipeline_enabled()
    pipeline_role_ratio = _pipeline_role_ratio() if pipeline_enabled else "7:1"
    pipeline_debug_counters = (
        _pipeline_debug_counters() if pipeline_enabled else False
    )
    pipeline_timing = _pipeline_timing()
    pipeline_component_timing = _pipeline_component_timing()
    if sum(
        (
            pipeline_debug_counters,
            pipeline_timing,
            pipeline_component_timing,
        )
    ) > 1:
        raise ValueError(
            "Pipeline debug counters, overlap timing, and component timing "
            "are mutually exclusive."
        )
    pipeline_dedicated_warps = (
        _pipeline_consumer_warps("BTORCH_PIPELINE_DEDICATED_WARPS", 1)
        if pipeline_enabled
        else 1
    )
    pipeline_helper_warps = (
        _pipeline_consumer_warps("BTORCH_PIPELINE_HELPER_WARPS", 1)
        if pipeline_enabled
        else 1
    )
    pipeline_ticket_chunk = (
        _pipeline_ticket_chunk() if pipeline_enabled else 1
    )
    pipeline_static_waves = (
        _pipeline_static_waves() if pipeline_enabled else 0
    )
    pipeline_binned_threshold = (
        _pipeline_binned_threshold() if pipeline_enabled else 512
    )
    pipeline_low_subwarp_size = (
        _pipeline_low_subwarp_size() if pipeline_enabled else 8
    )
    pipeline_high_low_ratio = (
        _pipeline_high_low_ratio() if pipeline_enabled else 2
    )
    if pipeline_binned_threshold and (
        pipeline_ticket_chunk != 1 or pipeline_static_waves != 0
    ):
        raise ValueError(
            "Pipeline binning requires ticket chunk 1 and static waves 0."
        )
    requested_config = (
        pipeline_enabled,
        pipeline_role_ratio,
        pipeline_debug_counters,
        pipeline_timing,
        pipeline_component_timing,
        pipeline_dedicated_warps,
        pipeline_helper_warps,
        pipeline_ticket_chunk,
        pipeline_static_waves,
        pipeline_binned_threshold,
        pipeline_low_subwarp_size,
        pipeline_high_low_ratio,
        enable_block_stats,
        enable_block_hash,
        block_budget,
        long_segment,
        tile_reduce,
        hash_aggregation,
        hash_capacity,
        hash_max_probe,
        hash_min_edges,
        hash_used_slots,
        warp_spec_mode,
        long_warp_spec_enabled,
        long_warp_spec_mode,
        long_warp_spec_chunk,
        long_warp_spec_stages,
        long_warp_spec_threshold,
    )
    if (
        _loaded_config is not None
        and _loaded_config != requested_config
    ):
        raise RuntimeError(
            "The persistent SNN extension cannot switch debug/experimental "
            "compile flags "
            "inside one process; start a fresh process for the other mode."
        )

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
    cflags = ["-O3"]
    suffixes = []
    if enable_block_stats:
        suffixes.append("stats")
    if enable_block_hash:
        suffixes.append("hash")
    suffixes.extend(
        [
            f"b{block_budget}",
            f"s{long_segment}",
            f"r{tile_reduce}",
        ]
    )
    if enable_block_hash:
        suffixes.extend(
            [
                f"ha{hash_aggregation}",
                f"hc{hash_capacity}",
                f"hp{hash_max_probe}",
                f"hm{hash_min_edges}",
                f"hu{hash_used_slots}",
            ]
        )
    if warp_spec_mode:
        suffixes.append(f"ws{warp_spec_mode}")
    if long_warp_spec_enabled:
        suffixes.extend(
            [
                f"lws{long_warp_spec_mode}",
                f"lc{long_warp_spec_chunk}",
                f"ls{long_warp_spec_stages}",
                f"lt{long_warp_spec_threshold}",
            ]
        )
    if pipeline_enabled:
        suffixes.append("pipeline")
        suffixes.append(f"lsw{pipeline_low_subwarp_size}")
        suffixes.append(f"hl{pipeline_high_low_ratio}")
    if pipeline_timing:
        suffixes.append("pipeline_timing")
    extension_suffix = f"_{'_'.join(suffixes)}"
    cuda_cflags.extend(
        [
            f"-DBTORCH_BLOCK_EDGE_BUDGET={block_budget}",
            f"-DBTORCH_LONG_SEGMENT_SIZE={long_segment}",
            f"-DBTORCH_TILE_REDUCE_MODE={tile_reduce}",
            f"-DBTORCH_BLOCK_HASH_AGGREGATION={hash_aggregation}",
            f"-DBTORCH_BLOCK_HASH_CAPACITY={hash_capacity}",
            f"-DBTORCH_BLOCK_HASH_MAX_PROBE={hash_max_probe}",
            f"-DBTORCH_BLOCK_HASH_MIN_EDGES={hash_min_edges}",
            f"-DBTORCH_BLOCK_HASH_USED_SLOTS={hash_used_slots}",
            f"-DBTORCH_WARP_SPEC_MODE={warp_spec_mode}",
            f"-DBTORCH_LONG_WARP_SPEC_MODE={long_warp_spec_mode}",
            f"-DBTORCH_LONG_WARP_SPEC_CHUNK={long_warp_spec_chunk}",
            f"-DBTORCH_LONG_WARP_SPEC_STAGES={long_warp_spec_stages}",
            f"-DBTORCH_LONG_WARP_SPEC_THRESHOLD={long_warp_spec_threshold}",
        ]
    )
    cflags.append(f"-DBTORCH_WARP_SPEC_MODE={warp_spec_mode}")
    cflags.append(
        f"-DBTORCH_LONG_WARP_SPEC_ENABLED="
        f"{int(long_warp_spec_enabled)}"
    )
    if pipeline_enabled:
        cflags.append("-DBTORCH_PERSISTENT_PIPELINE")
        cuda_cflags.append("-DBTORCH_PERSISTENT_PIPELINE")
        cuda_cflags.append(
            f"-DBTORCH_PIPELINE_LOW_SUBWARP_SIZE="
            f"{pipeline_low_subwarp_size}"
        )
        cuda_cflags.append(
            f"-DBTORCH_PIPELINE_HIGH_LOW_RATIO="
            f"{pipeline_high_low_ratio}"
        )
    if pipeline_timing:
        cflags.append("-DENABLE_PIPELINE_TIMING")
        cuda_cflags.append("-DENABLE_PIPELINE_TIMING")
    if enable_block_stats:
        cflags.append("-DENABLE_BLOCK_STATS")
        cuda_cflags.append("-DENABLE_BLOCK_STATS")
    if enable_block_hash:
        cflags.append("-DENABLE_BLOCK_HASH")
        cuda_cflags.append("-DENABLE_BLOCK_HASH")
    if cxx.endswith("g++-12"):
        cuda_cflags.append(f"-ccbin={cxx}")
    here = Path(__file__).resolve().parent
    if long_warp_spec_enabled:
        spike_block_source = (
            "persistent_snn_spike_block_segment_warp_spec_kernel.cu"
        )
    elif warp_spec_mode == 0:
        spike_block_source = "persistent_snn_spike_block_kernel.cu"
    else:
        spike_block_source = (
            "persistent_snn_spike_block_warp_spec_kernel.cu"
        )
    build_directory = Path(
        f"/tmp/btorch_extensions/btorch_persistent_snn_plain{extension_suffix}"
    )
    build_directory.mkdir(parents=True, exist_ok=True)
    with _temporary_cxx(cxx), _default_cuda_arch():
        extension = load_extension(
            name=f"btorch_persistent_snn_plain{extension_suffix}",
            sources=[
                str(here / "persistent_snn.cpp"),
                str(
                    here
                    / (
                        "persistent_snn_pipeline_kernel.cu"
                        if pipeline_enabled
                        else "persistent_snn_plain_kernel.cu"
                    )
                ),
                str(here / "persistent_snn_binned_kernel.cu"),
                str(here / spike_block_source),
            ],
            build_directory=str(build_directory),
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_cflags,
            is_python_module=False,
            verbose=False,
        )
    _loaded_config = requested_config
    return extension
