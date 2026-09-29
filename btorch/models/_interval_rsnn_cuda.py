"""Private CUDA implementation for cyclic-interval RSNN inference."""

from __future__ import annotations

import ctypes
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.sparse
import torch
from torch import nn


SparseRSNNStrategy = Literal[
    "cyclic_interval",
    "hybrid_rangegate",
    "ed_int32_natural",
]

_STRATEGY_FUNCTIONS: dict[SparseRSNNStrategy, bytes] = {
    "cyclic_interval": b"cyclic_interval_rsnn",
    "hybrid_rangegate": b"hybrid_rangegate_rsnn",
    "ed_int32_natural": b"ed_int32_natural_rsnn",
}


_CUDA_SOURCE = r"""
#include <cuda_runtime.h>

__device__ __forceinline__ int lower_bound_int32(
    const int* __restrict__ col, int lo, int hi, int target)
{
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (col[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

template <int strategy>
__device__ __forceinline__ void run_cyclic_sparse_rsnn(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const int* __restrict__ span_lo,
    const int* __restrict__ span_hi,
    float* __restrict__ spikes_out,
    float* __restrict__ voltage_out,
    float* __restrict__ current_out,
    int n, int active, int base_start, int stride, int steps,
    float current_decay, float voltage_decay,
    int last_lo, int last_hi, int last_wrap)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    const int row_start = row_ptr[row];
    const int row_end = row_ptr[row + 1];
    const bool empty_row = row_start == row_end;
    float current = 0.0f;
    float voltage = 0.0f;
    int interval_start = base_start;

    for (int step = 0; step < steps; ++step) {
        const int interval_end = interval_start + active;
        float recurrent = 0.0f;

        if (!empty_row) {
            if constexpr (strategy == 0) {
                const int begin = lower_bound_int32(
                    col_idx, row_start, row_end, interval_start);
                if (interval_end <= n) {
                    if (begin < row_end && col_idx[begin] < interval_end) {
                        const int end = lower_bound_int32(
                            col_idx, begin, row_end, interval_end);
                        for (int edge = begin; edge < end; ++edge) {
                            recurrent = __fadd_rn(recurrent, values[edge]);
                        }
                    }
                } else {
                    const int wrap = interval_end - n;
                    const int low_end = lower_bound_int32(
                        col_idx, row_start, row_end, wrap);
                    for (int edge = row_start; edge < low_end; ++edge) {
                        recurrent = __fadd_rn(recurrent, values[edge]);
                    }
                    for (int edge = begin; edge < row_end; ++edge) {
                        recurrent = __fadd_rn(recurrent, values[edge]);
                    }
                }
            } else {
                const int first_col = strategy == 1
                    ? span_lo[row]
                    : col_idx[row_start];
                const int last_col = strategy == 1
                    ? span_hi[row]
                    : col_idx[row_end - 1];
                if (interval_end <= n) {
                    if (first_col >= interval_start && last_col < interval_end) {
                        for (int edge = row_start; edge < row_end; ++edge) {
                            recurrent = __fadd_rn(recurrent, values[edge]);
                        }
                    } else if (last_col < interval_start || first_col >= interval_end) {
                        // The active interval does not intersect this row.
                    } else {
                        const int begin = lower_bound_int32(
                            col_idx, row_start, row_end, interval_start);
                        if (begin < row_end && col_idx[begin] < interval_end) {
                            const int end = lower_bound_int32(
                                col_idx, begin, row_end, interval_end);
                            for (int edge = begin; edge < end; ++edge) {
                                recurrent = __fadd_rn(recurrent, values[edge]);
                            }
                        }
                    }
                } else {
                    const int wrap = interval_end - n;
                    if (last_col < wrap || first_col >= interval_start) {
                        for (int edge = row_start; edge < row_end; ++edge) {
                            recurrent = __fadd_rn(recurrent, values[edge]);
                        }
                    } else if (first_col >= wrap && last_col < interval_start) {
                        // All columns lie in the inactive gap [wrap, interval_start).
                    } else {
                        const int low_end = lower_bound_int32(
                            col_idx, row_start, row_end, wrap);
                        for (int edge = row_start; edge < low_end; ++edge) {
                            recurrent = __fadd_rn(recurrent, values[edge]);
                        }
                        const int high_begin = lower_bound_int32(
                            col_idx, low_end, row_end, interval_start);
                        for (int edge = high_begin; edge < row_end; ++edge) {
                            recurrent = __fadd_rn(recurrent, values[edge]);
                        }
                    }
                }
            }
        }

        current = __fadd_rn(__fmul_rn(current_decay, current), recurrent);
        voltage = __fadd_rn(__fmul_rn(voltage_decay, voltage), current);
        const bool active_now = interval_end <= n
            ? row >= interval_start && row < interval_end
            : row >= interval_start || row < interval_end - n;
        if (active_now) voltage = 0.0f;

        interval_start += stride;
        if (interval_start >= n) interval_start -= n;
    }

    const bool active_last = last_wrap > 0
        ? row >= last_lo || row < last_wrap
        : row >= last_lo && row < last_hi;
    spikes_out[row] = active_last ? 1.0f : 0.0f;
    voltage_out[row] = voltage;
    current_out[row] = current;
}

#define DEFINE_CYCLIC_RNN_KERNEL(name, strategy) \
extern "C" __global__ void name( \
    const int* row_ptr, const int* col_idx, const float* values, \
    const int* span_lo, const int* span_hi, float* spikes_out, \
    float* voltage_out, float* current_out, int n, int active, \
    int base_start, int stride, int steps, float current_decay, \
    float voltage_decay, int last_lo, int last_hi, int last_wrap) { \
    run_cyclic_sparse_rsnn<strategy>( \
        row_ptr, col_idx, values, span_lo, span_hi, spikes_out, voltage_out, \
        current_out, n, active, base_start, stride, steps, current_decay, \
        voltage_decay, last_lo, last_hi, last_wrap); \
}

DEFINE_CYCLIC_RNN_KERNEL(cyclic_interval_rsnn, 0)
DEFINE_CYCLIC_RNN_KERNEL(hybrid_rangegate_rsnn, 1)
DEFINE_CYCLIC_RNN_KERNEL(ed_int32_natural_rsnn, 2)

#undef DEFINE_CYCLIC_RNN_KERNEL
"""


def _cuda_include_path() -> str:
    """Return the CUDA runtime include directory for the active environment."""

    env_root = Path(sys.executable).resolve().parent.parent
    candidates = (
        env_root / "targets" / "x86_64-linux" / "include",
        env_root / "include",
        Path("/usr/local/cuda/include"),
    )
    for candidate in candidates:
        if (candidate / "cuda_runtime.h").is_file():
            return str(candidate)
    raise RuntimeError("cuda_runtime.h was not found in the active environment")


def _check_cuda(result: int, operation: str) -> None:
    if result != 0:
        raise RuntimeError(f"CUDA driver error {result} during {operation}")


def _compile_cubin(source: str, architecture: str) -> bytes:
    """Compile CUDA source to a cubin with NVRTC."""

    nvrtc = ctypes.CDLL("libnvrtc.so")
    nvrtc.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    nvrtc.nvrtcCreateProgram.restype = ctypes.c_int
    nvrtc.nvrtcCompileProgram.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    nvrtc.nvrtcCompileProgram.restype = ctypes.c_int
    nvrtc.nvrtcGetProgramLogSize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    nvrtc.nvrtcGetCUBINSize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    nvrtc.nvrtcGetCUBIN.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    program = ctypes.c_void_p()
    result = nvrtc.nvrtcCreateProgram(
        ctypes.byref(program),
        source.encode(),
        b"cyclic_sparse_rsnn.cu",
        0,
        None,
        None,
    )
    if result != 0:
        raise RuntimeError(f"nvrtcCreateProgram failed with code {result}")

    options = (
        f"--gpu-architecture={architecture}".encode(),
        b"--std=c++17",
        f"--include-path={_cuda_include_path()}".encode(),
    )
    option_array = (ctypes.c_char_p * len(options))(*options)
    result = nvrtc.nvrtcCompileProgram(program, len(options), option_array)
    if result != 0:
        log_size = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(log_size))
        log = ctypes.create_string_buffer(log_size.value + 1)
        nvrtc.nvrtcGetProgramLog(program, log)
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
        raise RuntimeError(f"NVRTC compilation failed:\n{log.value.decode()}")

    cubin_size = ctypes.c_size_t()
    nvrtc.nvrtcGetCUBINSize(program, ctypes.byref(cubin_size))
    cubin = ctypes.create_string_buffer(cubin_size.value)
    nvrtc.nvrtcGetCUBIN(program, cubin)
    nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
    return cubin.raw


class _MarshalledArguments:
    """Keep CUDA kernel argument storage alive across cached launches."""

    __slots__ = ("params", "values")

    def __init__(self, params: ctypes.Array, values: list[Any]) -> None:
        self.params = params
        self.values = values


class _KernelLauncher:
    """Launch one NVRTC-compiled kernel through the CUDA driver API."""

    def __init__(self, cubin: bytes, function_name: bytes) -> None:
        self._cuda = ctypes.CDLL("libcuda.so.1")
        self._module = ctypes.c_void_p()
        self._function = ctypes.c_void_p()
        self._cached_key: tuple[Any, ...] | None = None
        self._cached_arguments: _MarshalledArguments | None = None
        self._configure_driver_api()
        cubin_buffer = ctypes.create_string_buffer(cubin)
        _check_cuda(
            self._cuda.cuModuleLoadData(
                ctypes.byref(self._module), ctypes.cast(cubin_buffer, ctypes.c_void_p)
            ),
            "cuModuleLoadData",
        )
        _check_cuda(
            self._cuda.cuModuleGetFunction(
                ctypes.byref(self._function),
                self._module,
                function_name,
            ),
            "cuModuleGetFunction",
        )

    def _configure_driver_api(self) -> None:
        cuda = self._cuda
        cuda.cuModuleLoadData.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        cuda.cuModuleLoadData.restype = ctypes.c_int
        cuda.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        cuda.cuModuleGetFunction.restype = ctypes.c_int
        cuda.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        cuda.cuLaunchKernel.restype = ctypes.c_int
        cuda.cuFuncGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        cuda.cuFuncGetAttribute.restype = ctypes.c_int

    @staticmethod
    def _marshal(arguments: list[torch.Tensor | int | float]) -> _MarshalledArguments:
        values: list[Any] = []
        pointers: list[ctypes.c_void_p] = []
        for argument in arguments:
            if isinstance(argument, torch.Tensor):
                value = ctypes.c_void_p(argument.data_ptr())
            elif isinstance(argument, int):
                value = ctypes.c_int(argument)
            else:
                value = ctypes.c_float(argument)
            values.append(value)
            pointers.append(ctypes.cast(ctypes.pointer(value), ctypes.c_void_p))
        params = (ctypes.c_void_p * len(pointers))(*pointers)
        return _MarshalledArguments(params, values)

    def launch(
        self,
        *,
        grid: int,
        block: int,
        stream: int,
        arguments: list[torch.Tensor | int | float],
        cache_key: tuple[Any, ...],
    ) -> None:
        if cache_key != self._cached_key:
            self._cached_key = cache_key
            self._cached_arguments = self._marshal(arguments)
        assert self._cached_arguments is not None
        _check_cuda(
            self._cuda.cuLaunchKernel(
                self._function,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                ctypes.c_void_p(stream),
                self._cached_arguments.params,
                None,
            ),
            "cuLaunchKernel",
        )

    def num_registers(self) -> int:
        """Return registers allocated per thread by the CUDA compiler."""

        value = ctypes.c_int()
        _check_cuda(
            self._cuda.cuFuncGetAttribute(ctypes.byref(value), 4, self._function),
            "cuFuncGetAttribute(NUM_REGS)",
        )
        return int(value.value)


_LAUNCHERS: dict[tuple[int, int, int, SparseRSNNStrategy], _KernelLauncher] = {}


def _launcher_for(
    device: torch.device, strategy: SparseRSNNStrategy
) -> _KernelLauncher:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device_index)
    key = (device_index, major, minor, strategy)
    launcher = _LAUNCHERS.get(key)
    if launcher is None:
        with torch.cuda.device(device_index):
            cubin = _compile_cubin(_CUDA_SOURCE, f"sm_{major}{minor}")
            launcher = _KernelLauncher(cubin, _STRATEGY_FUNCTIONS[strategy])
        _LAUNCHERS[key] = launcher
    return launcher


@dataclass(frozen=True)
class SparseRSNNProvenance:
    """Identify the experiment lineage integrated by this implementation."""

    run: str = "run_v2b"
    confirmed_variant: str = "gen8_peer2_launch_fastpath_repair"
    async_variant: str = "gen12_peer2_nosync_fix"
    cyclic_interval_variant: str = "gen3_peer7_interval_poolref"
    hybrid_rangegate_variant: str = "gen6_peer1_hybrid_rangegate"
    ed_int32_natural_variant: str = "gen5_peer7_ed_int32_natural"
    confirmed_speedup: float = 23.73988443345872
    evaluation_units: int = 456


class _CyclicIntervalRSNNCuda(nn.Module):
    """Run a fused sparse recurrent sequence for cyclic interval activity.

    The source connection uses Btorch's ``(source, destination)`` convention.
    Internally it is transposed to destination-major CSR.  One CUDA thread owns
    one destination row and keeps current and voltage in registers for the full
    horizon.  Sparse values are accumulated in ascending CSR order.

    Args:
        connection: Square SciPy sparse connection matrix.
        current_decay: Multiplicative current decay per step.
        voltage_decay: Multiplicative voltage decay per step.
        block_size: CUDA threads per block.
        device: CUDA device for topology and outputs.
        strategy: Sparse aggregation strategy. ``"hybrid_rangegate"`` is the
            validated default. ``"cyclic_interval"`` uses the baseline binary
            search path. ``"ed_int32_natural"`` derives range gates directly
            from the int32 CSR indices.
        reuse_outputs: Reuse the three final-state buffers across calls.  Set
            this to ``False`` when callers retain outputs from multiple calls.

    Notes:
        This module is inference-only and supports float32 values.  Launches are
        asynchronous on the current PyTorch stream.  Reading returned tensors or
        recording and synchronizing a later CUDA event provides completion.
    """

    provenance = SparseRSNNProvenance()

    def __init__(
        self,
        connection: scipy.sparse.sparray,
        *,
        current_decay: float = 0.92,
        voltage_decay: float = 0.95,
        block_size: int = 256,
        device: torch.device | str = "cuda",
        strategy: SparseRSNNStrategy = "hybrid_rangegate",
        reuse_outputs: bool = True,
    ) -> None:
        super().__init__()
        if strategy not in _STRATEGY_FUNCTIONS:
            choices = ", ".join(sorted(_STRATEGY_FUNCTIONS))
            raise ValueError(f"strategy must be one of: {choices}")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("CyclicIntervalRSNN requires a CUDA device")
        if connection.ndim != 2 or connection.shape[0] != connection.shape[1]:
            raise ValueError("connection must be a square sparse matrix")
        if block_size <= 0 or block_size > 1024:
            raise ValueError("block_size must lie in [1, 1024]")

        recurrent = connection.transpose().tocsr().astype(np.float32)
        recurrent.sum_duplicates()
        recurrent.sort_indices()
        n_neuron = int(recurrent.shape[0])
        if n_neuron >= 1 << 30 or recurrent.nnz >= 1 << 31:
            raise ValueError("topology exceeds the int32 kernel index range")

        degree = np.diff(recurrent.indptr)
        nonempty = degree > 0
        span_lo = np.zeros(n_neuron, dtype=np.int32)
        span_hi = np.zeros(n_neuron, dtype=np.int32)
        row_start = recurrent.indptr[:-1]
        span_lo[nonempty] = recurrent.indices[row_start[nonempty]]
        span_hi[nonempty] = recurrent.indices[recurrent.indptr[1:][nonempty] - 1]

        self.n_neuron = n_neuron
        self.nnz = int(recurrent.nnz)
        self.current_decay = float(current_decay)
        self.voltage_decay = float(voltage_decay)
        self.block_size = int(block_size)
        self.strategy = strategy
        self.reuse_outputs = bool(reuse_outputs)
        self.register_buffer(
            "row_ptr",
            torch.as_tensor(recurrent.indptr, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "col_idx",
            torch.as_tensor(recurrent.indices, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "values",
            torch.as_tensor(recurrent.data, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "span_lo", torch.as_tensor(span_lo, dtype=torch.int32, device=device)
        )
        self.register_buffer(
            "span_hi", torch.as_tensor(span_hi, dtype=torch.int32, device=device)
        )
        self._output_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = (
            None
        )

    def _apply(self, fn, recurse: bool = True):
        self._output_cache = None
        return super()._apply(fn, recurse=recurse)

    def _outputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.reuse_outputs or self._output_cache is None:
            outputs = tuple(
                torch.empty(
                    self.n_neuron, device=self.values.device, dtype=torch.float32
                )
                for _ in range(3)
            )
            if self.reuse_outputs:
                self._output_cache = outputs
            return outputs
        return self._output_cache

    def forward(
        self,
        base_start: int,
        active_count: int,
        steps: int,
        *,
        stride: int = 104729,
        clone_outputs: bool = False,
        synchronize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the complete recurrent horizon.

        Args:
            base_start: First active neuron at step zero.
            active_count: Size of the contiguous cyclic active interval.
            steps: Number of recurrent updates.
            stride: Cyclic interval displacement per step.
            clone_outputs: Return independent tensors instead of reusable views.
            synchronize: Synchronize the current CUDA stream before returning.

        Returns:
            Final ``(spikes, voltage, current)`` float32 tensors.
        """

        n = self.n_neuron
        if not 0 <= base_start < n:
            raise ValueError("base_start must lie in [0, n_neuron)")
        if not 1 <= active_count <= n:
            raise ValueError("active_count must lie in [1, n_neuron]")
        if steps <= 0:
            raise ValueError("steps must be positive")
        stride %= n
        last_lo = (base_start + (steps - 1) * stride) % n
        last_hi = last_lo + active_count
        last_wrap = last_hi - n if last_hi > n else 0
        spikes, voltage, current = self._outputs()
        stream = torch.cuda.current_stream(self.values.device)
        launcher = _launcher_for(self.values.device, self.strategy)
        arguments: list[torch.Tensor | int | float] = [
            self.row_ptr,
            self.col_idx,
            self.values,
            self.span_lo,
            self.span_hi,
            spikes,
            voltage,
            current,
            n,
            active_count,
            base_start,
            stride,
            steps,
            self.current_decay,
            self.voltage_decay,
            last_lo,
            last_hi,
            last_wrap,
        ]
        launcher.launch(
            grid=math.ceil(n / self.block_size),
            block=self.block_size,
            stream=stream.cuda_stream,
            arguments=arguments,
            cache_key=(
                self.row_ptr.data_ptr(),
                self.col_idx.data_ptr(),
                self.values.data_ptr(),
                self.span_lo.data_ptr(),
                self.span_hi.data_ptr(),
                spikes.data_ptr(),
                voltage.data_ptr(),
                current.data_ptr(),
                base_start,
                active_count,
                stride,
                steps,
                self.current_decay,
                self.voltage_decay,
                self.strategy,
            ),
        )
        outputs = (spikes, voltage, current)
        if clone_outputs:
            outputs = tuple(output.clone() for output in outputs)
        if synchronize:
            stream.synchronize()
        return outputs

    def num_registers(self) -> int:
        """Return registers allocated per CUDA thread."""

        return _launcher_for(self.values.device, self.strategy).num_registers()

    def extra_repr(self) -> str:
        return (
            f"n_neuron={self.n_neuron}, nnz={self.nnz}, "
            f"current_decay={self.current_decay}, "
            f"voltage_decay={self.voltage_decay}, "
            f"block_size={self.block_size}, strategy={self.strategy!r}, "
            f"reuse_outputs={self.reuse_outputs}"
        )


def interval_rsnn_reference(
    connection: scipy.sparse.sparray,
    *,
    base_start: int,
    active_count: int,
    steps: int,
    stride: int = 104729,
    current_decay: float = 0.92,
    voltage_decay: float = 0.95,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the matching Torch CSR recurrence for correctness reproduction."""

    device = torch.device(device)
    recurrent = connection.transpose().tocsr().astype(np.float32)
    recurrent.sum_duplicates()
    recurrent.sort_indices()
    n = int(recurrent.shape[0])
    weight = torch.sparse_csr_tensor(
        torch.as_tensor(recurrent.indptr, dtype=torch.int64, device=device),
        torch.as_tensor(recurrent.indices, dtype=torch.int64, device=device),
        torch.as_tensor(recurrent.data, dtype=torch.float32, device=device),
        size=recurrent.shape,
        device=device,
        check_invariants=False,
    )
    stride %= n
    voltage = torch.zeros(n, device=device)
    current = torch.zeros_like(voltage)
    spikes = torch.zeros_like(voltage)
    base = torch.arange(active_count, device=device, dtype=torch.int64)
    base = (base + base_start) % n
    for step in range(steps):
        active = (base + step * stride) % n
        spikes.zero_()
        spikes.index_fill_(0, active, 1.0)
        current = current_decay * current + torch.sparse.mm(
            weight, spikes[:, None]
        ).squeeze(1)
        voltage = voltage_decay * voltage + current
        voltage.index_fill_(0, active, 0.0)
    return spikes, voltage, current


__all__ = [
    "SparseRSNNStrategy",
    "SparseRSNNProvenance",
    "_CyclicIntervalRSNNCuda",
    "interval_rsnn_reference",
]
