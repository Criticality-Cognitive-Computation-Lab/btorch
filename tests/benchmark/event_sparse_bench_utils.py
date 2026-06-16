from __future__ import annotations

from dataclasses import dataclass

import torch

from btorch.backend.triton import (
    dense_spike_to_spike_list,
    post_span_spmm_from_spike_list,
    pre_span_spmm_from_spike_list,
)


@dataclass(frozen=True)
class EventSparseBenchConfig:
    batch_size: int
    n_pre: int
    n_post: int
    fanout: int
    active_ratio: float
    dtype: torch.dtype = torch.float32
    device: str = "cuda"
    seed: int = 0


@dataclass(frozen=True)
class EventSparseCase:
    spike: torch.Tensor
    spike_count: torch.Tensor
    spike_ind: torch.Tensor
    dense_weight: torch.Tensor
    row_length: torch.Tensor
    ind: torch.Tensor
    weight: torch.Tensor
    batch_size: int
    n_pre: int
    n_post: int
    fanout: int
    active_ratio: float


def build_event_sparse_case(cfg: EventSparseBenchConfig) -> EventSparseCase:
    """Build a benchmark case using presynaptic-row padded sparse storage."""
    if cfg.fanout <= 0:
        raise ValueError("fanout must be positive.")
    if cfg.fanout > cfg.n_post:
        raise ValueError("fanout must be <= n_post.")
    if not (0.0 <= cfg.active_ratio <= 1.0):
        raise ValueError("active_ratio must be in [0, 1].")

    generator = torch.Generator(device=cfg.device)
    generator.manual_seed(cfg.seed)

    row_length = torch.full(
        (cfg.n_pre,),
        cfg.fanout,
        device=cfg.device,
        dtype=torch.int64,
    )
    ind_rows = []
    weight_rows = []
    dense_weight = torch.zeros(
        (cfg.n_pre, cfg.n_post), device=cfg.device, dtype=cfg.dtype
    )
    for pre in range(cfg.n_pre):
        # randperm gives unique targets, which keeps the dense reference easy
        # to reason about and avoids accidental duplicate multapses.
        targets = torch.randperm(cfg.n_post, generator=generator, device=cfg.device)[
            : cfg.fanout
        ]
        values = torch.randn(
            (cfg.fanout,),
            generator=generator,
            device=cfg.device,
            dtype=cfg.dtype,
        )
        values = 0.1 * values / max(cfg.fanout, 1)
        ind_rows.append(targets)
        weight_rows.append(values)
        dense_weight[pre, targets] = values

    ind = torch.stack(ind_rows, dim=0)
    weight = torch.stack(weight_rows, dim=0)

    spike_mask = torch.rand(
        (cfg.batch_size, cfg.n_pre),
        generator=generator,
        device=cfg.device,
        dtype=cfg.dtype,
    ) < cfg.active_ratio
    spike = spike_mask.to(cfg.dtype)
    spike_count = spike_mask.sum(dim=1, dtype=torch.int32)
    max_spikes = max(int(spike_count.max().item()), 1)
    spike_ind = torch.empty(
        (cfg.batch_size, max_spikes), device=cfg.device, dtype=torch.int64
    )
    for batch in range(cfg.batch_size):
        active = torch.nonzero(spike_mask[batch], as_tuple=False).flatten()
        if active.numel() > 0:
            spike_ind[batch, : active.numel()] = active

    return EventSparseCase(
        spike=spike,
        spike_count=spike_count,
        spike_ind=spike_ind,
        dense_weight=dense_weight,
        row_length=row_length,
        ind=ind,
        weight=weight,
        batch_size=cfg.batch_size,
        n_pre=cfg.n_pre,
        n_post=cfg.n_post,
        fanout=cfg.fanout,
        active_ratio=cfg.active_ratio,
    )


class TorchDenseEventModule(torch.nn.Module):
    """Naive dense reference using spike @ W."""

    def __init__(self, dense_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("dense_weight", dense_weight)

    def forward(self, spike: torch.Tensor) -> torch.Tensor:
        return spike @ self.dense_weight


class TorchSparseEventModule(torch.nn.Module):
    """PyTorch sparse COO reference using all sparse matrix non-zeros."""

    def __init__(
        self,
        row_length: torch.Tensor,
        ind: torch.Tensor,
        weight: torch.Tensor,
        *,
        n_pre: int,
        n_post: int,
    ):
        super().__init__()
        rows = torch.arange(n_pre, device=ind.device, dtype=torch.int64)
        rows = rows[:, None].expand_as(ind)
        mask = torch.arange(ind.shape[1], device=ind.device)[None, :] < row_length[
            :, None
        ]
        # Store W.T so the forward path can use torch.sparse.mm(W.T, spike.T).
        indices = torch.stack([ind[mask], rows[mask]], dim=0)
        values = weight[mask]
        sparse_weight_t = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(n_post, n_pre),
            device=weight.device,
            dtype=weight.dtype,
        ).coalesce()
        self.register_buffer("sparse_weight_t", sparse_weight_t)

    def forward(self, spike: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self.sparse_weight_t, spike.T).T


class TritonPreSpanSpikeListModule(torch.nn.Module):
    """Benchmark wrapper for pre-span consumption of an existing spike list."""

    def __init__(
        self,
        row_length: torch.Tensor,
        ind: torch.Tensor,
        weight: torch.Tensor,
        *,
        n_post: int,
    ):
        super().__init__()
        self.register_buffer("row_length", row_length)
        self.register_buffer("ind", ind)
        self.register_buffer("weight", weight)
        self.n_post = n_post

    def forward(self, spike_count: torch.Tensor, spike_ind: torch.Tensor):
        return pre_span_spmm_from_spike_list(
            spike_count,
            spike_ind,
            self.row_length,
            self.ind,
            self.weight,
            size_m=self.n_post,
        )


class TritonPostSpanSpikeListModule(torch.nn.Module):
    """Benchmark wrapper for post-span consumption of an existing spike list."""

    def __init__(
        self,
        row_length: torch.Tensor,
        ind: torch.Tensor,
        weight: torch.Tensor,
        *,
        n_post: int,
    ):
        super().__init__()
        self.register_buffer("row_length", row_length)
        self.register_buffer("ind", ind)
        self.register_buffer("weight", weight)
        self.n_post = n_post

    def forward(self, spike_count: torch.Tensor, spike_ind: torch.Tensor):
        return post_span_spmm_from_spike_list(
            spike_count,
            spike_ind,
            self.row_length,
            self.ind,
            self.weight,
            size_m=self.n_post,
        )


class DenseSpikeToListModule(torch.nn.Module):
    """Benchmark wrapper for dense spike compaction."""

    def forward(self, spike: torch.Tensor):
        spike_list = dense_spike_to_spike_list(spike)
        return spike_list.count, spike_list.ind


def build_provider_modules(
    case: EventSparseCase,
    *,
    include_compiled: bool = True,
) -> dict[str, torch.nn.Module]:
    """Build the benchmark providers for one event sparse case."""
    modules: dict[str, torch.nn.Module] = {
        "torch_dense": TorchDenseEventModule(case.dense_weight),
        "torch_sparse": TorchSparseEventModule(
            case.row_length,
            case.ind,
            case.weight,
            n_pre=case.n_pre,
            n_post=case.n_post,
        ),
        "spike_list_build": DenseSpikeToListModule(),
        "triton_pre_span_list": TritonPreSpanSpikeListModule(
            case.row_length,
            case.ind,
            case.weight,
            n_post=case.n_post,
        ),
        "triton_post_span_list": TritonPostSpanSpikeListModule(
            case.row_length,
            case.ind,
            case.weight,
            n_post=case.n_post,
        ),
    }

    if include_compiled and hasattr(torch, "compile"):
        modules["torch_dense_compile"] = torch.compile(
            TorchDenseEventModule(case.dense_weight)
        )
        modules["torch_sparse_compile"] = torch.compile(
            TorchSparseEventModule(
                case.row_length,
                case.ind,
                case.weight,
                n_pre=case.n_pre,
                n_post=case.n_post,
            )
        )
        modules["spike_list_build_compile"] = torch.compile(DenseSpikeToListModule())
        modules["triton_pre_span_list_compile"] = torch.compile(
            TritonPreSpanSpikeListModule(
                case.row_length,
                case.ind,
                case.weight,
                n_post=case.n_post,
            )
        )
        modules["triton_post_span_list_compile"] = torch.compile(
            TritonPostSpanSpikeListModule(
                case.row_length,
                case.ind,
                case.weight,
                n_post=case.n_post,
            )
        )

    return modules
