import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import scipy.sparse
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, schedule

from btorch.models.base import MemoryModule
from btorch.models.functional import reset_net_state
from btorch.models.linear import DenseConn, SparseConn, available_sparse_backends
from btorch.models.rnn import make_rnn
from btorch.utils.file import fig_path


class SparseRNNCell(MemoryModule):
    """RNN cell with dense input projection and sparse recurrent weights."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        W_x: torch.Tensor,
        W_h_sparse: scipy.sparse.sparray,
        b: torch.Tensor,
        sparse_backend: str,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_x = DenseConn(
            input_size,
            hidden_size,
            weight=W_x,
            bias=None,
            device=W_x.device,
            dtype=W_x.dtype,
        )
        self.W_h = SparseConn(
            W_h_sparse,
            bias=None,
            enforce_dale=False,
            sparse_backend=sparse_backend,
        )
        self.b = nn.Parameter(b.clone())

        self.register_memory("h", torch.zeros(1), hidden_size)
        self.init_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.h = torch.tanh(self.W_x(x) + self.W_h(self.h) + self.b)
        return self.h


@dataclass(frozen=True)
class ProfileConfig:
    device: torch.device
    backend: str
    seq_len: int
    batch_size: int
    input_size: int
    hidden_size: int
    density: float
    grad_checkpoint: bool
    unroll: int
    wait_steps: int
    warmup_steps: int
    active_steps: int
    repeat: int
    output_dir: Path


def _activity_list(device: torch.device) -> list[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def _make_sparse_weights(
    hidden_size: int, density: float, device: torch.device, dtype: torch.dtype
) -> scipy.sparse.sparray:
    W_h_dense = torch.eye(hidden_size, device=device, dtype=dtype)
    W_h_dense = W_h_dense + 0.01 * torch.randn(
        hidden_size, hidden_size, device=device, dtype=dtype
    )
    mask = torch.rand_like(W_h_dense) < density
    mask.fill_diagonal_(True)
    W_h_dense = W_h_dense * mask
    W_h_sparse = scipy.sparse.coo_array(W_h_dense.cpu().numpy())
    return W_h_sparse


def _trace_handler(output_dir: Path):
    def handler(prof):
        trace_path = output_dir / f"trace_{prof.step_num}.json"
        prof.export_chrome_trace(str(trace_path))

    return handler


def _write_summary(prof, output_dir: Path, device: torch.device, row_limit: int = 50):
    if device.type == "cuda":
        time_key = "self_cuda_time_total"
        mem_key = "self_cuda_memory_usage"
    else:
        time_key = "self_cpu_time_total"
        mem_key = "self_cpu_memory_usage"

    summary_path = output_dir / "summary.txt"
    table_time = prof.key_averages().table(sort_by=time_key, row_limit=row_limit)
    table_mem = prof.key_averages().table(sort_by=mem_key, row_limit=row_limit)
    summary_path.write_text(
        f"time ({time_key})\n{table_time}\n\nmemory ({mem_key})\n{table_mem}\n"
    )

    stacks_path = output_dir / f"stacks_{device.type}_memory.txt"
    prof.export_stacks(str(stacks_path), mem_key)


def _parse_args() -> ProfileConfig:
    backends = available_sparse_backends()
    parser = argparse.ArgumentParser(description="Profile a sparse RNN.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--backend", default=backends[0], choices=backends)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--unroll", type=int, default=4)
    parser.add_argument("--wait-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--active-steps", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = fig_path(__file__) / timestamp
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    return ProfileConfig(
        device=device,
        backend=args.backend,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        density=args.density,
        grad_checkpoint=args.grad_checkpoint,
        unroll=args.unroll,
        wait_steps=args.wait_steps,
        warmup_steps=args.warmup_steps,
        active_steps=args.active_steps,
        repeat=args.repeat,
        output_dir=output_dir,
    )


def _run_profile(cfg: ProfileConfig) -> None:
    torch.manual_seed(42)

    W_x = torch.randn(
        cfg.input_size, cfg.hidden_size, device=cfg.device, dtype=torch.float32
    )
    W_h_sparse = _make_sparse_weights(
        cfg.hidden_size, cfg.density, cfg.device, W_x.dtype
    )
    b = torch.zeros(cfg.hidden_size, device=cfg.device, dtype=W_x.dtype)

    rnn = make_rnn(
        SparseRNNCell, grad_checkpoint=cfg.grad_checkpoint, unroll=cfg.unroll
    )(
        cfg.input_size,
        cfg.hidden_size,
        W_x,
        W_h_sparse,
        b,
        sparse_backend=cfg.backend,
    )
    rnn.to(device=cfg.device)

    x = torch.randn(
        cfg.seq_len,
        cfg.batch_size,
        cfg.input_size,
        device=cfg.device,
        dtype=W_x.dtype,
        requires_grad=True,
    )

    schedule_cfg = schedule(
        wait=cfg.wait_steps,
        warmup=cfg.warmup_steps,
        active=cfg.active_steps,
        repeat=cfg.repeat,
    )
    total_steps = (cfg.wait_steps + cfg.warmup_steps + cfg.active_steps) * cfg.repeat

    with profile(
        activities=_activity_list(cfg.device),
        schedule=schedule_cfg,
        on_trace_ready=_trace_handler(cfg.output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
    ) as prof:
        for _ in range(total_steps):
            reset_net_state(rnn, batch_size=cfg.batch_size)
            out, _ = rnn.multi_step_forward(x)
            loss = out.sum()
            loss.backward()
            rnn.zero_grad(set_to_none=True)
            if cfg.device.type == "cuda":
                torch.cuda.synchronize()
            prof.step()

    _write_summary(prof, cfg.output_dir, cfg.device)


def main() -> None:
    cfg = _parse_args()
    _run_profile(cfg)


if __name__ == "__main__":
    main()
