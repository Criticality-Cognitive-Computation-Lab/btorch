"""Reproduce the Praxist fused CUDA sparse-RSNN result on local graphs.

The script never downloads data.  It uses the machine-local
``connectome_dataset`` catalog and compares the integrated CUDA kernel with an
unchanged Torch CSR recurrence under matched graph, activity, horizon, seed,
dtype, and CUDA stream timing.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from btorch.models.rnn import CyclicIntervalRSNN  # noqa: E402


ACTIVITY_HZ = (1.0, 3.0, 10.0, 30.0)
TIMESTEPS = (8, 128, 2048)
FLYWIRE_NEURONS = 138_639
MANDATORY_DATASETS = (
    "flywire_783",
    "fly_hemibrain",
    "microns_mm3",
    "multiarea_mam",
)
EXCLUDED_TOKENS = ("elegans", "c_elegans", "hcp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connectome-repo",
        type=Path,
        default=Path("/home/fanqixuan/src/connectome_dataset"),
        help="Existing local connectome_dataset checkout.",
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "aligned", "complete"),
        default="smoke",
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--activity-hz", type=float, nargs="+", default=ACTIVITY_HZ)
    parser.add_argument("--timesteps", type=int, nargs="+", default=TIMESTEPS)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/sparse_rnn/results/praxist_cuda_rsnn.jsonl"),
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup must be non-negative and repeat must be positive")
    if any(value <= 0 for value in args.activity_hz):
        parser.error("activity rates must be positive")
    if any(value <= 0 for value in args.timesteps):
        parser.error("timesteps must be positive")
    return args


def import_catalog(connectome_repo: Path):
    """Import the existing local catalog without installing a dependency."""

    if not connectome_repo.is_dir():
        raise FileNotFoundError(f"connectome repo not found: {connectome_repo}")
    sys.path.insert(0, str(connectome_repo))
    from connectome_dataset.catalog import load_catalog
    from connectome_dataset.graph_loader import load_graph_entry

    return load_catalog, load_graph_entry


def select_entries(
    catalog: list[dict], mode: str, requested: list[str] | None
) -> list[dict]:
    """Select the same local graph regimes used by the Praxist harness."""

    by_name = {str(entry["name"]): entry for entry in catalog}
    if requested:
        missing = sorted(set(requested) - set(by_name))
        if missing:
            raise ValueError(f"unknown catalog datasets: {missing}")
        return [by_name[name] for name in requested]
    if mode == "smoke":
        names = ("drosophila_medulla_1",)
        return [by_name[name] for name in names if name in by_name]
    if mode == "aligned":
        return [by_name[name] for name in MANDATORY_DATASETS if name in by_name]

    selected = []
    for entry in catalog:
        name = str(entry["name"])
        large = int(entry.get("n_rows", 0) or 0) >= FLYWIRE_NEURONS
        source_exists = Path(str(entry.get("source_path", ""))).exists()
        excluded = any(token in name.lower() for token in EXCLUDED_TOKENS)
        if source_exists and not excluded and (large or name in MANDATORY_DATASETS):
            selected.append(entry)
    return sorted(selected, key=lambda entry: str(entry["name"]))


class TorchCsrReference:
    """Unchanged eager Torch CSR recurrence used by the original evaluator."""

    def __init__(self, connection: scipy.sparse.sparray, device: torch.device):
        recurrent = connection.transpose().tocsr().astype(np.float32)
        recurrent.sum_duplicates()
        recurrent.sort_indices()
        self.n = int(recurrent.shape[0])
        self.device = device
        self.weight = torch.sparse_csr_tensor(
            torch.as_tensor(recurrent.indptr, dtype=torch.int64, device=device),
            torch.as_tensor(recurrent.indices, dtype=torch.int64, device=device),
            torch.as_tensor(recurrent.data, dtype=torch.float32, device=device),
            size=recurrent.shape,
            device=device,
            check_invariants=False,
        )

    def run(
        self, base_start: int, active_count: int, steps: int, stride: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        voltage = torch.zeros(self.n, device=self.device)
        current = torch.zeros_like(voltage)
        spikes = torch.zeros_like(voltage)
        base = torch.arange(active_count, device=self.device, dtype=torch.int64)
        base = (base + base_start) % self.n
        for step in range(steps):
            active = (base + step * stride) % self.n
            spikes.zero_()
            spikes.index_fill_(0, active, 1.0)
            current = 0.92 * current + torch.sparse.mm(
                self.weight, spikes[:, None]
            ).squeeze(1)
            voltage = 0.95 * voltage + current
            voltage.index_fill_(0, active, 0.0)
        return spikes, voltage, current


def timed_ms(run: Callable[[], object], warmup: int, repeat: int) -> float:
    """Measure one callable with stream-ordered CUDA events."""

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples)


def max_state_error(
    left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...]
) -> float:
    """Return the maximum float64 absolute error across final states."""

    return max(
        float((left_state.double() - right_state.double()).abs().max())
        for left_state, right_state in zip(left, right, strict=True)
    )


def metadata(device: torch.device) -> dict[str, object]:
    props = torch.cuda.get_device_properties(device)
    return {
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "provenance": CyclicIntervalRSNN.provenance.__dict__,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    load_catalog, load_graph_entry = import_catalog(args.connectome_repo)
    catalog = load_catalog()
    entries = select_entries(catalog, args.mode, args.datasets)
    if not entries:
        raise SystemExit("no local datasets matched the requested mode")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in entries:
        name = str(entry["name"])
        print(f"loading {name}", flush=True)
        connection = load_graph_entry(entry, catalog).tocsr().astype(np.float32)
        connection.sum_duplicates()
        reference = TorchCsrReference(connection, device)
        candidate = CyclicIntervalRSNN(connection, device=device)
        n = int(connection.shape[0])
        stride = 104729 % n or 1

        for activity_index, activity_hz in enumerate(args.activity_hz):
            active_count = min(n, max(1, int(round(n * activity_hz / 1000.0))))
            for timestep_index, steps in enumerate(args.timesteps):
                case_seed = args.seed + activity_index * 101 + timestep_index * 1009
                base_start = case_seed % n
                reference_run = lambda: reference.run(
                    base_start, active_count, steps, stride
                )
                candidate_run = lambda: candidate(
                    base_start, active_count, steps, stride=stride
                )

                reference_ms = timed_ms(reference_run, args.warmup, args.repeat)
                candidate_ms = timed_ms(candidate_run, args.warmup, args.repeat)
                reference_state = reference_run()
                reference_repeat = reference_run()
                candidate_state = candidate(
                    base_start,
                    active_count,
                    steps,
                    stride=stride,
                    clone_outputs=True,
                    synchronize=True,
                )
                record = {
                    "dataset": name,
                    "n_neuron": n,
                    "nnz": int(connection.nnz),
                    "activity_hz_target": activity_hz,
                    "activity_hz_realized": active_count / n * 1000.0,
                    "active_count": active_count,
                    "timesteps": steps,
                    "seed": case_seed,
                    "stride": stride,
                    "torch_csr_ms": reference_ms,
                    "cuda_fused_ms": candidate_ms,
                    "speedup": reference_ms / candidate_ms,
                    "cuda_fused_step_us": candidate_ms * 1000.0 / steps,
                    "correctness_error": max_state_error(
                        candidate_state, reference_state
                    ),
                    "reference_self_error": max_state_error(
                        reference_repeat, reference_state
                    ),
                    "registers_per_thread": candidate.num_registers(),
                }
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)

        # Let the next loop assignment release the Python wrappers. Explicitly
        # deleting loop-local names confuses static definite-assignment analysis
        # because the loop may execute again.
        del connection
        torch.cuda.empty_cache()

    with args.output.open("w", encoding="utf-8") as output:
        output.write(json.dumps({"metadata": metadata(device)}) + "\n")
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"wrote {len(records)} records to {args.output.resolve()}")


if __name__ == "__main__":
    main()
