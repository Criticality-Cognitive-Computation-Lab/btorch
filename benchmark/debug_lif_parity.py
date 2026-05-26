import argparse

import torch

from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.neurons.lif import LIF


def _build_input(
    steps: int,
    batch_size: int,
    n_neuron: int,
    *,
    device: str,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x_seq = torch.randn(
        (steps, batch_size, n_neuron),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return 0.2 * x_seq + 0.35


def _build_torch_lif(
    n_neuron: int,
    *,
    tau_ref: float | None,
    device: str,
    dtype: torch.dtype,
) -> LIF:
    return LIF(
        n_neuron=n_neuron,
        v_threshold=1.0,
        v_reset=0.0,
        c_m=1.0,
        tau=20.0,
        tau_ref=tau_ref,
        hard_reset=False,
        step_mode="s",
        device=device,
        dtype=dtype,
    )


def _build_triton_lif(
    n_neuron: int,
    *,
    tau_ref: float | None,
    device: str,
    dtype: torch.dtype,
) -> LIF:
    return LIF(
        n_neuron=n_neuron,
        v_threshold=1.0,
        v_reset=0.0,
        c_m=1.0,
        tau=20.0,
        tau_ref=tau_ref,
        hard_reset=False,
        backend="triton",
        step_mode="s",
        device=device,
        dtype=dtype,
    )


def _first_true_index(mask: torch.Tensor) -> tuple[int, ...] | None:
    idx = torch.nonzero(mask, as_tuple=False)
    if idx.numel() == 0:
        return None
    return tuple(int(v) for v in idx[0].tolist())


def debug_parity(
    *,
    steps: int,
    batch_size: int,
    n_neuron: int,
    tau_ref: float | None,
    dt: float,
    seed: int,
    device: str,
    dtype: torch.dtype,
) -> None:
    x_seq = _build_input(
        steps,
        batch_size,
        n_neuron,
        device=device,
        dtype=dtype,
        seed=seed,
    )

    torch_lif = _build_torch_lif(
        n_neuron,
        tau_ref=tau_ref,
        device=device,
        dtype=dtype,
    )
    triton_lif = _build_triton_lif(
        n_neuron,
        tau_ref=tau_ref,
        device=device,
        dtype=dtype,
    )

    init_net_state(torch_lif, batch_size=batch_size, device=device, dtype=dtype)
    init_net_state(triton_lif, batch_size=batch_size, device=device, dtype=dtype)

    for t in range(steps):
        x_t = x_seq[t]
        with torch.no_grad(), environ.context(dt=dt):
            spike_torch = torch_lif(x_t)
            spike_triton = triton_lif(x_t)

        mismatch = spike_torch != spike_triton
        mismatch_idx = _first_true_index(mismatch)
        if mismatch_idx is not None:
            b_idx, n_idx = mismatch_idx
            print(f"First spike mismatch at t={t}, batch={b_idx}, neuron={n_idx}")
            print(
                f"spike_torch={float(spike_torch[b_idx, n_idx])}, "
                f"spike_triton={float(spike_triton[b_idx, n_idx])}"
            )
            print(
                f"v_torch={float(torch_lif.v[b_idx, n_idx]):.8f}, "
                f"v_triton={float(triton_lif.v[b_idx, n_idx]):.8f}"
            )
            print(
                f"v_threshold={float(torch_lif.v_threshold[n_idx]):.8f}, "
                f"v_reset={float(torch_lif.v_reset[n_idx]):.8f}"
            )
            if tau_ref is not None:
                print(
                    f"ref_torch={float(torch_lif.refractory[b_idx, n_idx]):.8f}, "
                    f"ref_triton={float(triton_lif.refractory[b_idx, n_idx]):.8f}"
                )
            return

        v_close = torch.allclose(torch_lif.v, triton_lif.v, atol=1e-6, rtol=0.0)
        if not v_close:
            v_diff = (torch_lif.v - triton_lif.v).abs()
            idx = _first_true_index(v_diff > 1e-6)
            assert idx is not None
            b_idx, n_idx = idx
            print(f"First voltage mismatch at t={t}, batch={b_idx}, neuron={n_idx}")
            print(
                f"v_torch={float(torch_lif.v[b_idx, n_idx]):.8f}, "
                f"v_triton={float(triton_lif.v[b_idx, n_idx]):.8f}, "
                f"abs_diff={float(v_diff[b_idx, n_idx]):.8e}"
            )
            if tau_ref is not None:
                ref_diff = (torch_lif.refractory - triton_lif.refractory).abs()
                print(
                    f"ref_torch={float(torch_lif.refractory[b_idx, n_idx]):.8f}, "
                    f"ref_triton={float(triton_lif.refractory[b_idx, n_idx]):.8f}, "
                    f"ref_abs_diff={float(ref_diff[b_idx, n_idx]):.8e}"
                )
            return

    print("No mismatch found.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug Torch vs Triton LIF parity.")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-neuron", type=int, default=3000)
    parser.add_argument("--tau-ref", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    debug_parity(
        steps=args.steps,
        batch_size=args.batch_size,
        n_neuron=args.n_neuron,
        tau_ref=args.tau_ref,
        dt=args.dt,
        seed=args.seed,
        device=args.device,
        dtype=torch.float32,
    )


if __name__ == "__main__":
    main()
