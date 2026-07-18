"""CUDA graphs for a recurrent LIF network (RSNN): inference and training,
timed.

A small RSNN is launch-bound -- the per-step kernels are tiny and the CPU cannot
queue them fast enough to keep the GPU busy. CUDA graphs collapse the many small
launches of the time loop into a single replay. Timed two ways below (T=100,
B=32, hidden=128; numbers from one GPU, re-run for yours):

Simulation (inference) -- flip on ``cudagraph=True`` and run the RSNN forward:

    approach          step ms   speedup
    plain (eager)      18.0      1.00x
    cudagraph=True      5.7      3.16x

``cudagraph=True`` is a one-flag speedup for inference. It composes with
``cpu_offload`` and ``torch.compile``, but is inference-only -- capture cannot
record an autograd graph. For training, use ``torch.compile(mode=
"reduce-overhead")`` or a manual whole-step capture:

Training -- fit a sine wave with Adam (setup = compile / capture, one-time):

    approach                        setup ms   step ms   speedup
    plain (eager)                          0     40.0      1.00x
    torch.compile (reduce-overhead)     8500      8.5      4.71x
    manual cudagraph                     350     11.2      3.57x

(reduce-overhead fuses kernels *and* cudagraphs them, so it beats the manual
whole-step capture, whose replay is one launch but of unfused kernels.)

--- CUDA graph key points (why the code below is shaped the way it is) --------
 * Capture records kernels over FIXED addresses; replay re-runs them. So every
   input / gradient / state tensor must live in a static buffer you copy new
   data into -- never a freshly allocated one.
 * Inference: cudagraph=True captures the forward loop. This flag is
   inference-only, but composes with cpu_offload and torch.compile.
 * Training: capture the WHOLE step (fwd + loss + backward + optim) as one graph.
     - grads static:      opt.zero_grad(set_to_none=False)
     - optimizer static:  Adam(..., capturable=True)  (device-side step counter)
     - state static:      do NOT reset_net_state inside the step (host copy +
                          rebind); restore a persistent start state instead.
 * Capture needs a side-stream warmup first, and a graph is valid only for the
   shape it saw (a new shape must be re-captured).

What a CUDA graph cannot capture at all (torch docs [1]): CPU<->GPU syncs such as
``.item()`` / ``.cpu()`` / ``print(tensor)``; data-dependent control flow like
``if x.sum() > 0`` (use ``torch.cond``); dynamic shapes; and CPU work, which is
silently elided on replay rather than re-run.

[1] https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
------------------------------------------------------------------------------
"""

import itertools
import math
import time

import torch
from torch import nn

from btorch.models import environ
from btorch.models.functional import (
    init_net_state,
    named_hidden_states,
    reset_net_state,
    set_hidden_states,
)
from btorch.models.linear import DenseConn
from btorch.models.neurons.lif import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import ExponentialPSC
from btorch.utils.bench import do_bench


DEVICE, DTYPE = "cuda", torch.float32
T, B, HIDDEN = 100, 32, 128  # long sequence, small net -> launch-bound
STEPS = 200  # timed training steps
environ.set(dt=1.0)  # integration step for the neuron/synapse ODEs


def make_rsnn(**rsnn_kwargs):
    """A recurrent LIF layer: LIF neurons with a dense recurrent synapse. Extra
    kwargs (``cudagraph``, ``cpu_offload``, ``chunk_size``, ...) go to RecurrentNN."""
    neuron = LIF(n_neuron=HIDDEN, v_threshold=1.0, v_reset=0.0, tau=20.0)
    recurrent = DenseConn(HIDDEN, HIDDEN, bias=None)  # dense recurrent weights
    synapse = ExponentialPSC(n_neuron=HIDDEN, tau_syn=5.0, linear=recurrent)
    return RecurrentNN(
        neuron=neuron,
        synapse=synapse,
        step_mode="m",
        update_state_names=("neuron.v",),  # record membrane potential per step
        **rsnn_kwargs,
    )


# ===========================================================================
# Simulation (inference): run the RSNN forward, no training.
# ===========================================================================
DRIVE = torch.randn(T, B, HIDDEN, device=DEVICE, dtype=DTYPE) * 0.5  # input current


def build_rsnn(**kwargs):
    torch.manual_seed(0)  # identical weights -> every variant produces the same output
    rsnn = make_rsnn(**kwargs).to(DEVICE, DTYPE)
    init_net_state(rsnn, batch_size=B, device=DEVICE, dtype=DTYPE)
    return rsnn


def run_sim(name, **rsnn_kwargs):
    rsnn = build_rsnn(**rsnn_kwargs)

    @torch.no_grad()
    def sim():
        reset_net_state(rsnn, batch_size=B)
        return rsnn(DRIVE)

    # do_bench warms up first, so cudagraph=True captures before the timed reps.
    step_ms = do_bench(sim, warmup=5, rep=50, return_mode="median")
    return name, step_ms


# ===========================================================================
# Training: fit a sine wave. Grad-recording, so cudagraph=True (inference-only)
# does not apply -- use torch.compile(reduce-overhead) or a whole-step capture.
# ===========================================================================
class SineSNN(nn.Module):
    """Linear drive -> recurrent LIF layer -> linear readout of the membrane
    potential, producing one value per timestep."""

    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(1, HIDDEN)
        self.rsnn = make_rsnn()
        self.fc_out = nn.Linear(HIDDEN, 1)

    def forward(self, x):  # x: (T, B, 1) -> (T, B, 1)
        _, states = self.rsnn(self.fc_in(x))
        return self.fc_out(states["neuron.v"])


def build_model():
    torch.manual_seed(0)  # identical init -> the three training runs are comparable
    model = SineSNN().to(DEVICE, DTYPE)
    init_net_state(model, batch_size=B, device=DEVICE, dtype=DTYPE)
    # capturable=True keeps Adam's state on-device so its step is CUDA-graphable.
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, capturable=True)
    return model, opt


_t = torch.arange(T, device=DEVICE, dtype=DTYPE) / T
TARGET = torch.sin(2 * math.pi * 3 * _t)[:, None, None].expand(T, B, 1).contiguous()
CLOCK = _t[:, None, None].expand(T, B, 1).contiguous()


def batch_input(step):
    """A clock ramp plus fresh per-step noise, so successive batches differ."""
    gen = torch.Generator(device=DEVICE).manual_seed(step)
    return CLOCK + 0.05 * torch.randn(
        T, B, 1, device=DEVICE, dtype=DTYPE, generator=gen
    )


def loss_of(out):
    return ((out - TARGET) ** 2).mean()


def run_plain():
    model, opt = build_model()
    batches = itertools.count()

    def step():
        opt.zero_grad()
        reset_net_state(model, batch_size=B)
        loss = loss_of(model(batch_input(next(batches))))
        loss.backward()
        opt.step()

    step_ms = do_bench(step, warmup=5, rep=STEPS, return_mode="median")
    return "plain (eager)", 0.0, step_ms


def run_compile():
    model, opt = build_model()
    compiled = torch.compile(model, mode="reduce-overhead")
    batches = itertools.count()

    def step():
        torch.compiler.cudagraph_mark_step_begin()  # Trees reuse buffers per step
        opt.zero_grad()
        reset_net_state(model, batch_size=B)
        loss = loss_of(compiled(batch_input(next(batches))))
        loss.backward()
        opt.step()

    torch.cuda.synchronize()
    start = time.perf_counter()
    step()  # first call triggers compilation
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - start) * 1e3

    step_ms = do_bench(step, warmup=5, rep=STEPS, return_mode="median")
    return "torch.compile (reduce-overhead)", compile_ms, step_ms


def run_cudagraph():
    model, opt = build_model()
    x = torch.zeros(T, B, 1, device=DEVICE, dtype=DTYPE)  # static input buffer

    # A persistent state to start every step from. reset_net_state can't go inside
    # the captured step -- it rebinds the state buffers and does a host copy, both
    # of which break capture. Snapshot the reset (zero) state once and restore it;
    # the cell reads it at step 0 and rebinds away, so it stays at a fixed address.
    reset_net_state(model, batch_size=B)
    start_state = {k: v.clone() for k, v in named_hidden_states(model).items()}

    def step_body():
        opt.zero_grad(set_to_none=False)  # keep .grad buffers at fixed addresses
        set_hidden_states(model, start_state)
        loss = loss_of(model(x))
        loss.backward()
        opt.step()

    torch.cuda.synchronize()
    start = time.perf_counter()
    x.copy_(batch_input(0))
    side = torch.cuda.Stream()  # capture requires a prior side-stream warmup
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(5):
            step_body()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step_body()
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter() - start) * 1e3

    batches = itertools.count()

    def step():
        x.copy_(batch_input(next(batches)))  # new batch into the static input buffer
        graph.replay()  # one launch runs forward + backward + optim.step

    step_ms = do_bench(step, warmup=5, rep=STEPS, return_mode="median")
    return "manual cudagraph", capture_ms, step_ms


def main():
    if not torch.cuda.is_available():
        raise SystemExit("this example needs a CUDA device")

    # --- Simulation (inference): the cudagraph=True toggle vs plain eager. ---
    sims = [run_sim("plain (eager)"), run_sim("cudagraph=True", cudagraph=True)]
    base = sims[0][1]
    print(f"\nsimulation (inference)\n{'approach':<24}{'step ms':>9}{'speedup':>9}")
    print("-" * 42)
    for name, step_ms in sims:
        print(f"{name:<24}{step_ms:>9.2f}{base / step_ms:>8.2f}x")

    # --- Training: cudagraph=True is inference-only; these two work with grad. ---
    runs = [run_plain(), run_compile(), run_cudagraph()]
    base = runs[0][2]
    print(f"\ntraining\n{'approach':<34}{'setup ms':>9}{'step ms':>9}{'speedup':>9}")
    print("-" * 61)
    for name, setup_ms, step_ms in runs:
        print(f"{name:<34}{setup_ms:>9.0f}{step_ms:>9.2f}{base / step_ms:>8.2f}x")


if __name__ == "__main__":
    main()
