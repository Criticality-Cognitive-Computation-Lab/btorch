"""CUDA graphs for a recurrent LIF network (RSNN): inference and training,
timed.

A small RSNN is launch-bound -- the per-step kernels are tiny and the CPU cannot
queue them fast enough to keep the GPU busy. CUDA graphs collapse the many small
launches of the time loop into a single replay.

1. Inference -- flip on ``cudagraph=True`` and run the RSNN forward:

       approach          step ms   speedup
       plain (eager)      16.6      1.00x
       cudagraph=True      5.7      2.90x

   One flag. It composes with ``cpu_offload`` and ``torch.compile``, but this flag
   is inference-only -- it does not record autograd. Grad recording itself *can* be
   graphed (torch.compile's reduce-overhead captures fwd+bwd), but the full training
   step (fwd + loss + backward + optimizer) needs to be captured by hand (section 2).

2. Training -- two DC inputs must produce two sine frequencies (0.5 -> 2 cycles,
   1.5 -> 3 cycles): the net learns to map input amplitude to output frequency.
   Adam + cosine LR; firing bounded to ~50 Hz by a per-neuron rate penalty. Each
   step alternates the two tasks, so a replay must copy the new input AND target
   into static buffers.

       approach                    setup ms   step ms   speedup
       plain (eager)                      0     45        1.00x
       torch.compile (no cudagraph)    ~9000     14        ~3.2x
       manual cudagraph                 ~300     11        ~4.1x

   plain and manual cudagraph train bit-identically (same kernels, weights match to
   the last bit). torch.compile is slower for a concrete reason, not a bug: you
   cannot torch.compile the *whole* step -- Dynamo graph-breaks at ``.backward()``
   ("does not support tracing Tensor.backward()"). Default mode compiles fwd and bwd
   (via aot_autograd) as separate launched graphs and runs the optimizer eagerly, so
   this launch-bound net keeps its per-op launch overhead; the manual capture
   replays the entire step in one launch. The one mode that WOULD cudagraph fwd+bwd,
   ``mode="reduce-overhead"``, errors on this net's partially-eager loop (see the
   commented run_compile_reduce_overhead).

--- CUDA graph key points (why the training code below is shaped the way it is) --
 * Capture records kernels over FIXED addresses; replay re-runs them. So every
   input / target / gradient / optimizer-state / model-state tensor must live in a
   static buffer you copy new data into -- never a freshly allocated one. Here
   load_task() copies the next task's input and target in before each replay.
 * Training capture is the whole step (fwd + loss + backward + optim) in one graph.
     - grads static:      opt.zero_grad(set_to_none=False)
     - optimizer static:  Adam(..., capturable=True)  (device-side step counter)
     - state static:      do NOT reset_net_state inside the step (host copy +
                          rebind); restore a persistent start state instead.
 * The warmup runs real opt.step()s, so BEFORE capturing, reset both the weights
   AND the optimizer state (Adam momentum + step) -- else the captured run starts
   from warmed-up momentum and does not match training from scratch.
 * A changing LR still works: make Adam's lr a device TENSOR and write the new
   value into it before each replay; the captured opt.step() reads it.

What a CUDA graph cannot capture at all (torch docs [1]): CPU<->GPU syncs such as
``.item()`` / ``.cpu()`` / ``print(tensor)``; data-dependent control flow like
``if x.sum() > 0`` (use ``torch.cond``); dynamic shapes; and CPU work, which is
silently elided on replay rather than re-run.

[1] https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
------------------------------------------------------------------------------
"""

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
environ.set(dt=1.0)  # integration step (ms) for the neuron/synapse ODEs


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
# Training: map two DC values to two sine frequencies with a recurrent LIF net.
# cudagraph=True (inference-only) does not apply -- capture the whole step by hand.
# ===========================================================================
NEURONS, BATCH = 256, 1
REC_GAIN = 1.2  # recurrent init gain: rich enough to oscillate, not enough to explode
STEPS, BASE_LR = 4000, 1.5e-3
# Penalize per-neuron firing above the budget (~50 Hz) to keep the net from exploding.
RATE_BUDGET, RATE_LAM = 0.05, 150.0
WINDOW_S = T * environ.get("dt") / 1000.0  # window length in seconds (dt in ms)


class SineSNN(nn.Module):
    """DC drive -> recurrent LIF layer -> linear readout of the membrane
    potential.

    Returns (readout, spikes); spikes feed the firing-rate penalty.
    """

    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(1, NEURONS)
        neuron = LIF(n_neuron=NEURONS, v_threshold=1.0, v_reset=0.0, tau=20.0)
        recurrent = DenseConn(NEURONS, NEURONS, bias=None)
        with torch.no_grad():
            recurrent.weight.mul_(REC_GAIN)
        synapse = ExponentialPSC(n_neuron=NEURONS, tau_syn=5.0, linear=recurrent)
        self.brain = RecurrentNN(
            neuron=neuron,
            synapse=synapse,
            step_mode="m",
            update_state_names=("neuron.v",),
        )
        self.fc_out = nn.Linear(NEURONS, 1)  # bias allowed

    def forward(
        self, x
    ):  # x: (T, BATCH, 1) -> readout (T, BATCH, 1), spikes (T, BATCH, N)
        spikes, states = self.brain(self.fc_in(x))
        return self.fc_out(states["neuron.v"]), spikes


def build_model():
    torch.manual_seed(0)  # identical init -> the three runs are comparable
    model = SineSNN().to(DEVICE, DTYPE)
    init_net_state(model, batch_size=BATCH, device=DEVICE, dtype=DTYPE)
    lr = torch.tensor(
        BASE_LR, device=DEVICE
    )  # device tensor so a schedule survives capture
    # capturable=True keeps Adam's state on-device so its step is CUDA-graphable.
    opt = torch.optim.Adam(model.parameters(), lr=lr, capturable=True)
    return model, opt, lr


# Two tasks: a constant DC drive of a given value must produce a sine of a given
# frequency. The network learns to map input amplitude -> output frequency.
_t = torch.arange(T, device=DEVICE, dtype=DTYPE) / T
DC_VALUES, FREQS = (0.5, 1.5), (2, 3)  # DC input value -> sine cycles over the window
INPUTS = [torch.full((T, BATCH, 1), v, device=DEVICE, dtype=DTYPE) for v in DC_VALUES]
TARGETS = [
    torch.sin(2 * math.pi * f * _t)[:, None, None].expand(T, BATCH, 1).contiguous()
    for f in FREQS
]

# Static input/target buffers the training step reads. Each step copies one task's
# data in; the CUDA graph capture bakes in THESE addresses, so every replay must
# copy the next task's input AND target into them -- exactly the copy CUDA graphs
# force on you (a fixed input/target would let you skip it and hide the pattern).
X = torch.zeros(T, BATCH, 1, device=DEVICE, dtype=DTYPE)
Y = torch.zeros(T, BATCH, 1, device=DEVICE, dtype=DTYPE)


def load_task(step):
    k = step % len(INPUTS)  # alternate the two tasks
    X.copy_(INPUTS[k])
    Y.copy_(TARGETS[k])


def cosine_lr(step):
    return BASE_LR * 0.5 * (1 + math.cos(math.pi * step / STEPS))


def loss_of(out, spikes):
    fit = (out - Y).pow(2).mean()  # Y is the current task's target
    over = torch.relu(spikes.mean(dim=0)[0] - RATE_BUDGET)  # per-neuron overage
    return fit + RATE_LAM * over.pow(2).sum()


def report(model):
    """Mean fit over both tasks (1.0 = perfect), and peak firing rate (Hz)."""
    corrs, max_hz = [], 0.0
    with torch.no_grad():
        for inp, tgt in zip(INPUTS, TARGETS):
            reset_net_state(model, batch_size=BATCH)
            out, spikes = model(inp)
            c = torch.corrcoef(torch.stack([out.flatten(), tgt.flatten()]))[0, 1]
            corrs.append(c.item())
            max_hz = max(max_hz, (spikes.sum(0)[0].max() / WINDOW_S).item())
    return sum(corrs) / len(corrs), max_hz


def run_plain():
    model, opt, lr = build_model()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(STEPS):
        load_task(i)  # copy this step's input + target into X, Y
        lr.fill_(cosine_lr(i))
        opt.zero_grad(set_to_none=False)
        reset_net_state(model, batch_size=BATCH)
        loss_of(*model(X)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # tame high-gain BPTT
        opt.step()
    torch.cuda.synchronize()
    step_ms = (time.perf_counter() - start) / STEPS * 1e3
    return ("plain (eager)", 0.0, step_ms, *report(model))


def run_compile():
    # Default mode fuses the forward (and, via aot_autograd, the backward) but does
    # NOT capture the whole step as one graph: fwd, bwd, and the eager optimizer each
    # still launch their kernels, so this launch-bound net keeps its per-op launch
    # overhead and lands just above the manual capture. reduce-overhead -- the mode
    # that WOULD cudagraph fwd+bwd -- errors here; see run_compile_reduce_overhead.
    model, opt, lr = build_model()
    forward = torch.compile(model)

    def step(i):
        load_task(i)
        lr.fill_(cosine_lr(i))
        opt.zero_grad(set_to_none=False)
        reset_net_state(model, batch_size=BATCH)
        loss_of(*forward(X)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    torch.cuda.synchronize()
    start = time.perf_counter()
    step(0)  # first call compiles
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - start) * 1e3

    start = time.perf_counter()
    for i in range(1, STEPS):
        step(i)
    torch.cuda.synchronize()
    step_ms = (time.perf_counter() - start) / (STEPS - 1) * 1e3
    return ("torch.compile", compile_ms, step_ms, *report(model))


# reduce-overhead graphs the compiled fwd+bwd, so in principle it should reach the
# manual capture's speed for free. It does NOT work for this net -- kept here,
# commented out, to show what was tried and why it fails on step 1's .backward():
#
#   RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten
#                 by a subsequent run
#
# The forward graph-breaks at self.brain(...) (multi_step_forward is
# @torch.compiler.disable'd to keep compile time O(1) in T), so Dynamo cudagraphs
# the post-break tail (fc_out) as its own segment. cudagraph-trees reuse that
# segment's pool on the next step's forward, clobbering the output the pending
# backward still needs. The two documented workarounds don't help: the step below
# already calls cudagraph_mark_step_begin() AND clones the outputs, and still fails
# -- a known torch limitation for hand-written backward loops (pytorch/pytorch
# #169545, #148439). The manual whole-step capture in run_cudagraph is the fix.
#
# def run_compile_reduce_overhead():
#     model, opt, lr = build_model()
#     forward = torch.compile(model, mode="reduce-overhead")
#
#     def step(i):
#         torch.compiler.cudagraph_mark_step_begin()  # documented workaround #1
#         load_task(i)
#         lr.fill_(cosine_lr(i))
#         opt.zero_grad(set_to_none=False)
#         reset_net_state(model, batch_size=BATCH)
#         out = tuple(t.clone() for t in forward(X))  # documented workaround #2
#         loss_of(*out).backward()  # <-- RuntimeError raised here on step i == 1
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         opt.step()
#
#     for i in range(STEPS):
#         step(i)
#     return ("reduce-overhead", 0.0, 0.0, *report(model))


def run_cudagraph():
    model, opt, lr = build_model()

    # Persistent zero state to start every step from -- reset_net_state can't run
    # inside the captured step (host copy + rebind). The cell reads this at step 0
    # and rebinds away, so it stays put at a fixed address.
    reset_net_state(model, batch_size=BATCH)
    start_state = {k: v.clone() for k, v in named_hidden_states(model).items()}
    init_weights = {k: v.clone() for k, v in model.state_dict().items()}

    def step_body():
        opt.zero_grad(set_to_none=False)
        set_hidden_states(model, start_state)
        loss_of(*model(X)).backward()  # reads the static X (input) and Y (target)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    torch.cuda.synchronize()
    start = time.perf_counter()
    load_task(0)  # warm up / capture on a real task, not the empty buffers
    side = torch.cuda.Stream()  # capture requires a prior side-stream warmup
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step_body()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    # Undo the warmup: restore weights AND zero Adam's warmed-up state, so the
    # captured run trains from scratch. Both in-place (fixed addresses).
    model.load_state_dict(init_weights)
    for p in model.parameters():
        st = opt.state[p]
        st["exp_avg"].zero_()
        st["exp_avg_sq"].zero_()
        st["step"].zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step_body()
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter() - start) * 1e3

    start = time.perf_counter()
    for i in range(STEPS):
        load_task(i)  # copy this step's input + target into the static X, Y buffers
        lr.fill_(cosine_lr(i))  # update lr in place; the captured opt.step() reads it
        graph.replay()  # one launch runs forward + backward + optim.step
    torch.cuda.synchronize()
    step_ms = (time.perf_counter() - start) / STEPS * 1e3
    return ("manual cudagraph", capture_ms, step_ms, *report(model))


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

    # --- Training: fit a sine; cudagraph=True can't (grad); capture by hand. ---
    runs = [run_plain(), run_compile(), run_cudagraph()]
    base = runs[0][2]
    header = (
        f"{'approach':<20}{'setup ms':>9}{'step ms':>9}"
        f"{'speedup':>9}{'corr':>7}{'maxHz':>7}"
    )
    print(f"\ntraining\n{header}")
    print("-" * len(header))
    for name, setup_ms, step_ms, corr, max_hz in runs:
        print(
            f"{name:<20}{setup_ms:>9.0f}{step_ms:>9.2f}"
            f"{base / step_ms:>8.2f}x{corr:>7.2f}{max_hz:>7.0f}"
        )


if __name__ == "__main__":
    main()
