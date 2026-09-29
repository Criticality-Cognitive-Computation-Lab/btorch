# Generated-code bundle

Copy-paste-able snapshots of the same neuron (`../kernels.py`,
`neuron_python.py` here) as emitted by three systems, so the codegen can be
compared or shared without rerunning anything.

| file | what it is |
|------|------------|
| `neuron_python.py` | the reference neuron: single-step `core_step`, `eager_loop`, `eager_loop_save` (torch / eager) |
| `inductor_inference_triton.py` | `torch.compile(eager_loop)` inference, inductor `output_code` |
| `inductor_training_triton.py` | `torch.compile(eager_loop)` training, inductor `output_code` |
| `inductor_save_history_triton.py` | `torch.compile(eager_loop_save)` training — the "return the history" experiment |
| `flexsn_inference_triton.py` | FlexSN `backend="triton"` inference kernel (inference + final-state) |
| `flexsn_training_triton.py` | FlexSN `backend="triton"` forward + backward kernels |

Snapshot: `T=4`, `N=32768`, fp32, RTX 5090 (sm_120), torch 2.11.0, triton 3.6.0,
spikingjelly (FlexSN) checkout at `/home/fanqixuan/src/spikingjelly`.

## The two training strategies

- **Inductor** (`inductor_training_triton.py`): forward saves only the
  **inputs** (`x, y, v0, rho0`); the backward **recomputes** the whole neuron
  from them. 2 forward + 3 backward kernels.
- **FlexSN** (`flexsn_training_triton.py`): forward **checkpoints** the per-step
  residuals `(rho_prev, h, sigmoid)` (plus spikes/states); backward reloads
  them. 1 forward + 1 backward kernel.

So FlexSN spends ~`7·T·N` extra forward writes to avoid a recompute pass;
inductor spends a recompute pass to avoid those writes.

## The `save_history` experiment

`eager_loop_save` returns `h`, `rho_prev`, `sigmoid(y)` from the compiled
forward, on the theory that autograd would then keep them instead of
rematerializing. It does **not** work: `inductor_save_history_triton.py` still
contains the entire forward in the backward kernel (`h, ge, spike, ...` in its
source nodes); the returned history only shows up as extra `tangent_5..7`
inputs. The rematerialization is AOTAutograd's default
`min_cut_rematerialization_partition`, and no amount of "returning" changes it —
only a manual `autograd.Function` (or a custom partitioner) does.

Regenerate all of the above with:

```bash
python benchmarks/flexsn_vs_compile/dump_triton.py --T 4 --N 32768
```
