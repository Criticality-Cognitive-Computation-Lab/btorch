import pytest
import scipy.sparse
import torch

from btorch.models import environ, functional
from btorch.models.linear import (
    DenseConn,
    SparseConn,
    SparseConstrainedConn,
    available_sparse_backends,
)
from btorch.models.neurons.alif import ALIF, ELIF
from btorch.models.neurons.glif import GLIF3
from tests.utils.compile import compile_or_skip


if not torch.cuda.is_available():
    pytest.skip("CUDA not available - skipping entire module", allow_module_level=True)


@pytest.mark.parametrize("backend", available_sparse_backends())
@pytest.mark.parametrize("use_compile", [False, True])
def test_cudagraph_linear(backend: str, use_compile: bool):
    """All connection classes match dense behavior with CUDA graphs."""
    if backend == "native":
        pytest.xfail(
            "Native sparse backward pass is currently incompatible with CUDA graphs."
        )

    torch.manual_seed(42)

    device = torch.device("cuda")

    # Create a small dense weight matrix
    W = torch.tensor([[1.0, 2.0, 0.0], [0.0, 3.0, -1.0], [2.0, 0.0, 1.0]])  # 3x3 matrix

    # Test inputs: single vector and batched vectors.
    x = torch.tensor([1.0, 2.0, 3.0])
    x_batch = torch.stack([x, x + 1.0], dim=0)

    # 1. Dense connection
    dense = DenseConn(3, 3, weight=W, bias=None).to(device)

    # 2. Sparse COO connection (convert dense to sparse)
    W_sparse = scipy.sparse.coo_array(W.numpy())

    # 3. Constrained sparse connection (each weight is its own group)
    # Create constraint matrix where each non-zero gets unique group ID
    constraint_data = []
    constraint_rows = []
    constraint_cols = []
    group_id = 1

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i, j] != 0:
                constraint_data.append(group_id)
                constraint_rows.append(i)
                constraint_cols.append(j)
                group_id += 1

    constraint = scipy.sparse.coo_array(
        (constraint_data, (constraint_rows, constraint_cols)), shape=W.shape
    )

    sparse_coo = SparseConn(
        W_sparse, bias=None, enforce_dale=False, sparse_backend=backend
    ).to(device)
    constrained = SparseConstrainedConn(
        W_sparse, constraint, enforce_dale=False, bias=None, sparse_backend=backend
    ).to(device)

    x = x.to(device)
    x_batch = x_batch.to(device)

    def _run_cudagraph(model, input_x):
        if use_compile:
            model = compile_or_skip(model)

        static_x = torch.zeros_like(input_x).requires_grad_(True)
        model.zero_grad(set_to_none=True)

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                out = model(static_x)
                out.sum().backward()
                static_x.grad.zero_()
                model.zero_grad()
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = model(static_x)
            static_out.sum().backward()

        # Replay with actual input
        static_x.data.copy_(input_x)
        g.replay()
        return static_out.clone(), static_x.grad.clone()

    # Forward pass without batch.
    out_dense, grad_dense = _run_cudagraph(dense, x)
    out_sparse, grad_sparse = _run_cudagraph(sparse_coo, x)
    out_constrained, grad_constrained = _run_cudagraph(constrained, x)

    # Check they're all the same for 1D input.
    torch.testing.assert_close(out_dense, out_sparse, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(out_dense, out_constrained, atol=1e-6, rtol=0.0)

    torch.testing.assert_close(grad_dense, grad_sparse, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(grad_dense, grad_constrained, atol=1e-6, rtol=0.0)

    # Forward pass with batch.
    out_dense_batch, grad_dense_batch = _run_cudagraph(dense, x_batch)
    out_sparse_batch, grad_sparse_batch = _run_cudagraph(sparse_coo, x_batch)
    out_constrained_batch, grad_constrained_batch = _run_cudagraph(constrained, x_batch)

    # Check they're all the same for batched input.
    torch.testing.assert_close(out_dense_batch, out_sparse_batch, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        out_dense_batch, out_constrained_batch, atol=1e-6, rtol=0.0
    )

    torch.testing.assert_close(grad_dense_batch, grad_sparse_batch, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        grad_dense_batch, grad_constrained_batch, atol=1e-6, rtol=0.0
    )


@pytest.mark.parametrize(
    "case",
    [
        {
            "name": "glif3_single_neuron",
            "stimulus": lambda steps: torch.cat(
                (torch.full((steps // 2,), 1000.0), torch.zeros((steps // 2,)))
            ),
            "build": lambda device: GLIF3(
                n_neuron=1,
                v_threshold=-45.0,
                v_reset=-65.0,
                c_m=200.0,
                tau=20.0,
                k=[0.05],
                asc_amps=[-50],
                tau_ref=2.0,
                step_mode="s",
                device=device,
            ),
        },
        {
            "name": "alif_single_neuron",
            "stimulus": lambda steps: torch.cat(
                (torch.full((steps // 2,), 1000.0), torch.full((steps // 2,), 8.0))
            ),
            "build": lambda device: ALIF(
                n_neuron=1,
                v_threshold=-50.0,
                v_reset=-65.0,
                c_m=1.0,
                g_leak=0.05,
                E_leak=-70.0,
                E_k=-80.0,
                g_k_init=0.0,
                tau_adapt=250.0,
                dg_k=0.12,
                tau_ref=2.0,
                step_mode="s",
                device=device,
            ),
        },
        {
            "name": "elif_single_neuron",
            "stimulus": lambda steps: torch.cat(
                (torch.full((steps // 2,), 12.0), torch.full((steps // 2,), 5.0))
            ),
            "build": lambda device: ELIF(
                n_neuron=1,
                v_threshold=-48.0,
                v_reset=-65.0,
                c_m=1.0,
                g_leak=0.05,
                E_leak=-70.0,
                E_k=-80.0,
                g_k_init=0.0,
                tau_adapt=150.0,
                dg_k=0.1,
                tau_ref=2.0,
                delta_T=2.0,
                v_T=-55.0,
                step_mode="s",
                device=device,
            ),
        },
    ],
    ids=lambda case: case["name"],
)
@pytest.mark.parametrize("use_compile", [False, True])
def test_cudagraph_neurons(case, use_compile: bool):
    """Neurons match dense behavior with CUDA graphs."""
    device = torch.device("cuda")

    def _simulate_traces(neuron, stimulus, with_cudagraph: bool):
        if with_cudagraph:
            actual_stimuli = torch.stack(
                [current.expand(neuron.n_neuron) for current in stimulus]
            )
            static_stimuli = torch.zeros_like(actual_stimuli)

            static_spikes = torch.empty_like(static_stimuli)
            static_vs = torch.empty_like(static_stimuli)
            static_adapts = torch.empty_like(static_stimuli)

            static_states = functional.named_memory_values(neuron)
            # Warmup
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    functional.set_memory_values(neuron, static_states)
                    for i in range(len(static_stimuli)):
                        neuron(static_stimuli[i])
            torch.cuda.current_stream().wait_stream(s)

            functional.set_memory_values(neuron, static_states)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_adapts.zero_()
                for i in range(len(static_stimuli)):
                    static_spikes[i] = neuron(static_stimuli[i])
                    static_vs[i] = neuron.v
                    adapt = getattr(neuron, "Iasc", getattr(neuron, "g_k", None))
                    if adapt is not None:
                        static_adapts[i] = adapt if adapt.ndim == 1 else adapt[..., 0]

            # Populate with actual input for the replay
            static_stimuli.copy_(actual_stimuli)
            g.replay()
            return {
                "spike": static_spikes.cpu(),
                "v": static_vs.cpu(),
                "adapt": static_adapts.cpu(),
            }
        else:
            spikes, vs, adapts = [], [], []
            for current in stimulus:
                spikes.append(neuron(current.expand(neuron.n_neuron)).detach().cpu())
                vs.append(neuron.v.detach().cpu())
                adapt = getattr(neuron, "Iasc", getattr(neuron, "g_k", None))
                adapts.append(adapt.detach().cpu() if adapt is not None else None)

            v_stack = torch.stack(vs)
            adapt_stack = torch.stack(
                [
                    torch.zeros_like(v_stack[0])
                    if a is None
                    else (a if a.ndim == 1 else a[..., 0])
                    for a in adapts
                ]
            )

            return {
                "spike": torch.stack(spikes),
                "v": v_stack,
                "adapt": adapt_stack,
            }

    dt = 1.0
    stimulus = case["stimulus"](100).to(device=device, dtype=torch.float32)

    with torch.no_grad(), environ.context(dt=dt):
        eager_neuron = case["build"](device)
        functional.init_net_state(eager_neuron, device=device, dtype=torch.float32)
        eager_traces = _simulate_traces(eager_neuron, stimulus, with_cudagraph=False)

        cudagraph_neuron = case["build"](device)
        functional.init_net_state(cudagraph_neuron, device=device, dtype=torch.float32)
        if use_compile:
            cudagraph_neuron = compile_or_skip(cudagraph_neuron)

        cudagraph_traces = _simulate_traces(
            cudagraph_neuron, stimulus, with_cudagraph=True
        )

    for key in ["spike", "v", "adapt"]:
        torch.testing.assert_close(
            eager_traces[key], cudagraph_traces[key], atol=1e-4, rtol=1e-4
        )
