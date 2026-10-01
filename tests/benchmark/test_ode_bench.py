import matplotlib.pyplot as plt
import pytest
import torch

from btorch.models.ode import euler_step, exp_euler_step
from btorch.utils.bench import do_bench
from btorch.utils.file import fig_path, save_fig


def _try_import_triton():
    """Safely import triton, returning None if not available."""
    try:
        import triton as tri

        return tri
    except ImportError:
        pytest.skip("Triton is not available, skipping benchmark.")


def test_numerically_close():
    # Logistic ODE: dx/dt = r x (1 - x)
    r = 1.0

    def f(x):
        return r * x * (1 - x)

    dt = 0.5
    T = 20
    steps = int(T / dt)

    x0 = torch.tensor(0.2)  # initial condition

    # --- numerical methods ---
    def for_loop_euler(x, n=steps):
        ret = torch.zeros(n)
        for i in range(n):
            x = euler_step(f, x, dt=dt)
            ret[i] = x
        return ret

    def for_loop_exp_euler(x, n=steps):
        ret = torch.zeros(n)
        for i in range(n):
            x = exp_euler_step(f, x, dt=dt)
            ret[i] = x
        return ret

    time = torch.arange(1, steps + 1) * dt
    euler_ret = for_loop_euler(x0)
    exp_euler_ret = for_loop_exp_euler(x0)

    # --- analytic solution ---
    K = 1.0
    exact_ret = K / (1 + ((K / x0) - 1) * torch.exp(-r * time))

    euler_err = torch.norm(euler_ret - exact_ret) / torch.norm(exact_ret)
    exp_euler_err = torch.norm(exp_euler_ret - exact_ret) / torch.norm(exact_ret)

    print(f"Relative error (Euler): {euler_err.item():.3e}")
    print(f"Relative error (Exp Euler): {exp_euler_err.item():.3e}")

    # --- plotting ---
    fig, ax = plt.subplots()

    ax.plot(time, euler_ret.detach(), label="Euler", linewidth=0.8)
    ax.plot(time, exp_euler_ret.detach(), label="Exp Euler", linestyle="--")
    ax.plot(
        time, exact_ret.detach(), label="Analytic exact", linestyle=":", linewidth=1.2
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("x(t)")
    ax.set_title("Logistic Equation: Numerical vs Analytic")
    ax.legend()
    save_fig(fig, "ode_solver_comparison")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA",
)
def benchmark_ode_solvers(shape=(512, 512), device="cuda"):
    # result: compiled versions of euler and exponent euler perform the same
    # in terms of speed/throughput and are generally 4-5 times faster
    # than non-compiled ones
    # ODE: dx/dt = -2.0*x + 3.0*y

    triton = _try_import_triton()

    def f(x, y):
        return -2.0 * x + 3.0 * y

    def vanilla_exp_euler(x, y, dt):
        dt = torch.tensor(dt, device=x.device)
        return torch.exp(-2.0 * dt) * x - torch.expm1(-2.0 * dt) / 2.0 * 3.0 * y

    dt = 0.1
    linear = torch.tensor([-2.0], device=device)

    # Step functions
    def euler_step_fn(x, y):
        return euler_step(f, x, y, dt=dt)

    def exp_euler_step_auto_fn(x, y):
        return exp_euler_step(f, x, y, dt=dt)

    def exp_euler_step_fn(x, y):
        return exp_euler_step(f, x, y, dt=dt, linear=linear)

    def vanilla_exp_euler_fn(x, y):
        return vanilla_exp_euler(x, y, dt=dt)

    # Compiled versions
    compiled_euler_step_fn = torch.compile(euler_step_fn)
    compiled_exp_euler_step_auto_fn = torch.compile(exp_euler_step_auto_fn)
    compiled_exp_euler_step_fn = torch.compile(exp_euler_step_fn)
    compiled_vanilla_exp_euler_fn = torch.compile(vanilla_exp_euler_fn)

    # Define benchmark functions - all take (x, y)
    bench_funcs = {
        "euler_step": euler_step_fn,
        "exp_euler_step_auto": exp_euler_step_auto_fn,
        "exp_euler_step": exp_euler_step_fn,
        "vanilla_exp_euler": vanilla_exp_euler_fn,
        "compiled_euler_step": compiled_euler_step_fn,
        "compiled_exp_euler_step_auto": compiled_exp_euler_step_auto_fn,
        "compiled_exp_euler_step": compiled_exp_euler_step_fn,
        "compiled_vanilla_exp_euler": compiled_vanilla_exp_euler_fn,
    }

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(12, 21, 1)],
            line_arg="provider",
            line_vals=list(bench_funcs.keys()),
            line_names=[
                "Euler",
                "Exp Euler Auto",
                "Exp Euler",
                "Vanilla Exp Euler",
                "Compiled Euler",
                "Compiled Exp Euler Auto",
                "Compiled Exp Euler",
                "Compiled Vanilla Exp Euler",
            ],
            styles=[
                ("red", "-"),
                ("blue", "-"),
                ("green", "-"),
                ("cyan", "-"),
                ("red", "--"),
                ("blue", "--"),
                ("green", "--"),
                ("cyan", "--"),
            ],
            ylabel="GB/s",
            plot_name="ode_solver_bandwidth",
            args={},
            x_log=True,
        )
    )
    def benchmark(N, provider):
        x0 = torch.rand((N,), device=device, dtype=torch.float32, requires_grad=True)
        y = torch.rand((N,), device=device, dtype=torch.float32)

        if provider not in bench_funcs:
            raise ValueError(f"Unknown provider: {provider}")

        # Perform one step of the selected solver
        ms = do_bench(
            lambda: bench_funcs[provider](x0.clone(), y),
            timing_method="total",
            quantiles=[0.5, 0.2, 0.8],
        )
        gbps = lambda ms: (2 * x0.numel() * x0.element_size()) / (ms * 1e-3) / 1e9
        return tuple(gbps(t) for t in ms)

    benchmark.run(
        show_plots=False,
        print_data=False,
        return_df=False,
        save_path=fig_path(),
    )
