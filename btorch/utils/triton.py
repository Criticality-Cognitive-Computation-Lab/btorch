"""Prepare Triton executables for invocation-owned, stable argument layouts."""

from .kernel import PreparedLaunch


def prepare_launch(
    kernel, args, grid, constants, *, changing=(), writes=(), work=0, **options
):
    """Resolve an executable once for stable dtypes and trailing constexprs.

    Callers own tensor layouts and scalar widths throughout the invocation.
    ``changing`` identifies argument slots rebound by :class:`PreparedLaunch`.
    Kernels must declare any pointer alignment policy at their public
    ``@triton.jit`` declaration; preparation does not inspect or rebuild JIT
    objects. The compiled launch retains Triton's current-stream and profiler
    hooks.
    """
    launch, constexprs = _prepare_executable(
        kernel,
        args,
        grid,
        constants,
        writes=writes,
        work=work,
        options=options,
    )

    return PreparedLaunch(
        launch,
        list(args),
        tuple(changing),
        constexprs,
    )


def prepare_launch_recipe(
    kernel,
    args,
    grid,
    constants,
    *,
    writes=(),
    work=0,
    runtime_step_index=None,
    **options,
):
    """Prepare a tensor-free Triton launch recipe.

    This is the counterpart to :func:`prepare_launch` for sequence operators
    whose runtime arguments include a scalar extent such as ``n_steps``.
    The returned launch callable owns only Triton's executable metadata; the
    caller supplies all pointer arguments on every invocation.
    ``runtime_step_index`` identifies the dynamic sequence length, excluded
    from tuning specialization and limited to one step during trials.
    """
    return _prepare_executable(
        kernel,
        args,
        grid,
        constants,
        writes=writes,
        work=work,
        options=options,
        runtime_step_index=runtime_step_index,
    )


def _prepare_executable(
    kernel, args, grid, constants, *, writes, work, options, runtime_step_index=None
):
    from .autotune import triton_config

    options = triton_config(
        kernel,
        args,
        grid,
        constants,
        options,
        writes=writes,
        work=work,
        runtime_step_index=runtime_step_index,
    )
    executable = kernel.warmup(*args, grid=grid, **dict(constants), **options)
    launch = executable[grid + (1,) * (3 - len(grid))]
    constexprs = tuple(value for _name, value in constants)
    return launch, constexprs
