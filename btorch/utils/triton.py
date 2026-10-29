"""Prepare Triton executables for invocation-owned, stable argument layouts."""

import functools

from .kernel import PreparedLaunch


@functools.lru_cache(maxsize=128)
def _unaligned_kernel(kernel, changing):
    import triton

    return triton.jit(
        kernel.fn,
        do_not_specialize=kernel.do_not_specialize,
        do_not_specialize_on_alignment=list(
            set(kernel.do_not_specialize_on_alignment)
            | {kernel.arg_names[index] for index in changing}
        ),
    )


def prepare_launch(
    kernel, args, grid, constants, *, changing=(), writes=(), work=0, **options
):
    """Resolve an executable once for stable dtypes and trailing constexprs.

    Callers own tensor layouts and scalar widths throughout the invocation.
    ``changing`` identifies pointers whose addresses can change alignment,
    such as sequence rows. Their first address must not specialize the binary.
    The compiled launch retains Triton's current-stream and profiler hooks.
    """
    launch, constexprs = _prepare_executable(
        kernel,
        args,
        grid,
        constants,
        changing=changing,
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
    kernel, args, grid, constants, *, changing=(), writes=(), work=0, **options
):
    """Prepare a tensor-free Triton launch recipe.

    This is the counterpart to :func:`prepare_launch` for sequence operators
    whose runtime arguments include a scalar extent such as ``n_steps``.
    The returned launch callable owns only Triton's executable metadata; the
    caller supplies all pointer arguments on every invocation.
    """
    return _prepare_executable(
        kernel,
        args,
        grid,
        constants,
        changing=changing,
        writes=writes,
        work=work,
        options=options,
    )


def _prepare_executable(
    kernel, args, grid, constants, *, changing, writes, work, options
):
    if changing:
        kernel = _unaligned_kernel(kernel, tuple(changing))
    from .autotune import triton_config

    options = triton_config(
        kernel,
        args,
        grid,
        constants,
        options,
        writes=writes,
        work=work,
    )
    executable = kernel.warmup(*args, grid=grid, **dict(constants), **options)
    launch = executable[grid + (1,) * (3 - len(grid))]
    constexprs = tuple(value for _name, value in constants)
    return launch, constexprs
