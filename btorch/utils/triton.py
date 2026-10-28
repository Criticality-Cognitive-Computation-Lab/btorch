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


def prepare_launch(kernel, args, grid, constants, *, changing=(), **options):
    """Resolve an executable once for stable dtypes and trailing constexprs.

    Callers own tensor layouts and scalar widths throughout the invocation.
    ``changing`` identifies pointers whose addresses can change alignment,
    such as sequence rows. Their first address must not specialize the binary.
    The compiled launch retains Triton's current-stream and profiler hooks.
    """
    if changing:
        kernel = _unaligned_kernel(kernel, tuple(changing))
    executable = kernel.warmup(*args, grid=grid, **dict(constants), **options)
    launch = executable[grid + (1,) * (3 - len(grid))]
    constexprs = tuple(value for _name, value in constants)

    return PreparedLaunch(
        launch,
        list(args),
        tuple(changing),
        constexprs,
    )
