"""Shared ordered exponential-Euler scalar operations."""

from ...ode import _EXP_EULER_SERIES_CUTOFF


def _lines(value: str):
    return tuple(line.strip() for line in value.strip().splitlines() if line.strip())


def _exp_euler_forward(prefix: str, linear: str, dt: str) -> tuple[str, ...]:
    """Emit the Module integrator's stable exponential-Euler coefficient."""
    return (
        f"{prefix}_z = {dt} * {linear}",
        (
            f"{prefix}_safe_linear = where(abs({prefix}_z) <= "
            f"{_EXP_EULER_SERIES_CUTOFF!r}, 1.0, {linear})"
        ),
        f"{prefix}_series_z = where(abs({prefix}_z) <= "
        f"{_EXP_EULER_SERIES_CUTOFF!r}, {prefix}_z, 0.0)",
        f"{prefix}_delta = expm1({prefix}_z)",
        f"{prefix}_ratio = {prefix}_delta / {prefix}_safe_linear",
        (
            f"{prefix}_exprel = 1.0 + {prefix}_series_z * (0.5 + "
            f"{prefix}_series_z * (1.0 / 6.0 + "
            f"{prefix}_series_z * (1.0 / 24.0 + {prefix}_series_z * (1.0 "
            f"/ 120.0 + {prefix}_series_z * "
            f"(1.0 / 720.0 + {prefix}_series_z / 5040.0)))))"
        ),
        (
            f"{prefix}_phi = where(abs({prefix}_z) <= {_EXP_EULER_SERIES_CUTOFF!r}, "
            f"{dt} * {prefix}_exprel, {prefix}_ratio)"
        ),
    )


def _exp_euler_backward(prefix: str, linear: str, dt: str) -> tuple[str, ...]:
    """Emit a stable coefficient VJP while preserving the far-field graph."""
    return (
        (
            f"grad_{prefix}_ratio = where(abs({prefix}_z) <= "
            f"{_EXP_EULER_SERIES_CUTOFF!r}, 0.0, grad_{prefix}_phi)"
        ),
        f"grad_{prefix}_delta = grad_{prefix}_ratio / {prefix}_safe_linear",
        f"grad_{prefix}_ratio_again = {prefix}_delta / {prefix}_safe_linear",
        (
            f"grad_{prefix}_ratio_square = grad_{prefix}_ratio_again / "
            f"{prefix}_safe_linear"
        ),
        (f"grad_{prefix}_safe = -grad_{prefix}_ratio * grad_{prefix}_ratio_square"),
        (
            f"grad_{prefix}_linear_safe = where({linear} == 0.0, 0.0, "
            f"grad_{prefix}_safe)"
        ),
        (
            f"grad_{prefix}_linear_delta = (grad_{prefix}_delta * ({prefix}_delta + "
            f"1.0)) * {dt}"
        ),
        (
            f"grad_{prefix}_linear_direct = grad_{prefix}_linear_delta + "
            f"grad_{prefix}_linear_safe"
        ),
        (
            f"{prefix}_dexprel = 0.5 + {prefix}_series_z * (1.0 / 3.0 + "
            f"{prefix}_series_z * (1.0 "
            f"/ 8.0 + {prefix}_series_z * (1.0 / 30.0 + {prefix}_series_z "
            f"* (1.0 / 144.0 + "
            f"{prefix}_series_z / 840.0))))"
        ),
        (
            f"grad_{prefix}_linear_series = grad_{prefix}_phi * {dt} * {dt} * "
            f"{prefix}_dexprel"
        ),
        (
            f"grad_{prefix}_linear = where(abs({prefix}_z) <= "
            f"{_EXP_EULER_SERIES_CUTOFF!r}, grad_{prefix}_linear_series, "
            f"grad_{prefix}_linear_direct)"
        ),
    )


def empty_stage(*args):
    """Return an empty optional mathematical stage."""
    return ()
