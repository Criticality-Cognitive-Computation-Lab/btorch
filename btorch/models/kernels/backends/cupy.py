"""Execute directly generated cupy cell sequence kernels."""

from functools import partial

from .engine import backward as _backward, forward as _forward


forward = partial(_forward, "cupy")
backward = partial(_backward, "cupy")
