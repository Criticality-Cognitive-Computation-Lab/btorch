"""Execute directly generated triton cell sequence kernels."""

from functools import partial

from .engine import backward as _backward, forward as _forward


forward = partial(_forward, "triton")
backward = partial(_backward, "triton")
