"""Execute directly generated tilelang cell sequence kernels."""

from functools import partial

from .engine import backward as _backward, forward as _forward


forward = partial(_forward, "tilelang")
backward = partial(_backward, "tilelang")
